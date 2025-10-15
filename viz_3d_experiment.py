import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

from utilities3 import UnitGaussianNormalizer, LpLoss

plt.switch_backend('agg')


# -------------------------------
# Robust loading/inference helpers
# -------------------------------
def _load_state_robust(model_path, device):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    return state


def _infer_fno_hparams_from_state(state_dict):
    """Infer (width, modes1, modes2, modes3, in_channels_total) from state_dict.
    支持两类权重：
      - 标准FNO的 conv0.weights (5D)
      - LowRankFNO 的 conv0.B (4D, rank,m1,m2,m3)
    宽度以 fc0.weight 的第0维为准；总输入维度以 fc0.weight 第1维为准。
    """
    # in_channels from fc0
    in_channels_total = None
    width_from_fc0 = None
    if 'fc0.weight' in state_dict and isinstance(state_dict['fc0.weight'], torch.Tensor):
        width_from_fc0 = int(state_dict['fc0.weight'].shape[0])
        in_channels_total = int(state_dict['fc0.weight'].shape[1])

    candidates = ['conv0.weights', 'conv0.weights1', 'conv0.weights2']
    for k in state_dict.keys():
        for pat in candidates:
            if k.endswith(pat):
                w = state_dict[k]
                if isinstance(w, torch.Tensor) and w.ndim == 5 and w.shape[0] == w.shape[1]:
                    width = width_from_fc0 if width_from_fc0 is not None else int(w.shape[0])
                    return width, (w.shape[-3], w.shape[-2], w.shape[-1]), in_channels_total
    # Low-rank: conv0.B
    if 'conv0.B' in state_dict and isinstance(state_dict['conv0.B'], torch.Tensor):
        b = state_dict['conv0.B']  # (rank,m1,m2,m3)
        width = width_from_fc0 if width_from_fc0 is not None else 32
        return width, (int(b.shape[-3]), int(b.shape[-2]), int(b.shape[-1])), in_channels_total
    # Fallback: any 5D weight with square first two dims
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 5 and v.shape[0] == v.shape[1]:
            width = width_from_fc0 if width_from_fc0 is not None else int(v.shape[0])
            return width, (v.shape[-3], v.shape[-2], v.shape[-1]), in_channels_total
    raise RuntimeError('无法从 state_dict 推断 FNO 的 width/modes/in_channels_total')


def _build_and_load_model(tag, ctor, model_path, device, T_in, T_out, fallback_modes=6, fallback_width=20, has_coords=True):
    state = _load_state_robust(model_path, device)
    try:
        width, (m1, m2, m3), in_needed_total = _infer_fno_hparams_from_state(state)
    except Exception:
        width, (m1, m2, m3), in_needed_total = fallback_width, (fallback_modes,)*3, (T_in+1 + (3 if has_coords else 0))

    # fc0 对于 FNO / FNO-RC 都是 Linear(in_channels + 3, width)，
    # 因此 state_dict 中的 fc0.weight.shape[1] 是总输入特征(in_channels+3)。
    if in_needed_total is None:
        in_channels_effective = (T_in + 1)
    else:
        # 若 has_coords=True（仅 RC），fc0 接受 (in_channels + 3)；否则就是 in_channels 本身
        in_channels_effective = max(1, int(in_needed_total) - (3 if has_coords else 0))
    model = ctor(m1, m2, m3, width, in_channels=in_channels_effective, out_channels=1).to(device)
    # 对 FNO 使用严格加载，避免部分权重未加载导致误差异常
    strict_flag = True if tag in ['FNO', 'U-FNO'] else False
    missing, unexpected = model.load_state_dict(state, strict=strict_flag)
    if missing or unexpected:
        print(f'[{tag}] non-strict load: missing={list(missing)}, unexpected={list(unexpected)}')
    model.eval()
    return model, in_channels_effective


# -------------------------------
# Data helpers
# -------------------------------
def build_windows(series, T_in, T_out, stride, max_windows_per_sample=None):
    """series: (N,S1,S2,T_total) -> X:(M,S1,S2,T_in), Y:(M,S1,S2,T_out), starts(M,), T_total(int)"""
    N, S1, S2, T_total = series.shape
    xs, ys, starts = [], [], []
    for n in range(N):
        positions = list(range(0, max(1, T_total - (T_in + T_out) + 1), stride))
        if max_windows_per_sample is not None:
            positions = positions[:max_windows_per_sample]
        for t0 in positions:
            xs.append(series[n, :, :, t0:t0+T_in])
            ys.append(series[n, :, :, t0+T_in:t0+T_in+T_out])
            starts.append(t0)
    X = torch.from_numpy(np.stack(xs, 0)).float()   # (M,S1,S2,T_in)
    Y = torch.from_numpy(np.stack(ys, 0)).float()   # (M,S1,S2,T_out)
    starts = torch.tensor(starts, dtype=torch.float32)
    return X, Y, starts, T_total


def make_input_with_abs_time(x_win, t0, T_total, T_in, T_out, a_norm, device, in_channels_needed, encode_input=True):
    """x_win: (S1,S2,T_in) CPU -> (1,S1,S2,1,C) repeated to T_out, where C=in_channels_needed.
       encode_input=True: 使用 a_norm.encode 与 RC 一致；False: 保持原始输入（FNO 一致）。
       若 C==T_in 则不拼 t_abs；若 C==T_in+1 则拼接；若更大则在末尾 0-pad 到 C。
    """
    x = a_norm.encode(x_win.unsqueeze(0)) if encode_input else x_win.unsqueeze(0)  # (1,S1,S2,T_in) CPU
    S1, S2 = x.shape[1], x.shape[2]
    channels = x.shape[-1]
    if in_channels_needed is None:
        in_channels_needed = channels + 1
    if in_channels_needed == channels:
        pass
    elif in_channels_needed == channels + 1:
        t_abs_val = float(t0/float(T_total))
        t_abs = torch.full((1, S1, S2, 1), t_abs_val, dtype=x.dtype)
        x = torch.cat([x, t_abs], dim=-1)
    elif in_channels_needed > channels + 1:
        t_abs_val = float(t0/float(T_total))
        t_abs = torch.full((1, S1, S2, 1), t_abs_val, dtype=x.dtype)
        pad_zeros = torch.zeros((1, S1, S2, in_channels_needed - (channels + 1)), dtype=x.dtype)
        x = torch.cat([x, t_abs, pad_zeros], dim=-1)
    else:
        # in_needed < channels：截断到所需通道
        x = x[..., :in_channels_needed]
    x = x.to(device)
    x5 = x.reshape(1, S1, S2, 1, in_channels_needed).repeat(1, 1, 1, T_out, 1)
    return x5


# -------------------------------
# Main visualization
# -------------------------------
def viz_3d(models, data_path, save_dir, T_in=10, T_out=20, modes=6, width=20, sample_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_func = LpLoss(size_average=True)

    # Load (S1,S2,T,N) -> (N,S1,S2,T)
    with h5py.File(data_path, 'r') as f:
        u_np = f['u'][()]
    u_np = np.transpose(u_np, (3, 0, 1, 2))

    # Split sequences
    ntrain, ntest = 40, 10
    train_series = u_np[:ntrain]
    test_series  = u_np[-ntest:]

    # Sequential blocks, stride=T_out
    train_a, train_u, train_starts, train_T = build_windows(train_series, T_in, T_out, stride=T_out)
    test_a,  test_u,  test_starts,  test_T  = build_windows(test_series,  T_in, T_out, stride=T_out)

    # Normalizers (use training-time saved stats to ensure SAME baseline)
    a_norm = UnitGaussianNormalizer(train_a)  # input norm用现场拟合即可
    # 尝试优先加载 RC/FNO 对应的 y_normalizer 统计量
    yn_rc_path  = '/content/models/rc_y_normalizer.pt'
    yn_fno_path = '/content/drive/MyDrive/fno_models/fno_y_normalizer.pt'
    # 默认使用 RC 的统计量（与 RC 一致），FNO 在 norm 度量时也使用同一份统计以保证比较一致
    yn_stats = None
    for p in [yn_rc_path, yn_fno_path]:
        if os.path.exists(p):
            yn_stats = torch.load(p, map_location='cpu')
            print(f"[info] Loaded y_normalizer stats from {p}")
            break
    if yn_stats is None:
        # 兜底：用训练窗口拟合（可能与训练时略有偏差）
        print('[warn] y_normalizer stats not found; falling back to fit on train windows')
        y_norm_cpu = UnitGaussianNormalizer(train_u)
    else:
        y_norm_cpu = UnitGaussianNormalizer(train_u)
        y_norm_cpu.mean = yn_stats['mean']
        y_norm_cpu.std  = yn_stats['std']
    y_norm_gpu = UnitGaussianNormalizer(train_u)
    y_norm_gpu.load_state_dict(y_norm_cpu.state_dict())
    y_norm_gpu.to(device)

    # Build models (robust load)
    built_models = []
    for tag, ctor, path in models:
        # FNO / FNO-RC / AFNO 的 fc0 输入包含坐标通道（+3）；U-FNO / LowRank 不拼
        has_coords = (tag in ['FNO', 'FNO-RC', 'AFNO'])
        model, in_ch_eff = _build_and_load_model(tag, ctor, path, device, T_in, T_out,
                                                fallback_modes=modes, fallback_width=width,
                                                has_coords=has_coords)
        built_models.append((tag, model, in_ch_eff))

    # Aggregate errors over all test windows (norm/raw)
    per_model_norm = {tag: [] for tag, _, _ in built_models}
    per_model_raw  = {tag: [] for tag, _, _ in built_models}

    for i in range(test_a.shape[0]):
        x_win = test_a[i]                     # (S1,S2,T_in)
        y_win = test_u[i].unsqueeze(0)        # (1,S1,S2,T_out)
        y_win = y_win.to(device)
        y_win_norm = y_norm_cpu.encode(y_win.cpu()).to(device)

        with torch.no_grad():
            for tag, model, in_ch_eff in built_models:
                # 与训练一致：仅 FNO-RC 做输入归一化；FNO 与 U-FNO 均不做
                encode_in = (tag == 'FNO-RC')
                x5 = make_input_with_abs_time(x_win, test_starts[i].item(), test_T, T_in, T_out, a_norm, device, in_ch_eff, encode_input=encode_in)
                y_out = model(x5).squeeze(-1)                  # (1,S1,S2,T_out)
                if tag == 'FNO':
                    # FNO 基线输出已在原始空间
                    y_pred_raw = y_out
                    y_pred_norm_for_metric = y_norm_gpu.encode(y_out)
                elif tag == 'FNO-RC':
                    # FNO-RC 输出在归一化空间
                    y_pred_norm_for_metric = y_out
                    y_pred_raw = y_norm_gpu.decode(y_out)
                else:  # U-FNO 输出在原始空间
                    y_pred_raw = y_out
                    y_pred_norm_for_metric = y_norm_gpu.encode(y_out)

                # collect per-time-step
                norm_ts = []
                raw_ts  = []
                for t in range(T_out):
                    norm_ts.append(loss_func(y_pred_norm_for_metric[..., t], y_win_norm[..., t]).item())
                    raw_ts.append(loss_func(y_pred_raw[..., t],  y_win[..., t]).item())
                per_model_norm[tag].append(norm_ts)
                per_model_raw[tag].append(raw_ts)

    def agg(stat_dict):
        out = {}
        for tag, rows in stat_dict.items():
            arr = np.array(rows)                 # (num_windows,T_out)
            out[tag] = (arr.mean(0), arr.std(0))
        return out

    norm_mean = agg(per_model_norm)
    raw_mean  = agg(per_model_raw)

    # Plot raw mean±std error curve
    ts = np.arange(T_in, T_in+T_out)
    plt.figure(figsize=(10,6))
    for tag, (mu, sig) in raw_mean.items():
        plt.plot(ts, mu, label=f'{tag} (final={mu[-1]:.4f})')
        plt.fill_between(ts, mu-sig, mu+sig, alpha=0.2)
    plt.yscale('log'); plt.xlabel('Time Step'); plt.ylabel('Relative L2 (log)')
    plt.title('Prediction Error vs Time (Raw, mean±std over test windows)')
    plt.grid(True, which='both', ls='--'); plt.legend()
    plt.savefig(os.path.join(save_dir, 'error_vs_time_raw_mean.png')); plt.close()

    # Plot norm mean±std (optional)
    plt.figure(figsize=(10,6))
    for tag, (mu, sig) in norm_mean.items():
        plt.plot(ts, mu, label=f'{tag} (final={mu[-1]:.4f})')
        plt.fill_between(ts, mu-sig, mu+sig, alpha=0.2)
    plt.yscale('log'); plt.xlabel('Time Step'); plt.ylabel('Relative L2 (log)')
    plt.title('Prediction Error vs Time (Norm, mean±std over test windows)')
    plt.grid(True, which='both', ls='--'); plt.legend()
    plt.savefig(os.path.join(save_dir, 'error_vs_time_norm_mean.png')); plt.close()

    # -------------------
    # Single-sample visuals
    # -------------------
    # Rebuild input for a chosen test window
    x_win = test_a[sample_idx]
    y_win = test_u[sample_idx].unsqueeze(0)
    y_win = y_win.to(device)

    preds_single = {}
    with torch.no_grad():
        for tag, model, in_ch_eff in built_models:
            encode_in = (tag == 'FNO-RC')
            x5 = make_input_with_abs_time(x_win, test_starts[sample_idx].item(), test_T, T_in, T_out, a_norm, device, in_ch_eff, encode_input=encode_in)
            y_out = model(x5).squeeze(-1)
            if tag == 'FNO':
                y_pred_raw = y_out.cpu()
            elif tag == 'FNO-RC':
                y_pred_raw = y_norm_gpu.decode(y_out).cpu()
            else:  # U-FNO
                y_pred_raw = y_out.cpu()
            preds_single[tag] = y_pred_raw

    # Final slice (Raw)
    final_t = -1
    gt = y_win[0, :, :, final_t].detach().cpu().numpy()
    S1, S2 = gt.shape[0], gt.shape[1]
    vmax, vmin = float(gt.max()), float(gt.min())
    fig, axes = plt.subplots(1, len(preds_single)+1, figsize=(6*(len(preds_single)+1), 5))
    im0 = axes[0].imshow(gt, cmap='viridis', vmax=vmax, vmin=vmin); axes[0].set_title('Ground Truth'); plt.colorbar(im0, ax=axes[0])
    for i, (tag, vol) in enumerate(preds_single.items(), start=1):
        im = axes[i].imshow(vol[0, :, :, final_t], cmap='viridis', vmax=vmax, vmin=vmin)
        plt.colorbar(im, ax=axes[i]); axes[i].set_title(f'{tag}')
        axes[i].set_xlabel('x'); axes[i].set_ylabel('y')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'final_slice.png')); plt.close()

    # 3D isosurface (on Raw volume)
    def plot_iso(ax, vol, ttl, level):
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()
        if vol.ndim != 3 or vol.shape[-1] < 2:
            ax.set_title(f'{ttl} (skip iso: not 3D)'); return
        verts, faces, _, _ = marching_cubes(vol, level=level)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='viridis', lw=1)
        ax.set_title(ttl); ax.set_xlabel('x'); ax.set_ylabel('y')

    vol_gt = y_win[0].detach().cpu().numpy()
    level = float(vol_gt.mean())
    fig = plt.figure(figsize=(7*(len(preds_single)+1), 7))
    ax = fig.add_subplot(1, len(preds_single)+1, 1, projection='3d'); plot_iso(ax, vol_gt, 'GT', level)
    for i, (tag, vol) in enumerate(preds_single.items(), start=2):
        ax = fig.add_subplot(1, len(preds_single)+1, i, projection='3d'); plot_iso(ax, vol[0], tag, level)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'final_iso.png')); plt.close()

    # -------------------
    # Overall solution views (more intuitive)
    # -------------------
    # 1) Maximum-Intensity Projection (MIP) over time (Raw)
    mip_gt = vol_gt.max(axis=2)
    fig, axes = plt.subplots(1, len(preds_single)+1, figsize=(6*(len(preds_single)+1), 5))
    im0 = axes[0].imshow(mip_gt, cmap='viridis'); axes[0].set_title('GT MIP (time)'); plt.colorbar(im0, ax=axes[0])
    for i, (tag, vol) in enumerate(preds_single.items(), start=1):
        mip = vol[0].numpy().max(axis=2)
        im = axes[i].imshow(mip, cmap='viridis'); plt.colorbar(im, ax=axes[i]); axes[i].set_title(f'{tag} MIP (time)')
    for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'mip_time.png')); plt.close()

    # 2) Mean over time (Raw)
    mean_gt = vol_gt.mean(axis=2)
    fig, axes = plt.subplots(1, len(preds_single)+1, figsize=(6*(len(preds_single)+1), 5))
    im0 = axes[0].imshow(mean_gt, cmap='viridis'); axes[0].set_title('GT Mean (time)'); plt.colorbar(im0, ax=axes[0])
    for i, (tag, vol) in enumerate(preds_single.items(), start=1):
        meanv = vol[0].numpy().mean(axis=2)
        im = axes[i].imshow(meanv, cmap='viridis'); plt.colorbar(im, ax=axes[i]); axes[i].set_title(f'{tag} Mean (time)')
    for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'mean_time.png')); plt.close()

    # -------------------
    # Error distribution & spatial error maps (Raw)
    # -------------------
    fig, axes = plt.subplots(2, len(preds_single), figsize=(6*len(preds_single), 8))
    for j, (tag, vol) in enumerate(preds_single.items()):
        pred = vol[0].numpy()                      # (S1,S2,T_out)
        err = np.abs(pred - vol_gt)               # abs error
        # Histogram (flatten all voxels)
        axh = axes[0, j]
        axh.hist(err.flatten(), bins=80, log=True, color='tab:orange', alpha=0.8)
        axh.set_title(f'{tag} Error Histogram (Raw)')
        axh.set_xlabel('|pred-gt|'); axh.set_ylabel('count (log)')
        # Time-averaged spatial error
        axm = axes[1, j]
        im = axm.imshow(err.mean(axis=2), cmap='magma')
        axm.set_title(f'{tag} Mean |Error| (time)')
        plt.colorbar(im, ax=axm)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'error_distribution_and_maps.png')); plt.close()

    # -------------------
    # Dynamic flow animation (Raw)
    # -------------------
    try:
        import matplotlib.animation as animation
        # Prepare frames for GT and each model
        names = ['GT'] + [k for k in preds_single.keys()]
        vols = [vol_gt] + [v[0].numpy() for _, v in preds_single.items()]  # list of (S1,S2,T_out)
        S1, S2, TT = vols[0].shape
        fig, axes = plt.subplots(1, len(vols), figsize=(5*len(vols), 5))
        ims = []
        vmin = min(v.min() for v in vols)
        vmax = max(v.max() for v in vols)
        for ax, name in zip(axes, names):
            ax.set_title(name); ax.set_axis_off()
        # init images
        imgs = [axes[i].imshow(vols[i][...,0], cmap='viridis', vmin=vmin, vmax=vmax, animated=True) for i in range(len(vols))]
        def update(t):
            for i in range(len(vols)):
                imgs[i].set_array(vols[i][..., t])
            return imgs
        ani = animation.FuncAnimation(fig, update, frames=TT, interval=200, blit=True)
        out_path = os.path.join(save_dir, 'dynamic_flow.mp4')
        try:
            ani.save(out_path, writer='ffmpeg', dpi=120)
        except Exception:
            ani.save(os.path.join(save_dir, 'dynamic_flow.gif'), writer='imagemagick', dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f'[warn] 动态流场动画生成失败: {e}')


if __name__ == '__main__':
    import argparse
    from fourier_3d_clean import FNO3d as FNO_ctor
    from fourier_3d_cft_residual import FNO_RC_3D as RC_ctor
    from fourier_3d_unet import U_FNO_3d as UFNO_ctor
    from fourier_3d_lowrank import LowRankFNO3d as LRFNO_ctor
    from fourier_3d_afno import AFNO3D as AFNO_ctor

    parser = argparse.ArgumentParser(description='3D Visualization for FNO vs FNO-RC')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--save_dir', type=str, default='/content/viz_plots')
    parser.add_argument('--rc_path', type=str, default='/content/models/fno_rc_3d_seq.pt')
    parser.add_argument('--fno_path', type=str, default='/content/drive/MyDrive/fno_models/fno_3d_standard.pt')
    parser.add_argument('--ufno_path', type=str, default='/content/models/u_fno_3d.pt')
    parser.add_argument('--lowrank_path', type=str, default='/content/models/lowrank_fno_3d.pt')
    parser.add_argument('--afno_path', type=str, default='/content/models/afno_3d.pt')
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_out', type=int, default=20)
    parser.add_argument('--modes', type=int, default=6)
    parser.add_argument('--width', type=int, default=20)
    parser.add_argument('--sample_idx', type=int, default=0)
    args = parser.parse_args()

    models = [
        ('FNO-RC',   RC_ctor,   args.rc_path),
        ('FNO',      FNO_ctor,  args.fno_path),
        ('U-FNO',    UFNO_ctor, args.ufno_path),
        ('LowRank',  LRFNO_ctor, args.lowrank_path),
        ('AFNO',     AFNO_ctor, args.afno_path)
    ]

    viz_3d(models,
           data_path=args.data_path,
           save_dir=args.save_dir,
           T_in=args.T_in, T_out=args.T_out, modes=args.modes, width=args.width,
           sample_idx=args.sample_idx)


