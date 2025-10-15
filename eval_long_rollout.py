import os
import h5py
import torch
import numpy as np
import argparse
from typing import Tuple

from utilities3 import UnitGaussianNormalizer, LpLoss
from fourier_3d_clean import FNO3d as FNO_ctor
from fourier_3d_cft_residual import FNO_RC_3D as RC_ctor


def _load_state_robust(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    return state


def _infer_hparams(sd):
    in_total = None
    width = None
    if 'fc0.weight' in sd:
        w = sd['fc0.weight']
        if isinstance(w, torch.Tensor):
            width = int(w.shape[0])
            in_total = int(w.shape[1])
    # try 5D spectral weight to get modes
    m1 = m2 = m3 = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 5 and v.shape[0] == v.shape[1]:
            if v.shape[-3] == 1 and v.shape[-2] == 1 and v.shape[-1] == 1:
                continue
            m1, m2, m3 = int(v.shape[-3]), int(v.shape[-2]), int(v.shape[-1])
            if width is None:
                width = int(v.shape[0])
            break
    if m1 is None:
        m1 = m2 = m3 = 6
    return width, (m1, m2, m3), in_total


def _build_model(tag: str, ctor, path: str, device, T_in: int, has_coords: bool):
    sd = _load_state_robust(path, device)
    width, (m1, m2, m3), in_total = _infer_hparams(sd)
    if in_total is None:
        in_effective = T_in + 1
    else:
        in_effective = max(1, int(in_total) - (3 if has_coords else 0))
    model = ctor(m1, m2, m3, width, in_channels=in_effective, out_channels=1).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, in_effective, (m1, m2, m3)


def _spectral_resize_batch(x_spatial: torch.Tensor, target_res: int) -> torch.Tensor:
    # x_spatial: (B, H, W)
    B, H, W = x_spatial.shape
    F = torch.fft.fftshift(torch.fft.fft2(x_spatial), dim=(-2, -1))
    # rows
    if target_res > H:
        pad_h = target_res - H
        phb = pad_h // 2
        pha = pad_h - phb
        F = torch.nn.functional.pad(F, (0, 0, phb, pha))
    elif target_res < H:
        ch = H - target_res
        chb = ch // 2
        cha = ch - chb
        F = F[:, chb:H-cha, :]
    # cols
    H2, W2 = F.shape[-2], F.shape[-1]
    if target_res > W2:
        pad_w = target_res - W2
        pwb = pad_w // 2
        pwa = pad_w - pwb
        F = torch.nn.functional.pad(F, (pwb, pwa, 0, 0))
    elif target_res < W2:
        cw = W2 - target_res
        cwb = cw // 2
        cwa = cw - cwb
        F = F[:, :, cwb:W2-cwa]
    x_res = torch.fft.ifft2(torch.fft.ifftshift(F, dim=(-2, -1))).real
    x_res = x_res * (target_res / H) * (target_res / W)
    return x_res


def resample_series(series_np, target_res, mode: str = 'spectral'):
    N, S1, S2, T = series_np.shape
    if mode == 'bilinear':
        x = torch.from_numpy(series_np).float().unsqueeze(1)
        x = x.permute(0, 4, 1, 2, 3)
        x_rs = torch.nn.functional.interpolate(x.reshape(N*T,1,S1,S2), size=(target_res,target_res), mode='bilinear', align_corners=False)
        x_rs = x_rs.reshape(N, T, 1, target_res, target_res).permute(0,3,4,1,2).squeeze(-1)
        return x_rs
    else:
        x = torch.from_numpy(series_np).float().permute(0,3,1,2)  # (N,T,S1,S2)
        NT = N*T
        x_flat = x.reshape(NT, x.shape[-2], x.shape[-1])
        x_rs = _spectral_resize_batch(x_flat, target_res)
        x_rs = x_rs.reshape(N, T, target_res, target_res).permute(0,2,3,1)
        return x_rs


def make_input(x_win, T_out, device, in_effective):
    # x_win: (S1,S2,T_in+1) already contains abs-time if desired; here我们保持简洁：仅通道维齐平
    S1, S2, C = x_win.shape
    if C < in_effective:
        pad = torch.zeros(S1, S2, in_effective - C)
        x_win = torch.cat([x_win, pad], dim=-1)
    elif C > in_effective:
        x_win = x_win[..., :in_effective]
    x = x_win.unsqueeze(0)  # (1,S1,S2,C)
    x5 = x.reshape(1, S1, S2, 1, in_effective).repeat(1, 1, 1, T_out, 1)
    return x5.to(device)


def main():
    parser = argparse.ArgumentParser(description='Long-horizon rollout evaluation (auto-regressive, RC/FNO)')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--save_dir', type=str, default='/content/rollout')
    parser.add_argument('--rc_path', type=str, default='/content/models/fno_rc_3d_seq_multires.pt')
    parser.add_argument('--fno_path', type=str, default='/content/drive/MyDrive/fno_models/fno_3d_standard.pt')
    parser.add_argument('--target_res', type=int, default=96)
    parser.add_argument('--resample_mode', type=str, default='spectral', choices=['spectral','bilinear'])
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--step_out', type=int, default=20, help='model multi-step output used per rollout iteration')
    parser.add_argument('--rollout_T', type=int, default=100)
    parser.add_argument('--ntest', type=int, default=5)
    parser.add_argument('--rc_disable', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_func = LpLoss(size_average=True)

    # load data (N,S,S,T)
    with h5py.File(args.data_path, 'r') as f:
        u = f['u'][()]
    u = np.transpose(u, (3,1,2,0))
    test_series = u[-args.ntest:]
    test_series = resample_series(test_series, args.target_res, mode=args.resample_mode)

    # build models
    rc_model, rc_in_eff, rc_modes = _build_model('FNO-RC', RC_ctor, args.rc_path, device, args.T_in, has_coords=True)
    if args.rc_disable:
        for name in ['conv0','conv1','conv2','conv3']:
            if hasattr(rc_model, name):
                getattr(rc_model, name).enable_correction = False
    fno_model, fno_in_eff, fno_modes = _build_model('FNO', FNO_ctor, args.fno_path, device, args.T_in, has_coords=True)

    # ensure per-iteration temporal chunk length is large enough for spectral modes
    # rfftn along time gives size D//2+1; to index [:modes3], need D//2+1 >= modes3 => D >= 2*(modes3-1)
    # use a safe margin: chunk_len = max(step_out, 2*modes3)
    rc_chunk_len = max(int(args.step_out), int(2 * max(1, rc_modes[2])))
    fno_chunk_len = max(int(args.step_out), int(2 * max(1, fno_modes[2])))

    # rollout per sample
    results = {'FNO-RC': [], 'FNO': []}
    for n in range(test_series.shape[0]):
        series = test_series[n].to(device)  # (S,S,T) on device
        S1, S2, Ttot = series.shape
        # initial context
        ctx = series[..., :args.T_in]  # (S,S,T_in) on device
        # add dummy abs-time channel = 0 (不使用时间编码，保持与 cross-res 一致)
        x_win = torch.cat([ctx, torch.zeros(S1, S2, 1, dtype=series.dtype, device=series.device)], dim=-1)
        H_eff = int(min(args.rollout_T, max(0, Ttot - args.T_in)))
        if H_eff == 0:
            continue
        gt_tail = series[..., args.T_in:args.T_in+H_eff].float()  # (S,S,H_eff) on device

        # RC rollout
        pred = []
        cur = x_win.clone()
        t_left = H_eff
        while t_left > 0:
            step = min(args.step_out, t_left)
            x5 = make_input(cur, rc_chunk_len, device, rc_in_eff)
            out = rc_model(x5).squeeze(-1)[..., :step]  # (1,S1,S2,step)
            pred.append(out[0])
            # update cur: take last T_in from concatenated prev context + new out
            cur_seq = torch.cat([cur[..., :args.T_in], out[0]], dim=-1)  # (S,S,T_in+step)
            new_ctx = cur_seq[..., -args.T_in:]
            cur = torch.cat([new_ctx, torch.zeros(S1, S2, 1, dtype=new_ctx.dtype, device=new_ctx.device)], dim=-1)
            t_left -= step
        pred_rc = torch.cat(pred, dim=-1)  # (S,S,H_eff)
        e_rc = loss_func(pred_rc.unsqueeze(0), gt_tail.unsqueeze(0)).item()
        results['FNO-RC'].append(e_rc)

        # FNO rollout
        pred = []
        cur = x_win.clone()
        t_left = H_eff
        while t_left > 0:
            step = min(args.step_out, t_left)
            x5 = make_input(cur, fno_chunk_len, device, fno_in_eff)
            out = fno_model(x5).squeeze(-1)[..., :step]
            pred.append(out[0])
            cur_seq = torch.cat([cur[..., :args.T_in], out[0]], dim=-1)
            new_ctx = cur_seq[..., -args.T_in:]
            cur = torch.cat([new_ctx, torch.zeros(S1, S2, 1, dtype=new_ctx.dtype, device=new_ctx.device)], dim=-1)
            t_left -= step
        pred_fno = torch.cat(pred, dim=-1)  # (S,S,H_eff)
        e_fno = loss_func(pred_fno.unsqueeze(0), gt_tail.unsqueeze(0)).item()
        results['FNO'].append(e_fno)

    # report
    for tag in results:
        arr = np.array(results[tag])
        if len(arr) == 0:
            continue
        print(f"{tag} rollout Raw L2 (mean±std): {arr.mean():.4f} ± {arr.std():.4f} (N={len(arr)})")

    # save simple numpy results
    np.save(os.path.join(args.save_dir, 'rollout_results.npy'), {k: np.array(v) for k,v in results.items()})


if __name__ == '__main__':
    main()


