import os
import h5py
import torch
import numpy as np
import argparse
from typing import Tuple

from utilities3 import UnitGaussianNormalizer, LpLoss
from fourier_3d_clean import FNO3d as FNO_ctor
from fourier_3d_cft_residual import FNO_RC_3D as RC_ctor
from fourier_3d_unet import U_FNO_3d as UFNO_ctor
from fourier_3d_lowrank import LowRankFNO3d as LRFNO_ctor
from fourier_3d_afno import AFNO3D as AFNO_ctor


def build_windows(series: torch.Tensor, T_in: int, T_out: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    N, S1, S2, T_total = series.shape
    xs, ys, starts = [], [], []
    for n in range(N):
        for t0 in range(0, max(1, T_total - (T_in + T_out) + 1), stride):
            xs.append(series[n, :, :, t0:t0+T_in])
            ys.append(series[n, :, :, t0+T_in:t0+T_in+T_out])
            starts.append(t0)
    X = torch.stack(xs, 0)
    Y = torch.stack(ys, 0)
    starts = torch.tensor(starts, dtype=torch.float32)
    return X, Y, starts, T_total


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
    # 1) Prefer LowRank-FNO spectral core (*.B)
    m1 = m2 = m3 = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4 and k.endswith('.B'):
            m1, m2, m3 = int(v.shape[-3]), int(v.shape[-2]), int(v.shape[-1])
            break
    # 2) Otherwise, fall back to a 5D spectral weight but skip 1x1x1 conv kernels
    if m1 is None:
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 5 and v.shape[0] == v.shape[1]:
                if v.shape[-3] == 1 and v.shape[-2] == 1 and v.shape[-1] == 1:
                    continue  # skip 1x1x1 conv weights
                m1, m2, m3 = int(v.shape[-3]), int(v.shape[-2]), int(v.shape[-1])
                if width is None:
                    width = int(v.shape[0])
                break
    return width, (m1, m2, m3), in_total


def _build_model(tag, ctor, path, device, T_in, has_coords):
    sd = _load_state_robust(path, device)
    width, (m1, m2, m3), in_total = _infer_hparams(sd)
    if m1 is None:
        # fallback
        m1 = m2 = m3 = 6
    if in_total is None:
        in_effective = T_in + 1
    else:
        in_effective = max(1, int(in_total) - (3 if has_coords else 0))
    model = ctor(m1, m2, m3, width, in_channels=in_effective, out_channels=1).to(device)
    strict = True
    model.load_state_dict(sd, strict=strict)
    model.eval()
    return model, in_effective


def append_abs_time(x, starts, T_total, T_in):
    num, S1, S2 = x.shape[0], x.shape[1], x.shape[2]
    t_abs = (starts / float(T_total)).view(num, 1, 1, 1).expand(num, S1, S2, 1)
    return torch.cat([x, t_abs], dim=-1)


def make_input(x_win, t0, T_total, T_in, T_out, a_norm, device, in_effective, encode_input):
    x = a_norm.encode(x_win.unsqueeze(0)) if encode_input else x_win.unsqueeze(0)
    S1, S2 = x.shape[1], x.shape[2]
    # default: only abs time channel; coords由各模型内部或权重定义决定
    if x.shape[-1] < in_effective:
        # pad zeros to required in_channels
        pad = torch.zeros(1, S1, S2, in_effective - x.shape[-1])
        x = torch.cat([x, pad], dim=-1)
    elif x.shape[-1] > in_effective:
        x = x[..., :in_effective]
    x = x.to(device)
    x5 = x.reshape(1, S1, S2, 1, in_effective).repeat(1, 1, 1, T_out, 1)
    return x5


def _spectral_resize_batch(x_spatial: torch.Tensor, target_res: int) -> torch.Tensor:
    # x_spatial: (B, H, W), returns (B, target_res, target_res)
    B, H, W = x_spatial.shape
    assert H == W, "Only square inputs supported for spectral resize"
    F = torch.fft.fftshift(torch.fft.fft2(x_spatial), dim=(-2, -1))
    if target_res == H:
        F_res = F
    elif target_res > H:
        pad_total = target_res - H
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        F_res = torch.nn.functional.pad(F, (pad_before, pad_after, pad_before, pad_after))
    else:
        crop = H - target_res
        crop_before = crop // 2
        crop_after = crop - crop_before
        F_res = F[:, crop_before:H-crop_after, crop_before:W-crop_after]
    x_res = torch.fft.ifft2(torch.fft.ifftshift(F_res, dim=(-2, -1))).real
    # scale to roughly preserve energy when zero-padding/cropping
    x_res = x_res * (target_res / H)
    return x_res


def resample_series(series_np, target_res, mode: str = 'bilinear'):
    # series_np: (N,S1,S2,T)
    N, S1, S2, T = series_np.shape
    if mode == 'bilinear':
        x = torch.from_numpy(series_np).float().unsqueeze(1)  # (N,1,S1,S2,T)
        x = x.permute(0, 4, 1, 2, 3)  # (N,T,1,S1,S2)
        x_rs = torch.nn.functional.interpolate(x.reshape(N*T,1,S1,S2), size=(target_res,target_res), mode='bilinear', align_corners=False)
        x_rs = x_rs.reshape(N, T, 1, target_res, target_res).permute(0,3,4,1,2).squeeze(-1)
        return x_rs  # (N,target_res,target_res,T)
    elif mode == 'spectral':
        x = torch.from_numpy(series_np).float()  # (N,S1,S2,T)
        x = x.permute(0, 3, 1, 2)  # (N,T,S1,S2)
        NT = N * T
        x2 = x.reshape(NT, S1, S2)
        x2r = _spectral_resize_batch(x2, target_res)
        x_rs = x2r.reshape(N, T, target_res, target_res).permute(0, 2, 3, 1)
        return x_rs  # (N,target_res,target_res,T)
    else:
        raise ValueError(f"Unknown resample_mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description='Cross-resolution evaluation (unified protocol)')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--save_dir', type=str, default='/content/cross_res')
    parser.add_argument('--fno_path', type=str, default='/content/drive/MyDrive/fno_models/fno_3d_standard.pt')
    parser.add_argument('--rc_path', type=str, default='/content/models/fno_rc_3d_seq.pt')
    parser.add_argument('--ufno_path', type=str, default='/content/models/u_fno_3d.pt')
    parser.add_argument('--lowrank_path', type=str, default='/content/models/lowrank_fno_3d.pt')
    parser.add_argument('--afno_path', type=str, default='/content/models/afno_3d.pt')
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_out', type=int, default=20)
    parser.add_argument('--target_res', type=int, default=96)
    parser.add_argument('--resample_mode', type=str, default='spectral', choices=['bilinear','spectral'])
    parser.add_argument('--ntrain', type=int, default=40)
    parser.add_argument('--ntest', type=int, default=10)
    # RC diagnostics for cross-res: disable or scale correction branch at eval time
    parser.add_argument('--rc_disable', action='store_true', help='Disable RC correction branch during eval')
    parser.add_argument('--rc_gamma_scale', type=float, default=1.0, help='Multiply RC correction_scale by this factor at eval')
    parser.add_argument('--rc_set_gamma', type=float, default=None, help='If set, force RC correction_scale to this value at eval')
    parser.add_argument('--encode_input_rc', action='store_true', help='Encode RC inputs with normalizer (default: off)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_func = LpLoss(size_average=True)

    # load data
    with h5py.File(args.data_path, 'r') as f:
        u = f['u'][()]
    u = np.transpose(u, (3,1,2,0))  # (N,S1,S2,T)
    test_series = u[-args.ntest:]
    test_series_rs = resample_series(test_series, args.target_res, mode=args.resample_mode)

    # build windows on resampled test
    # test_series_rs already returns a torch.Tensor of shape (N,S,S,T)
    test_a, test_u, test_starts, test_T = build_windows(test_series_rs.float(), args.T_in, args.T_out, stride=args.T_out)

    # normalizer: for cross-resolution, fit on resampled test windows to match target grid shape
    y_norm_cpu = UnitGaussianNormalizer(test_u)
    y_norm_gpu = UnitGaussianNormalizer(test_u)
    y_norm_gpu.load_state_dict(y_norm_cpu.state_dict()); y_norm_gpu.to(device)

    a_norm = UnitGaussianNormalizer(test_a)  # for RC encode only

    # build models
    models = [
        ('FNO-RC', RC_ctor, args.rc_path, True),
        ('FNO',    FNO_ctor, args.fno_path, True),
        ('U-FNO',  UFNO_ctor, args.ufno_path, False),
        ('LowRank',LRFNO_ctor, args.lowrank_path, False),
        ('AFNO',   AFNO_ctor, args.afno_path, True),
    ]

    built = []
    for tag, ctor, path, has_coords in models:
        model, in_eff = _build_model(tag, ctor, path, device, args.T_in, has_coords)
        # Apply RC overrides for diagnostics
        if tag == 'FNO-RC':
            # toggle enable_correction and scale correction_scale on all SpectralConv3d_RC layers if present
            rc_layers = []
            for name in ['conv0','conv1','conv2','conv3']:
                if hasattr(model, name):
                    rc_layers.append(getattr(model, name))
            if args.rc_disable:
                for l in rc_layers:
                    if hasattr(l, 'enable_correction'):
                        l.enable_correction = False
            if args.rc_set_gamma is not None:
                for l in rc_layers:
                    if hasattr(l, 'correction_scale'):
                        with torch.no_grad():
                            l.correction_scale.data.fill_(float(args.rc_set_gamma))
            elif abs(args.rc_gamma_scale - 1.0) > 1e-8:
                for l in rc_layers:
                    if hasattr(l, 'correction_scale'):
                        with torch.no_grad():
                            l.correction_scale.data.mul_(float(args.rc_gamma_scale))
        built.append((tag, model, in_eff))

    # evaluate
    per_model = {tag: [] for tag,_,_ in built}
    for i in range(test_a.shape[0]):
        x_win = test_a[i]
        y_win = test_u[i].unsqueeze(0).to(device)
        with torch.no_grad():
            for tag, model, in_eff in built:
                encode_in = (tag == 'FNO-RC' and args.encode_input_rc)
                x5 = make_input(x_win, test_starts[i].item(), test_T, args.T_in, args.T_out, a_norm, device, in_eff, encode_input=encode_in)
                out = model(x5).squeeze(-1)
                # raw metric (no decode for any model to ensure fair Raw-space comparison)
                out_raw = out
                e = loss_func(out_raw, y_win).item()
                per_model[tag].append(e)

    print("\nCross-resolution @{}x{} results (Raw L2, mean over windows):".format(args.target_res, args.target_res))
    for tag, vals in per_model.items():
        arr = np.array(vals)
        print("{:8s}: {:.4f} ± {:.4f} (N={})".format(tag, arr.mean(), arr.std(), len(arr)))


if __name__ == '__main__':
    main()


