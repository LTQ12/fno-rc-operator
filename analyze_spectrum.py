import os
import h5py
import torch
import numpy as np
import argparse
from typing import Tuple
import matplotlib.pyplot as plt

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
    return model, in_effective


def make_input(x_win, T_out, device, in_effective):
    S1, S2, C = x_win.shape
    if C < in_effective:
        pad = torch.zeros(S1, S2, in_effective - C)
        x_win = torch.cat([x_win, pad], dim=-1)
    elif C > in_effective:
        x_win = x_win[..., :in_effective]
    x = x_win.unsqueeze(0)
    x5 = x.reshape(1, S1, S2, 1, in_effective).repeat(1, 1, 1, T_out, 1)
    return x5.to(device)


def _radial_indices(H: int, W: int):
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    rr = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    r = rr.round().long()
    max_r = int(r.max().item())
    return r, max_r


def radial_spectrum_2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W = x.shape
    F = torch.fft.fftshift(torch.fft.fft2(x))
    P = (F.real ** 2 + F.imag ** 2)
    r, max_r = _radial_indices(H, W)
    spec = torch.zeros(max_r + 1)
    count = torch.zeros(max_r + 1)
    for i in range(H):
        for j in range(W):
            ri = int(r[i, j].item())
            spec[ri] += P[i, j]
            count[ri] += 1
    spec = spec / (count + 1e-8)
    return torch.arange(max_r + 1), spec


def main():
    parser = argparse.ArgumentParser(description='Spectral energy and phase/amplitude error analysis')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--save_dir', type=str, default='/content/spectrum')
    parser.add_argument('--rc_path', type=str, default='/content/models/fno_rc_3d_seq_multires.pt')
    parser.add_argument('--fno_path', type=str, default='/content/drive/MyDrive/fno_models/fno_3d_standard.pt')
    parser.add_argument('--target_res', type=int, default=96)
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_out', type=int, default=20)
    parser.add_argument('--ntest', type=int, default=5)
    parser.add_argument('--save_plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with h5py.File(args.data_path, 'r') as f:
        u = f['u'][()]
    u = np.transpose(u, (3,1,2,0))
    test_series = u[-args.ntest:]
    S1, S2, T = test_series.shape[1], test_series.shape[2], test_series.shape[3]

    rc_model, rc_in_eff = _build_model('FNO-RC', RC_ctor, args.rc_path, device, args.T_in, has_coords=True)
    fno_model, fno_in_eff = _build_model('FNO', FNO_ctor, args.fno_path, device, args.T_in, has_coords=True)

    # choose one window per sample (first window)
    k_axis = None
    spec_gt = []
    spec_rc = []
    spec_fno = []
    amp_rel_errs_rc = []
    amp_rel_errs_fno = []
    phase_abs_errs_rc = []
    phase_abs_errs_fno = []
    for n in range(test_series.shape[0]):
        series = test_series[n]
        ctx = series[..., :args.T_in]
        y = torch.from_numpy(series[..., args.T_in:args.T_in+args.T_out]).float()
        x_win = torch.from_numpy(np.concatenate([ctx, np.zeros((S1,S2,1), dtype=ctx.dtype)], axis=-1)).float()
        with torch.no_grad():
            out_rc = rc_model(make_input(x_win, args.T_out, device, rc_in_eff)).squeeze(-1)[0].cpu()
            out_fno = fno_model(make_input(x_win, args.T_out, device, fno_in_eff)).squeeze(-1)[0].cpu()
        # pick mid time slice for spectrum
        tidx = args.T_out // 2
        y_mid = y[..., tidx]
        rc_mid = out_rc[..., tidx]
        fno_mid = out_fno[..., tidx]
        # energy spectra
        kr, e_gt = radial_spectrum_2d(y_mid)
        _, e_rc = radial_spectrum_2d(rc_mid)
        _, e_fno = radial_spectrum_2d(fno_mid)
        k_axis = kr
        spec_gt.append(e_gt)
        spec_rc.append(e_rc)
        spec_fno.append(e_fno)
        # amplitude/phase errors on 2D FFT
        def amp_phase(x2d: torch.Tensor):
            F = torch.fft.fft2(x2d)
            A = torch.abs(F)
            P = torch.angle(F)
            return A, P
        A_gt, P_gt = amp_phase(y_mid)
        A_rc, P_rc = amp_phase(rc_mid)
        A_fno, P_fno = amp_phase(fno_mid)
        mask = A_gt > (1e-6 * float(A_gt.max().item() if A_gt.numel()>0 else 1.0))
        # amplitude relative error
        amp_rel_rc = (torch.abs(A_rc - A_gt) / (A_gt + 1e-12))[mask].mean().item()
        amp_rel_fno = (torch.abs(A_fno - A_gt) / (A_gt + 1e-12))[mask].mean().item()
        amp_rel_errs_rc.append(amp_rel_rc)
        amp_rel_errs_fno.append(amp_rel_fno)
        # phase absolute error (wrap to [-pi,pi])
        def phase_err(Pp, Pg):
            d = Pp - Pg
            d = (d + np.pi) % (2*np.pi) - np.pi
            return torch.abs(d)
        ph_err_rc = phase_err(P_rc, P_gt)[mask].mean().item()
        ph_err_fno = phase_err(P_fno, P_gt)[mask].mean().item()
        phase_abs_errs_rc.append(ph_err_rc)
        phase_abs_errs_fno.append(ph_err_fno)

    spec_gt = torch.stack(spec_gt).mean(0)
    spec_rc = torch.stack(spec_rc).mean(0)
    spec_fno = torch.stack(spec_fno).mean(0)

    # save to npy for plotting elsewhere
    np.save(os.path.join(args.save_dir, 'k.npy'), k_axis.numpy())
    np.save(os.path.join(args.save_dir, 'spec_gt.npy'), spec_gt.numpy())
    np.save(os.path.join(args.save_dir, 'spec_rc.npy'), spec_rc.numpy())
    np.save(os.path.join(args.save_dir, 'spec_fno.npy'), spec_fno.numpy())
    # high-frequency energy ratio (top 1/3 of k)
    k_max = int(k_axis.max().item()) if k_axis is not None else 0
    k_cut = max(1, k_max * 2 // 3)
    hi_gt = spec_gt[k_cut:].sum().item()
    hi_rc = spec_rc[k_cut:].sum().item()
    hi_fno = spec_fno[k_cut:].sum().item()
    # amplitude/phase summary
    amp_rc = np.mean(amp_rel_errs_rc), np.std(amp_rel_errs_rc)
    amp_fno = np.mean(amp_rel_errs_fno), np.std(amp_rel_errs_fno)
    ph_rc = np.mean(phase_abs_errs_rc), np.std(phase_abs_errs_rc)
    ph_fno = np.mean(phase_abs_errs_fno), np.std(phase_abs_errs_fno)
    with open(os.path.join(args.save_dir, 'spectrum_summary.txt'), 'w') as f:
        f.write(f'high_freq_k_cut={k_cut} (of max {k_max})\n')
        f.write(f'Hi-Energy (GT/RC/FNO): {hi_gt:.6e} / {hi_rc:.6e} / {hi_fno:.6e}\n')
        f.write(f'AmpRelErr RC mean±std: {amp_rc[0]:.6f} ± {amp_rc[1]:.6f}\n')
        f.write(f'AmpRelErr FNO mean±std: {amp_fno[0]:.6f} ± {amp_fno[1]:.6f}\n')
        f.write(f'PhaseAbsErr RC mean±std (rad): {ph_rc[0]:.6f} ± {ph_rc[1]:.6f}\n')
        f.write(f'PhaseAbsErr FNO mean±std (rad): {ph_fno[0]:.6f} ± {ph_fno[1]:.6f}\n')

    if args.save_plot:
        plt.figure(figsize=(6,4))
        plt.loglog(k_axis[1:].numpy(), spec_gt[1:].numpy() + 1e-12, label='GT')
        plt.loglog(k_axis[1:].numpy(), spec_rc[1:].numpy() + 1e-12, label='FNO-RC')
        plt.loglog(k_axis[1:].numpy(), spec_fno[1:].numpy() + 1e-12, label='FNO')
        plt.xlabel('k'); plt.ylabel('Energy')
        plt.legend(); plt.grid(True, which='both', ls='--', alpha=.3)
        plt.tight_layout()
        out_png = os.path.join(args.save_dir, 'spectrum_energy.png')
        plt.savefig(out_png, dpi=200)
        plt.close()

    print('Saved spectra and summary to', args.save_dir)


if __name__ == '__main__':
    main()


