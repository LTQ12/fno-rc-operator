import torch
import numpy as np
import torch.nn as nn
import argparse
import os
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend to prevent empty plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fourier_3d_clean import FNO3d
from fourier_3d_cft_residual import FNO_RC_3D
from utilities3 import LpLoss, MatReader, UnitGaussianNormalizer

def compare_3d_models(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    try:
        reader = MatReader(args.data_path)
        u_field = reader.read_field('u')
        
        ntrain = args.ntrain
        ntest_actual = args.ntest
        print(f"Partitioning data: {ntrain} for normalizer, {ntest_actual} for testing.")

        train_a_norm_src = u_field[:ntrain, ..., :args.T_in]
        train_u_norm_src = u_field[:ntrain, ..., args.T_in:args.T_in + args.T_out]
        
        test_a_data = u_field[-ntest_actual:, ..., :args.T_in]
        test_u_data = u_field[-ntest_actual:, ..., args.T_in:args.T_in + args.T_out]
    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        return

    a_normalizer = UnitGaussianNormalizer(train_a_norm_src)
    test_a_data = a_normalizer.encode(test_a_data)
    y_normalizer_last_step = UnitGaussianNormalizer(train_u_norm_src[..., -1])
    y_normalizer_last_step.to(device)

    S = test_a_data.shape[1]
    x_test = test_a_data.reshape(ntest_actual, S, S, 1, args.T_in).repeat([1, 1, 1, args.T_out, 1])
    y_test_full_sequence = test_u_data
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test_full_sequence), batch_size=args.batch_size, shuffle=False)

    try:
        model_fno = FNO3d(args.modes, args.modes, args.modes, args.width, in_dim=13, out_dim=1).to(device)
        model_fno.load_state_dict(torch.load(args.fno_model_path, map_location=device))
        model_fno.eval()
        print(f"Baseline FNO loaded from {args.fno_model_path}")

        model_rc = FNO_RC_3D(args.modes, args.modes, args.modes, args.width, in_channels=args.T_in, out_channels=1).to(device)
        model_rc.load_state_dict(torch.load(args.rc_model_path, map_location=device))
        model_rc.eval()
        print(f"FNO-RC loaded from {args.rc_model_path}")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    loss_func = LpLoss(size_average=False)
    
    grid = torch.cat([
        torch.linspace(0, 1, S).view(1, S, 1, 1, 1).repeat(1, 1, S, args.T_out, 1),
        torch.linspace(0, 1, S).view(1, 1, S, 1, 1).repeat(1, S, 1, args.T_out, 1),
        torch.linspace(0, 1, args.T_out).view(1, 1, 1, args.T_out, 1).repeat(1, S, S, 1, 1)
    ], dim=-1).to(device)
    
    mean = y_normalizer_last_step.mean.unsqueeze(-1) 
    std = y_normalizer_last_step.std.unsqueeze(-1)   

    loss_fno_last_step, loss_rc_last_step = 0.0, 0.0
    
    with torch.no_grad():
        for x, y_full in test_loader:
            x, y_full = x.to(device), y_full.to(device)
            y_last = y_full[..., -1]
            
            x_fno_in = x[..., :args.T_in] 
            batch_grid = grid.repeat(x.size(0), 1, 1, 1, 1)
            x_fno_with_grid = torch.cat((x_fno_in, batch_grid), dim=-1)

            pred_fno_3d_raw = model_fno(x_fno_with_grid).squeeze(-1)
            pred_fno_3d_decoded = pred_fno_3d_raw * std + mean
            loss_fno_last_step += loss_func(pred_fno_3d_decoded[..., -1], y_last).item()

            pred_rc_last_step_raw = model_rc(x).squeeze(-1)
            pred_rc_last_step_decoded = pred_rc_last_step_raw * std + mean
            loss_rc_last_step += loss_func(pred_rc_last_step_decoded[..., -1], y_last).item()

    avg_loss_fno_last_step = (loss_fno_last_step / ntest_actual)
    avg_loss_rc_last_step = (loss_rc_last_step / ntest_actual)
    
    print("\n--- Final Performance Comparison ---")
    print("Metric: Error at Final Timestep - FAIR COMPARISON")
    print(f"Baseline FNO | Avg. L2 Error: {avg_loss_fno_last_step:.6f}")
    print(f"Our FNO-RC   | Avg. L2 Error: {avg_loss_rc_last_step:.6f}")
    improvement = ((avg_loss_fno_last_step - avg_loss_rc_last_step) / avg_loss_fno_last_step) * 100
    print(f"-> Relative Improvement: {improvement:.2f}%")
    print("------------------------------------")

    print("\nGenerating comprehensive comparison plots for the first sample...")
    os.makedirs(args.plot_save_dir, exist_ok=True)
    
    x_sample, y_full_sample = x_test[0:1].to(device), y_test_full_sequence[0:1].to(device)
    
    with torch.no_grad():
        x_fno_in = x_sample[..., :args.T_in]
        batch_grid = grid.repeat(x_sample.size(0), 1, 1, 1, 1)
        x_fno_with_grid = torch.cat((x_fno_in, batch_grid), dim=-1)

        pred_fno_3d_raw = model_fno(x_fno_with_grid).squeeze(-1)
        pred_rc_last_step_raw = model_rc(x_sample).squeeze(-1)
        
        pred_fno_decoded = pred_fno_3d_raw * std + mean
        pred_rc_decoded = pred_rc_last_step_raw * std + mean

    y_last_cpu = y_full_sample.cpu().squeeze(0)[..., -1].numpy()
    pred_fno_cpu = pred_fno_decoded.cpu().squeeze(0).numpy()
    pred_rc_cpu = pred_rc_decoded.cpu().squeeze(0).numpy()
    
    # Plot 1: The Core Result - Comparison at z=0 slice
    print("Plotting comparison at z=0 slice...")
    gt_slice = y_last_cpu
    fno_slice = pred_fno_cpu[..., -1]
    rc_slice = pred_rc_cpu[..., -1]
    
    err_fno = np.abs(fno_slice - gt_slice)
    err_rc = np.abs(rc_slice - gt_slice)
    
    vmax_err = max(err_fno.max(), err_rc.max())
    vmax_pred = max(gt_slice.max(), fno_slice.max(), rc_slice.max())
    vmin_pred = min(gt_slice.min(), fno_slice.min(), rc_slice.min())

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('3D Navier-Stokes: Prediction vs. Ground Truth at z=0, t=29', fontsize=20)
    
    im = axes[0, 0].imshow(gt_slice, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred, origin='lower')
    axes[0, 0].set_title('Ground Truth (z=0)', fontsize=14); fig.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].imshow(fno_slice, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred, origin='lower')
    axes[0, 1].set_title('Baseline FNO Pred. (z=0)', fontsize=14); fig.colorbar(im, ax=axes[0, 1])
    im = axes[0, 2].imshow(rc_slice, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred, origin='lower')
    axes[0, 2].set_title('Our FNO-RC Pred. (z=0)', fontsize=14); fig.colorbar(im, ax=axes[0, 2])
    axes[1, 0].axis('off')
    im = axes[1, 1].imshow(err_fno, cmap='Reds', vmin=0, vmax=vmax_err, origin='lower')
    axes[1, 1].set_title(f'Baseline FNO Abs. Error', fontsize=14); fig.colorbar(im, ax=axes[1, 1])
    im = axes[1, 2].imshow(err_rc, cmap='Reds', vmin=0, vmax=vmax_err, origin='lower')
    axes[1, 2].set_title(f'Our FNO-RC Abs. Error', fontsize=14); fig.colorbar(im, ax=axes[1, 2])
    
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(args.plot_save_dir, 'comparison_z0_slice.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close(fig)
    
    # Plot 2: Enhanced Orthogonal Slices Comparison
    print("Plotting orthogonal slices comparison...")
    s_x, s_y, s_z = pred_rc_cpu.shape
    slice_idx_x, slice_idx_y, slice_idx_z = s_x // 2, s_y // 2, s_z // 2

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle('3D Prediction Orthogonal Slices Comparison (t=29)', fontsize=20)
    
    # Row 1: FNO-RC
    axes[0, 0].set_title(f'FNO-RC: XY Slice at z={slice_idx_z}'); axes[0, 0].imshow(pred_rc_cpu[..., slice_idx_z].T, cmap='viridis', origin='lower')
    axes[0, 1].set_title(f'FNO-RC: XZ Slice at y={slice_idx_y}'); axes[0, 1].imshow(pred_rc_cpu[:, slice_idx_y, :].T, cmap='viridis', origin='lower')
    axes[0, 2].set_title(f'FNO-RC: YZ Slice at x={slice_idx_x}'); axes[0, 2].imshow(pred_rc_cpu[slice_idx_x, :, :].T, cmap='viridis', origin='lower')
    
    # Row 2: Baseline FNO
    axes[1, 0].set_title(f'Baseline FNO: XY Slice at z={slice_idx_z}'); axes[1, 0].imshow(pred_fno_cpu[..., slice_idx_z].T, cmap='viridis', origin='lower')
    axes[1, 1].set_title(f'Baseline FNO: XZ Slice at y={slice_idx_y}'); axes[1, 1].imshow(pred_fno_cpu[:, slice_idx_y, :].T, cmap='viridis', origin='lower')
    axes[1, 2].set_title(f'Baseline FNO: YZ Slice at x={slice_idx_x}'); axes[1, 2].imshow(pred_fno_cpu[slice_idx_x, :, :].T, cmap='viridis', origin='lower')
    
    for i, ax in enumerate(axes.flat):
        if i < 3: ax.set_ylabel('FNO-RC', fontsize=14)
        else: ax.set_ylabel('Baseline', fontsize=14)
        if i % 3 == 0: ax.set_xlabel('x'); ax.set_ylabel('y')
        if i % 3 == 1: ax.set_xlabel('x'); ax.set_ylabel('z (output time)')
        if i % 3 == 2: ax.set_xlabel('y'); ax.set_ylabel('z (output time)')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(args.plot_save_dir, 'orthogonal_slices_comparison.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive 3D FNO Model Comparison.')
    
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--fno_model_path', type=str, default='models/fno_baseline_3d.pt')
    parser.add_argument('--rc_model_path', type=str, default='models/fno_rc_3d.pt')
    
    parser.add_argument('--ntrain', type=int, default=40)
    parser.add_argument('--ntest', type=int, default=10)
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_out', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    
    parser.add_argument('--modes', type=int, default=8)
    parser.add_argument('--width', type=int, default=20)
    
    parser.add_argument('--plot_save_dir', type=str, default='comparison_plots_3d')
    
    args = parser.parse_args()
    compare_3d_models(args)