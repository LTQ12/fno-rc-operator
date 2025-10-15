import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes

# It's crucial to import the model definitions and utilities
from fourier_3d_clean import FNO3d
from fourier_3d_cft_residual import FNO_RC_3D
from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss

# Use a non-interactive backend for saving plots
plt.switch_backend('agg')

def ultimate_comparison(args):
    """
    Compares two sequence-to-sequence models: FNO-RC and the original FNO baseline.
    Generates plots for L2 error vs. time step, final timestep slices, and 3D isosurfaces.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Data and Normalizers ---
    print("Loading data...")
    try:
        reader = MatReader(args.data_path)
        u_field = reader.read_field('u')
        
        ntrain = 40 # Using the same split as the last successful training
        ntest = 10
        
        train_a = u_field[:ntrain, ..., :args.T_in]
        test_a = u_field[-ntest:, ..., :args.T_in]
        test_u = u_field[-ntest:, ..., args.T_in:args.T_in + args.T_out]

        a_normalizer = UnitGaussianNormalizer(train_a)
        y_normalizer = UnitGaussianNormalizer(u_field[:ntrain, ..., args.T_in:args.T_in + args.T_out])
        y_normalizer.to(device)

        # Input tensors will be prepared after normalization inside the prediction loop.
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Load Models ---
    print("Loading models...")
    try:
        model_fno_rc = FNO_RC_3D(args.modes, args.modes, args.modes, args.width, args.T_in, args.T_out).to(device)
        model_fno_rc.load_state_dict(torch.load(args.model_path_fno_rc, map_location=device))
        model_fno_rc.eval()
        print(f"FNO-RC model loaded from {args.model_path_fno_rc}")

        model_fno_baseline = FNO3d(args.modes, args.modes, args.modes, args.width, in_channels=args.T_in, out_channels=args.T_out).to(device)
        model_fno_baseline.load_state_dict(torch.load(args.model_path_fno, map_location=device))
        model_fno_baseline.eval()
        print(f"Baseline FNO model loaded from {args.model_path_fno}")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # --- 3. Run Predictions and Calculate Errors ---
    loss_func = LpLoss(size_average=True)
    
    # We will analyze a single sample, e.g., the first one in the test set
    sample_idx = 0
    
    with torch.no_grad():
        # Prepare inputs by NORMALIZING FIRST, then creating model-specific shapes.
        raw_input_slice = test_a[sample_idx:sample_idx+1]
        ground_truth = test_u[sample_idx:sample_idx+1] # Shape: (1, S, S, T_out)

        # Step 1: Normalize the raw input.
        normalized_input = a_normalizer.encode(raw_input_slice).to(device)
        
        # Step 2: Create model-specific inputs from the *normalized* data.
        # 两个模型输入保持一致：使用 4D (B,S,S,T_in) 张量，内部各自拼接网格。
        # 为公平比较，将输入在深度维复制到 T_out（与训练一致的协议）。
        S = normalized_input.shape[1]
        x_rc = normalized_input.reshape(1, S, S, 1, args.T_in).repeat([1,1,1,args.T_out,1])[..., 0:args.T_in]
        x_rc = x_rc.squeeze(3) if x_rc.shape[3] == 1 else x_rc
        x_baseline = x_rc

        # Get predictions
        pred_rc_norm = model_fno_rc(x_rc)
        pred_rc = y_normalizer.decode(pred_rc_norm).cpu()

        pred_baseline_norm = model_fno_baseline(x_baseline)
        pred_baseline = y_normalizer.decode(pred_baseline_norm).cpu()

        # Calculate step-wise L2 error
        errors_rc = []
        errors_baseline = []
        for t in range(args.T_out):
            error_t_rc = loss_func(pred_rc[..., t], ground_truth[..., t]).item()
            errors_rc.append(error_t_rc)

            error_t_baseline = loss_func(pred_baseline[..., t], ground_truth[..., t]).item()
            errors_baseline.append(error_t_baseline)
    
    print("\n--- Final Performance Comparison (Sequence) ---")
    print(f"FNO-RC      | Avg L2 Error over {args.T_out} steps: {np.mean(errors_rc):.6f}")
    print(f"Baseline FNO  | Avg L2 Error over {args.T_out} steps: {np.mean(errors_baseline):.6f}")
    
    # --- 4. Generate Plots ---
    print("\nGenerating plots...")
    os.makedirs(args.plot_save_dir, exist_ok=True)
    
    # Plot 1: L2 Error vs. Time Step
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.T_in, args.T_in + args.T_out), errors_baseline, 'r-o', label=f'Baseline FNO (Final Error: {errors_baseline[-1]:.4f})')
    plt.plot(range(args.T_in, args.T_in + args.T_out), errors_rc, 'b-s', label=f'FNO-RC (Final Error: {errors_rc[-1]:.4f})')
    plt.yscale('log')
    plt.xlabel('Time Step')
    plt.ylabel('Relative L2 Error (log scale)')
    plt.title('Prediction Error vs. Time Step')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plot_path = os.path.join(args.plot_save_dir, 'error_vs_time_step.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved error vs time plot to {plot_path}")

    # Plot 2: Final Timestep Slice Comparison (z=S/2)
    final_t = -1
    slice_z = ground_truth.shape[1] // 2
    
    gt_slice = ground_truth[0, :, :, final_t].numpy()
    rc_slice = pred_rc[0, :, :, final_t].numpy()
    baseline_slice = pred_baseline[0, :, :, final_t].numpy()
    
    vmax = np.max(gt_slice)
    vmin = np.min(gt_slice)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Comparison at Final Timestep (t={args.T_in + args.T_out -1}, z={slice_z})', fontsize=16)
    
    im1 = axes[0].imshow(gt_slice, cmap='viridis', vmax=vmax, vmin=vmin)
    axes[0].set_title('Ground Truth')
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(rc_slice, cmap='viridis', vmax=vmax, vmin=vmin)
    axes[1].set_title(f'FNO-RC Prediction (Error: {errors_rc[final_t]:.4f})')
    fig.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(baseline_slice, cmap='viridis', vmax=vmax, vmin=vmin)
    axes[2].set_title(f'Baseline FNO Prediction (Error: {errors_baseline[final_t]:.4f})')
    fig.colorbar(im3, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plot_path = os.path.join(args.plot_save_dir, 'final_timestep_slice_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved slice comparison plot to {plot_path}")

    # Plot 3: 3D Isosurface Comparison
    level = np.mean(ground_truth[0, ..., final_t].numpy())
    
    try:
        verts_gt, faces_gt, _, _ = marching_cubes(ground_truth[0, ..., final_t].numpy(), level=level)
        verts_rc, faces_rc, _, _ = marching_cubes(pred_rc[0, ..., final_t].numpy(), level=level)
        verts_baseline, faces_baseline, _, _ = marching_cubes(pred_baseline[0, ..., final_t].numpy(), level=level)

        fig = plt.figure(figsize=(21, 7))
        fig.suptitle(f'3D Isosurface Comparison at Final Timestep (t={args.T_in + args.T_out -1})', fontsize=16)

        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_trisurf(verts_gt[:, 0], verts_gt[:, 1], faces_gt, verts_gt[:, 2], cmap='viridis', lw=1)
        ax1.set_title('Ground Truth')

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_trisurf(verts_rc[:, 0], verts_rc[:, 1], faces_rc, verts_rc[:, 2], cmap='viridis', lw=1)
        ax2.set_title('FNO-RC Prediction')
        
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_trisurf(verts_baseline[:, 0], verts_baseline[:, 1], faces_baseline, verts_baseline[:, 2], cmap='viridis', lw=1)
        ax3.set_title('Baseline FNO Prediction')

        plot_path = os.path.join(args.plot_save_dir, 'final_timestep_isosurface_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved isosurface plot to {plot_path}")

    except Exception as e:
        print(f"Could not generate isosurface plot. This can happen if the field is too flat. Error: {e}")

    print("\nUltimate comparison finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ultimate 3D FNO Comparison.')
    
    # --- Paths ---
    parser.add_argument('--data_path', type=str, default='/content/data/ns_V1e-4_N10000_T30.mat',
                        help='Path to the .mat file for 3D Navier-Stokes data.')
    parser.add_argument('--model_path_fno', type=str, default='/content/fourier_neural_operator-master/models/fno_3d_standard.pt',
                        help='Path to the trained baseline FNO model.')
    parser.add_argument('--model_path_fno_rc', type=str, default='/content/fourier_neural_operator-master/models/fno_rc_3d_seq.pt',
                        help='Path to the trained FNO-RC sequence-to-sequence model.')
    parser.add_argument('--plot_save_dir', type=str, default='/content/ultimate_comparison_3d_plots',
                        help='Directory to save the comparison plots.')

    # --- Model & Data Parameters (should match training) ---
    parser.add_argument('--T_in', type=int, default=10, help='Input time steps.')
    parser.add_argument('--T_out', type=int, default=20, help='Output time steps.')
    parser.add_argument('--modes', type=int, default=8, help='Fourier modes.')
    parser.add_argument('--width', type=int, default=20, help='Width of the FNO layers.')
    
    args = parser.parse_args()
    ultimate_comparison(args)