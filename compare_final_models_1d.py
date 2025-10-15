import torch
import numpy as np
import h5py
import argparse
import os
import matplotlib.pyplot as plt

from fourier_1d_baseline import FNO1d
from fourier_1d_cft_residual import FNO_RC_1D
from utilities3 import LpLoss, UnitGaussianNormalizer, count_params

def create_training_data_from_sequence(sequence, T_in, T_out, step=1):
    """
    Creates training pairs (x, y) from a long time sequence.
    """
    N, S, T = sequence.shape
    num_samples = (T - T_in - T_out) // step
    
    X = torch.zeros(N * num_samples, S, T_in)
    Y = torch.zeros(N * num_samples, S, T_out)
    
    idx = 0
    for i in range(N):
        for t in range(0, T - T_in - T_out, step):
            X[idx] = sequence[i, :, t : t + T_in]
            Y[idx] = sequence[i, :, t + T_in : t + T_in + T_out]
            idx += 1
            
    return X, Y

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.plot_save_dir, exist_ok=True)
    loss_func = LpLoss()

    # --- Data Loading and Preparation ---
    try:
        with h5py.File(args.data_path, 'r') as f:
            u_field = torch.from_numpy(f['output'][:].astype(np.float32)).permute(2, 0, 1)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # This part MUST be identical to the training scripts to ensure correct test set
    all_a, all_y = create_training_data_from_sequence(u_field, args.T_in, args.T_out)
    ntrain = int(all_a.shape[0] * 0.8)
    
    # Use a fixed seed for shuffling to get the same test set every time
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(all_a.shape[0])
    train_indices = shuffled_indices[:ntrain]
    test_indices = shuffled_indices[ntrain:]

    test_a = all_a[test_indices].to(device)
    test_y = all_y[test_indices] # Keep y on CPU for plotting

    # Re-create normalizers from the training set to correctly decode outputs
    train_a_for_norm = all_a[train_indices]
    train_y_for_norm = all_y[train_indices]
    a_normalizer = UnitGaussianNormalizer(train_a_for_norm)
    y_normalizer = UnitGaussianNormalizer(train_y_for_norm)
    y_normalizer.to(device)

    # --- Load Models ---
    model_baseline = FNO1d(args.modes, args.width, in_channels=args.T_in, out_channels=args.T_out).to(device)
    model_rc = FNO_RC_1D(args.modes, args.width, in_channels=args.T_in, out_channels=args.T_out).to(device)

    try:
        model_baseline.load_state_dict(torch.load(args.model_path_baseline, map_location=device))
        model_rc.load_state_dict(torch.load(args.model_path_rc, map_location=device))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    model_baseline.eval()
    model_rc.eval()

    # --- Evaluation and Plotting ---
    print(f"Generating {args.num_plots} comparison plots...")
    with torch.no_grad():
        for i in range(args.num_plots):
            x = test_a[i:i+1] # Shape: (1, S, T_in)
            y_true = test_y[i]    # Shape: (S, T_out)

            # Get predictions
            pred_baseline_norm = model_baseline(x)
            pred_rc_norm = model_rc(x)

            # Decode predictions
            pred_baseline = y_normalizer.decode(pred_baseline_norm).squeeze(0).cpu() # Shape: (S, T_out)
            pred_rc = y_normalizer.decode(pred_rc_norm).squeeze(0).cpu()       # Shape: (S, T_out)

            # Calculate errors
            error_baseline = loss_func(pred_baseline.view(1,-1), y_true.view(1,-1)).item()
            error_rc = loss_func(pred_rc.view(1,-1), y_true.view(1,-1)).item()

            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            plt.suptitle(f'Burgers Equation - Comparison for Test Sample {i}', fontsize=16)

            # Determine shared color range
            vmin = y_true.min()
            vmax = y_true.max()

            # Ground Truth
            im1 = axes[0].imshow(y_true.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, extent=[0,1,1,0])
            axes[0].set_title('Ground Truth')
            axes[0].set_xlabel('Space (x)')
            axes[0].set_ylabel('Time (t)')

            # Baseline FNO Prediction
            im2 = axes[1].imshow(pred_baseline.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, extent=[0,1,1,0])
            axes[1].set_title(f'Baseline FNO (L2 Error: {error_baseline:.4f})')
            axes[1].set_xlabel('Space (x)')
            axes[1].set_yticklabels([])

            # FNO-RC Prediction
            im3 = axes[2].imshow(pred_rc.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, extent=[0,1,1,0])
            axes[2].set_title(f'Our FNO-RC (L2 Error: {error_rc:.4f})')
            axes[2].set_xlabel('Space (x)')
            axes[2].set_yticklabels([])

            # Add a colorbar
            fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
            
            save_path = os.path.join(args.plot_save_dir, f'comparison_1d_sample_{i}.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved plot to {save_path}")

    print("--- Comparison Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare 1D FNO models')
    parser.add_argument('--data_path', type=str, default='data/burgers.mat', help='Path to the .mat data file')
    parser.add_argument('--model_path_baseline', type=str, default='models/fno_baseline_1d.pt', help='Path to baseline model weights')
    parser.add_argument('--model_path_rc', type=str, default='models/fno_rc_1d.pt', help='Path to FNO-RC model weights')
    parser.add_argument('--plot_save_dir', type=str, default='comparison_plots_1d', help='Directory to save comparison plots')
    parser.add_argument('--num_plots', type=int, default=5, help='Number of comparison plots to generate')
    
    # Model and data params - should match training scripts
    parser.add_argument('--modes', type=int, default=16, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=64, help='Feature width of the model')
    parser.add_argument('--T_in', type=int, default=10, help='Number of input time steps')
    parser.add_argument('--T_out', type=int, default=10, help='Number of output time steps')

    args = parser.parse_args()
    main(args) 