import torch
import numpy as np
import torch.nn as nn
import argparse
import os
from timeit import default_timer
import matplotlib.pyplot as plt

# Import model definitions
from fourier_2d_baseline import FNO2d
from fourier_2d_cft_residual import FNO_RC
from utilities3 import LpLoss, count_params, GaussianNormalizer

def compare_models(args):
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    try:
        data = torch.load(args.data_path, map_location='cpu')
        if data.dim() > 4: data = data.squeeze()
    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        return
    
    ntest = args.ntest
    x_test = data[-ntest:, ..., :args.T_in]
    y_test = data[-ntest:, ..., args.T_in:args.T_in+args.T_out]

    # Normalize test data using the same approach as in training
    # We need to compute normalizer stats on training data part
    ntrain = args.ntrain
    x_train_for_norm = data[:ntrain, ..., :args.T_in]
    y_train_for_norm = data[:ntrain, ..., args.T_in:args.T_in+args.T_out]
    x_normalizer = GaussianNormalizer(x_train_for_norm)
    x_test = x_normalizer.encode(x_test)
    y_normalizer = GaussianNormalizer(y_train_for_norm)
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    # --- Load Models ---
    # Load Baseline FNO
    try:
        model_fno = FNO2d(args.fno_modes, args.fno_modes, args.fno_width, args.T_in, args.T_out).to(device)
        model_fno.load_state_dict(torch.load(args.fno_model_path, map_location=device))
        model_fno.eval()
        print(f"Baseline FNO loaded successfully from {args.fno_model_path}")
    except Exception as e:
        print(f"Error loading baseline FNO model: {e}")
        return

    # Load our FNO-RC
    try:
        model_rc = FNO_RC(args.rc_modes, args.rc_modes, args.rc_width, args.T_in, args.T_out).to(device)
        model_rc.load_state_dict(torch.load(args.rc_model_path, map_location=device))
        model_rc.eval()
        print(f"FNO-RC loaded successfully from {args.rc_model_path}")
    except Exception as e:
        print(f"Error loading FNO-RC model: {e}")
        return

    # --- Evaluation ---
    loss_func = LpLoss(size_average=False)
    total_loss_fno = 0
    total_loss_rc = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Evaluate FNO
            out_fno = model_fno(x)
            out_fno_decoded = y_normalizer.decode(out_fno.cpu()).to(device)
            total_loss_fno += loss_func(out_fno_decoded, y).item()
            
            # Evaluate FNO-RC
            out_rc = model_rc(x)
            out_rc_decoded = y_normalizer.decode(out_rc.cpu()).to(device)
            total_loss_rc += loss_func(out_rc_decoded, y).item()

    avg_loss_fno = total_loss_fno / ntest
    avg_loss_rc = total_loss_rc / ntest

    print("\n--- Final Performance Comparison ---")
    print(f"Baseline FNO      | Average L2 Error: {avg_loss_fno:.6f}")
    print(f"Our FNO-RC        | Average L2 Error: {avg_loss_rc:.6f}")
    print("------------------------------------")
    perf_improvement = ((avg_loss_fno - avg_loss_rc) / avg_loss_fno) * 100
    print(f"Relative Improvement: {perf_improvement:.2f}%")

    # --- Visualization ---
    if args.num_plots > 0:
        print(f"\nGenerating {args.num_plots} comparison plots...")
        plot_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for i, (x, y) in enumerate(plot_loader):
                if i >= args.num_plots:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Get predictions
                pred_fno = y_normalizer.decode(model_fno(x).cpu()).to(device)
                pred_rc = y_normalizer.decode(model_rc(x).cpu()).to(device)

                # Move all to CPU for plotting
                y_cpu = y.cpu()
                pred_fno_cpu = pred_fno.cpu()
                pred_rc_cpu = pred_rc.cpu()

                # Calculate individual errors
                error_fno = loss_func(pred_fno_cpu, y_cpu).item()
                error_rc = loss_func(pred_rc_cpu, y_cpu).item()
                
                # Calculate absolute error fields
                error_field_fno = torch.abs(pred_fno_cpu - y_cpu)
                error_field_rc = torch.abs(pred_rc_cpu - y_cpu)
                vmax = max(error_field_fno.max(), error_field_rc.max()) # Sync colorbars

                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                
                # Plot Ground Truth
                im0 = axes[0].imshow(y_cpu[0, ..., -1], cmap='viridis', interpolation='nearest')
                axes[0].set_title(f'Ground Truth (t={args.T_in+args.T_out-1})')
                fig.colorbar(im0, ax=axes[0])

                # Plot Baseline FNO Absolute Error
                im1 = axes[1].imshow(error_field_fno[0, ..., -1], cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax)
                axes[1].set_title(f'Baseline FNO Abs. Error (Avg: {error_fno:.4f})')
                fig.colorbar(im1, ax=axes[1])

                # Plot FNO-RC Absolute Error
                im2 = axes[2].imshow(error_field_rc[0, ..., -1], cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax)
                axes[2].set_title(f'Our FNO-RC Abs. Error (Avg: {error_rc:.4f})')
                fig.colorbar(im2, ax=axes[2])

                plt.tight_layout()
                save_path = os.path.join(args.plot_save_dir, f'error_comparison_plot_{i}.png')
                if not os.path.exists(args.plot_save_dir):
                    os.makedirs(args.plot_save_dir)
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
        
        plt.close('all')

        # --- Generate 2 Extra Plots as requested for the first sample ---
        print("\nGenerating 2 extra requested plots for the first sample...")
        x, y = next(iter(plot_loader))
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            # Get predictions
            pred_fno = y_normalizer.decode(model_fno(x).cpu()).to(device)
            pred_rc = y_normalizer.decode(model_rc(x).cpu()).to(device)

            y_cpu = y.cpu()
            pred_fno_cpu = pred_fno.cpu()
            pred_rc_cpu = pred_rc.cpu()

            # --- PLOT 1: First Timestep (t=10) ---
            error_field_fno_t0 = torch.abs(pred_fno_cpu[..., 0] - y_cpu[..., 0])
            error_field_rc_t0 = torch.abs(pred_rc_cpu[..., 0] - y_cpu[..., 0])
            vmax = max(error_field_fno_t0.max(), error_field_rc_t0.max())

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            im0 = axes[0].imshow(y_cpu[0, ..., 0].squeeze(0), cmap='viridis', interpolation='nearest')
            axes[0].set_title(f'Ground Truth (t={args.T_in})')
            fig.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(error_field_fno_t0.squeeze(0), cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax)
            axes[1].set_title(f'Baseline FNO Abs. Error (t={args.T_in})')
            fig.colorbar(im1, ax=axes[1])
            im2 = axes[2].imshow(error_field_rc_t0.squeeze(0), cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax)
            axes[2].set_title(f'Our FNO-RC Abs. Error (t={args.T_in})')
            fig.colorbar(im2, ax=axes[2])
            plt.tight_layout()
            save_path = os.path.join(args.plot_save_dir, 'extra_plot_t_start.png')
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
            plt.close(fig)

            # --- PLOT 2: Mean Timestep Error ---
            error_field_fno_mean = torch.mean(torch.abs(pred_fno_cpu - y_cpu), dim=-1)
            error_field_rc_mean = torch.mean(torch.abs(pred_rc_cpu - y_cpu), dim=-1)
            vmax = max(error_field_fno_mean.max(), error_field_rc_mean.max())

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            im0 = axes[0].imshow(y_cpu[0, ..., -1].squeeze(0), cmap='viridis', interpolation='nearest') # Show last frame as context
            axes[0].set_title('Ground Truth (Context)')
            fig.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(error_field_fno_mean.squeeze(0), cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax)
            axes[1].set_title('Baseline FNO Mean Abs. Error')
            fig.colorbar(im1, ax=axes[1])
            im2 = axes[2].imshow(error_field_rc_mean.squeeze(0), cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax)
            axes[2].set_title('Our FNO-RC Mean Abs. Error')
            fig.colorbar(im2, ax=axes[2])
            plt.tight_layout()
            save_path = os.path.join(args.plot_save_dir, 'extra_plot_t_mean.png')
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare FNO-RC and Baseline FNO.')
    
    # Paths
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_data_N600_clean.pt', help='Path to data file.')
    parser.add_argument('--fno_model_path', type=str, default='/content/drive/MyDrive/my_fno_models/fno_ns_2d_N600.pt', help='Path to the baseline FNO model.')
    parser.add_argument('--rc_model_path', type=str, default='models/fno_rc.pt', help='Path to our FNO-RC model.')

    # Data Params
    parser.add_argument('--ntrain', type=int, default=1000, help='Number of training samples for normalizer.')
    parser.add_argument('--ntest', type=int, default=200, help='Number of testing samples.')
    parser.add_argument('--T_in', type=int, default=10, help='Input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Output time steps.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation.')
    
    # Model Hyperparameters (must match the trained models)
    parser.add_argument('--fno_modes', type=int, default=16, help='Modes for baseline FNO.')
    parser.add_argument('--fno_width', type=int, default=32, help='Width for baseline FNO.')
    parser.add_argument('--rc_modes', type=int, default=16, help='Modes for FNO-RC.')
    parser.add_argument('--rc_width', type=int, default=32, help='Width for FNO-RC.')

    # Plotting
    parser.add_argument('--num_plots', type=int, default=10, help='Number of comparison plots to generate.')
    parser.add_argument('--plot_save_dir', type=str, default='error_comparison_plots', help='Directory to save plots.')

    args = parser.parse_args()
    compare_models(args) 