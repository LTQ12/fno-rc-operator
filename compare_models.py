import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from fourier_2d_cft_attention import FNO_CFT_Attention
from fourier_2d_baseline import FNO2d
from utilities3 import LpLoss, GaussianNormalizer

torch.manual_seed(0)
np.random.seed(0)

def compare_models(args):
    """
    Compares the performance of the CFT-Gated Attention FNO against the baseline FNO.
    """
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    try:
        data = torch.load(args.data_path)
        if data.dim() > 4: data = data.squeeze()
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        return

    # --- Data Splitting and Normalization ---
    ntest = args.ntest
    test_data = data[-ntest:]
    
    # We need a normalizer from the training set to correctly process test data
    # Assuming the first ntrain samples were used for training
    ntrain = args.ntrain
    train_data = data[:ntrain]
    
    x_train = train_data[..., :args.T_in]
    y_train = train_data[..., args.T_in:args.T_in+args.T_out]
    
    x_test = test_data[..., :args.T_in]
    y_test = test_data[..., :args.T_in:args.T_in+args.T_out]

    x_normalizer = GaussianNormalizer(x_train)
    x_test = x_normalizer.encode(x_test)
    y_normalizer = GaussianNormalizer(y_train)

    x_test = x_test.to(device)
    y_normalizer.to(device)

    # --- Model Initialization and Loading ---
    # 1. CFT-Gated Attention FNO (Our Model)
    model_cft_attention = FNO_CFT_Attention(
        modes1=args.cft_modes, 
        modes2=args.cft_modes, 
        width=args.cft_width,
        in_channels=args.T_in,
        out_channels=args.T_out
    ).to(device)
    try:
        model_cft_attention.load_state_dict(torch.load(args.cft_model_path, map_location=device))
        model_cft_attention.eval()
        print(f"CFT-Attention FNO loaded successfully from {args.cft_model_path}")
    except Exception as e:
        print(f"Error loading CFT-Attention FNO model: {e}")
        return
        
    # 2. Baseline FNO
    model_baseline = FNO2d(
        modes1=args.fno_modes,
        modes2=args.fno_modes,
        width=args.fno_width,
        in_channels=args.T_in,
        out_channels=args.T_out
    ).to(device)
    try:
        model_baseline.load_state_dict(torch.load(args.fno_model_path, map_location=device))
        model_baseline.eval()
        print(f"Baseline FNO loaded successfully from {args.fno_model_path}")
    except Exception as e:
        print(f"Error loading Baseline FNO model: {e}")
        # Add a check to see if the fourier_2d_baseline needs to be used for old models
        if "in_channels" in str(e):
             print("Hint: The baseline model might be an older version. Trying to load from 'fourier_2d' definition...")
             from fourier_2d import FNO2d as FNO2d_legacy
             model_baseline = FNO2d_legacy(args.fno_modes, args.fno_modes, args.fno_width).to(device)
             model_baseline.load_state_dict(torch.load(args.fno_model_path, map_location=device))
             model_baseline.eval()
             print("Successfully loaded legacy Baseline FNO.")
        else:
            return

    # --- Evaluation ---
    loss_func = LpLoss(size_average=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    total_loss_cft = 0
    total_loss_fno = 0
    
    preds_cft = []
    preds_fno = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # CFT-Attention FNO prediction
            out_cft = y_normalizer.decode(model_cft_attention(x))
            total_loss_cft += loss_func(out_cft, y).item()
            preds_cft.append(out_cft.cpu())

            # Baseline FNO prediction
            out_fno = y_normalizer.decode(model_baseline(x))
            total_loss_fno += loss_func(out_fno, y).item()
            preds_fno.append(out_fno.cpu())

    avg_loss_cft = total_loss_cft / ntest
    avg_loss_fno = total_loss_fno / ntest

    print("\n--- Performance Comparison ---")
    print(f"CFT-Gated Attention FNO | Average L2 Error: {avg_loss_cft:.6f}")
    print(f"Baseline FNO              | Average L2 Error: {avg_loss_fno:.6f}")
    print("----------------------------")
    
    # Concatenate predictions from all batches
    preds_cft = torch.cat(preds_cft, dim=0)
    preds_fno = torch.cat(preds_fno, dim=0)

    # --- Visualization ---
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Move y_test to CPU once before the loop for plotting and loss calculation
    y_test_cpu = y_test.cpu()

    for i in range(args.num_plots):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot Ground Truth
        im = axes[0].imshow(y_test_cpu[i, :, :, -1], cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Ground Truth (t={args.T_out-1})')
        fig.colorbar(im, ax=axes[0])

        # Plot Baseline FNO Prediction
        im = axes[1].imshow(preds_fno[i, :, :, -1], cmap='viridis', interpolation='nearest')
        axes[1].set_title(f'Baseline FNO Pred. (Error: {loss_func(preds_fno[i:i+1], y_test_cpu[i:i+1]).item():.4f})')
        fig.colorbar(im, ax=axes[1])

        # Plot CFT-Attention FNO Prediction
        im = axes[2].imshow(preds_cft[i, :, :, -1], cmap='viridis', interpolation='nearest')
        axes[2].set_title(f'CFT-Attention FNO Pred. (Error: {loss_func(preds_cft[i:i+1], y_test_cpu[i:i+1]).item():.4f})')
        fig.colorbar(im, ax=axes[2])
        
        fig.suptitle(f'Comparison for Test Sample {i}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = f'plots/comparison_sample_{i}.png'
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FNO models.")
    # Paths
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_data_N600_clean.pt', help='Path to the data file.')
    parser.add_argument('--cft_model_path', type=str, default='models/cft_attention_fno_ns_2d_retrain.pt', help='Path to the retrained CFT-Attention FNO model.')
    parser.add_argument('--fno_model_path', type=str, required=True, help='Path to the baseline FNO model.')
    # Data params
    parser.add_argument('--ntrain', type=int, default=1000, help='Number of training samples for normalizer.')
    parser.add_argument('--ntest', type=int, default=200, help='Number of test samples.')
    parser.add_argument('--T_in', type=int, default=10, help='Input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Output time steps.')
    # CFT-Attention FNO params
    parser.add_argument('--cft_modes', type=int, default=16, help='Modes for CFT-Attention FNO.')
    parser.add_argument('--cft_width', type=int, default=32, help='Width for CFT-Attention FNO.')
    # Baseline FNO params
    parser.add_argument('--fno_modes', type=int, default=16, help='Modes for baseline FNO.')
    parser.add_argument('--fno_width', type=int, default=32, help='Width for baseline FNO.')
    # Eval params
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation.')
    parser.add_argument('--num_plots', type=int, default=5, help='Number of comparison plots to generate.')
    
    args = parser.parse_args()
    compare_models(args)