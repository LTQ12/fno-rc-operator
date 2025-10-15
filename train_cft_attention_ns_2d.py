
"""
Trainer for 2D CFT-Gated Fourier Attention Network for the Navier-Stokes equation.

This script is designed to train and evaluate our custom CFT-gated attention
model, comparing its performance against the baseline FNO.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer
import scipy.io
import os
import sys

# Add parent directory to path to import model and utilities
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fourier_2d_cft_attention import FNO_CFT_Attention
from utilities3 import LpLoss, count_params, GaussianNormalizer
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
# Main Logic
################################################################
def main(args):
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # --- Data Loading ---
    print("Loading data...")
    try:
        # For .mat, data is a dict. For .pt, it's a tensor.
        file_ext = os.path.splitext(args.data_path)[1]
        if file_ext == '.mat':
            data = scipy.io.loadmat(args.data_path)
            # Adjust key based on common patterns in .mat files from this project
            data_key = 'u' if 'u' in data else list(data.keys())[-1]
            data = torch.from_numpy(data[data_key]).float()
        elif file_ext == '.pt':
            data = torch.load(args.data_path)
        else:
            raise ValueError("Unsupported data file format. Please use .mat or .pt")
        
        # Squeeze out any singleton dimensions
        if data.dim() > 4:
            data = data.squeeze()

    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        return
    
    print(f"Data shape: {data.shape}")

    # --- Data Splitting ---
    ntrain = args.ntrain
    ntest = args.ntest
    
    # Assuming data shape: [num_samples, x, y, time]
    x_train = data[:ntrain, ..., :args.T_in]
    y_train = data[:ntrain, ..., args.T_in:args.T_in+args.T_out]
    x_test = data[-ntest:, ..., :args.T_in]
    y_test = data[-ntest:, ..., args.T_in:args.T_in+args.T_out]

    # --- Normalization ---
    x_normalizer = GaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = GaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    
    x_normalizer.to(device)
    y_normalizer.to(device)

    # --- Dataloaders ---
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    # --- Model Initialization ---
    model = FNO_CFT_Attention(args.modes, args.modes, args.width, in_channels=args.T_in, out_channels=args.T_out).to(device)
    print(f"Model: CFT-Gated Fourier Attention")
    print(f"Parameters: {count_params(model)}")
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_func = LpLoss(size_average=False)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            # Denormalize for loss calculation
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = loss_func(out.view(out.size(0), -1), y.view(y.size(0), -1))
            loss.backward()
            
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                out = model(x)
                out = y_normalizer.decode(out)
                # y is already denormalized in test loader as well
                
                test_l2 += loss_func(out.view(out.size(0), -1), y.view(y.size(0), -1)).item()
        
        train_l2 /= ntrain
        test_l2 /= ntest
        
        t2 = default_timer()
        if (ep + 1) % 50 == 0:
            print(f"Epoch {ep+1}/{args.epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.4f} | Test L2: {test_l2:.4f}")
    
    print("--- Training Finished ---")

    # --- Final Evaluation ---
    final_test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = y_normalizer.decode(out)
            final_test_loss += loss_func(out.view(out.size(0), -1), y.view(y.size(0), -1)).item()
    final_test_loss /= ntest
    print(f"\nFinal Average Test Error: {final_test_loss:.4f}")

    # --- Save Model ---
    if args.model_save_path:
        model_dir = os.path.dirname(args.model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        
        torch.save(model.state_dict(), args.model_save_path)
        print(f"Model saved to {args.model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CFT-Gated Fourier Attention FNO for 2D Navier-Stokes.')
    
    # Paths and data options
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_data_N600_clean.pt', help='Path to the data file.')
    parser.add_argument('--model_save_path', type=str, default='models/cft_attention_fno_ns_2d_retrain.pt', help='Path to save the trained model.')
    parser.add_argument('--ntrain', type=int, default=1000, help='Number of training samples.')
    parser.add_argument('--ntest', type=int, default=200, help='Number of testing samples.')
    parser.add_argument('--T_in', type=int, default=10, help='Number of input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Number of output time steps.')
    
    # Model hyperparameters
    parser.add_argument('--modes', type=int, default=16, help='Number of Fourier modes to use in each dimension.')
    parser.add_argument('--width', type=int, default=32, help='Width of the FNO (number of channels).')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate.')
    
    args = parser.parse_args()
    main(args) 