"""
Trainer for 2D Fourier Neural Operator (with Dynamic Gating) for the 
Navier-Stokes equation.

This script is adapted from the baseline FNO trainer to specifically train and 
evaluate the FNO with per-layer dynamic gating, allowing for a fair comparison.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer
import scipy.io
import os

# Import the DYNAMIC GATING model definition and utilities
from fourier_2d_per_layer_gate import FNO2d_per_layer_gate
from utilities3 import GaussianNormalizer, LpLoss, count_params
from Adam import Adam
from sklearn.model_selection import KFold

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
            # This path is less likely for this script, but kept for robustness
            data = scipy.io.loadmat(args.data_path)['u']
            data = torch.from_numpy(data).float()
        elif file_ext == '.pt':
            data = torch.load(args.data_path)
        else:
            raise ValueError("Unsupported data file format. Please use .mat or .pt")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Data shape: {data.shape}")

    # --- K-Fold Cross-Validation Setup ---
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=0)
    fold_test_losses = []
    
    print(f"\n--- Starting {args.k_folds}-Fold Cross-Validation for Dynamic Gated FNO ---")

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        print(f"\n===== Fold {fold+1}/{args.k_folds} =====")
        
        # --- Data Splitting for current fold ---
        train_data = data[train_ids]
        test_data = data[test_ids]
        
        train_a = train_data[..., :args.T_in]
        train_u = train_data[..., args.T_in:args.T_in + args.T_out] # Corrected Slice
        
        test_a = test_data[..., :args.T_in]
        test_u = test_data[..., args.T_in:args.T_in + args.T_out] # Corrected Slice

        ntrain_fold = len(train_ids)
        ntest_fold = len(test_ids)

        # Normalize data based on the CURRENT training fold
        a_normalizer = GaussianNormalizer(train_a)
        train_a = a_normalizer.encode(train_a)
        test_a = a_normalizer.encode(test_a)

        y_normalizer = GaussianNormalizer(train_u)
        train_u = y_normalizer.encode(train_u)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        a_normalizer.to(device)
        y_normalizer.to(device)

        # Append grid coordinates
        grid_train = get_grid(train_a.shape, device)
        train_a = torch.cat((train_a, grid_train), dim=-1)
        grid_test = get_grid(test_a.shape, device)
        test_a = torch.cat((test_a, grid_test), dim=-1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=args.batch_size, shuffle=False)

        # --- Model Re-initialization for each fold ---
        model = FNO2d_per_layer_gate(args.modes, args.modes, args.width, in_channels=args.T_in + 2, out_channels=args.T_out).to(device)
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        loss_func = LpLoss(size_average=False)

        # --- Training Loop for current fold ---
        for ep in range(args.epochs):
            model.train()
            t1 = default_timer()
            train_l2 = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                loss = loss_func(out.view(out.size(0), args.res, args.res, args.T_out), y.view(y.size(0), args.res, args.res, args.T_out))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    y = y_normalizer.decode(y)
                    test_l2 += loss_func(out.view(out.size(0), args.res, args.res, args.T_out), y.view(y.size(0), args.res, args.res, args.T_out)).item()
            
            train_l2 /= ntrain_fold
            test_l2 /= ntest_fold
            
            if (ep + 1) % 50 == 0:
                print(f"Epoch {ep+1}/{args.epochs} | Train L2: {train_l2:.4f} | Test L2: {test_l2:.4f}")
        
        # --- Final evaluation for current fold ---
        final_test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                final_test_loss += loss_func(out.view(out.size(0), args.res, args.res, args.T_out), y.view(y.size(0), args.res, args.res, args.T_out)).item()
        final_test_loss /= ntest_fold
        print(f"Fold {fold+1} Final Test Error: {final_test_loss:.4f}")
        fold_test_losses.append(final_test_loss)

    # --- Aggregate and Report Final Results ---
    mean_loss = np.mean(fold_test_losses)
    std_loss = np.std(fold_test_losses)
    print("\n\n===== K-Fold Cross-Validation Summary (Dynamic Gated FNO) =====")
    print(f"Individual Fold Test Errors: {[f'{l:.4f}' for l in fold_test_losses]}")
    print(f"Average Test Error: {mean_loss:.4f}")
    print(f"Standard Deviation: {std_loss:.4f}")
    print("==============================================================")
    
    # --- Save Model ---
    if args.model_path:
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        
        torch.save(model.state_dict(), args.model_path)
        print(f"\nModel saved to {args.model_path}")


def get_grid(shape, device):
    """
    Helper function to generate a 2D grid of coordinates.
    The FNO2d_per_layer_gate model requires this to be done externally.
    """
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a 2D FNO with Dynamic Gating and K-Fold Cross-Validation.')
    
    # Paths and data options
    parser.add_argument('--data_path', type=str, default='data/ns_data_N600_clean.pt', help='Path to the .pt data file.')
    parser.add_argument('--model_path', type=str, default='models/fno_dynamic_gate_ns_2d_N600.pt', help='Path to save the trained model.')
    parser.add_argument('--ntrain', type=int, default=500, help='Number of training samples.')
    parser.add_argument('--ntest', type=int, default=100, help='Number of testing samples.')
    parser.add_argument('--res', type=int, default=128, help='Resolution of the data (S).')
    parser.add_argument('--T_in', type=int, default=10, help='Number of input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Number of output time steps.')
    
    # Model hyperparameters
    parser.add_argument('--modes', type=int, default=16, help='Number of Fourier modes to use in each dimension.')
    parser.add_argument('--width', type=int, default=32, help='Width of the FNO (number of channels).')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate.')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation.')

    args = parser.parse_args()
    main(args)
 