"""
Baseline Trainer for the 3D FNO (using fourier_3d_clean.py).
This script is designed to establish a reliable performance baseline for the 3D Navier-Stokes problem.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer
import os
import scipy.io

# Import the CLEAN baseline model definition and utilities
from fourier_3d_clean import FNO3d
from utilities3 import LpLoss, count_params, MatReader, UnitGaussianNormalizer
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

def main(args):
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    try:
        reader = MatReader(args.data_path)
        u_field = reader.read_field('u')
        
        train_a = u_field[:args.ntrain, ..., :args.T_in]
        # The label y is the single frame at the end of the prediction window
        train_u = u_field[:args.ntrain, ..., args.T_in:args.T_in + args.T_out]
        
        test_a = u_field[-args.ntest:, ..., :args.T_in]
        test_u = u_field[-args.ntest:, ..., args.T_in:args.T_in + args.T_out]

        print(f"Data shapes: train_a: {train_a.shape}, train_u: {train_u.shape}")

    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        return

    # --- Normalization and Preprocessing ---
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

    # Reshape input to be broadcastable in the time dimension
    S1, S2 = train_a.shape[1], train_a.shape[2]
    ntrain_actual = train_a.shape[0]
    ntest_actual = test_a.shape[0]
    train_a = train_a.reshape(ntrain_actual, S1, S2, 1, args.T_in).repeat([1,1,1,args.T_out,1])
    test_a = test_a.reshape(ntest_actual, S1, S2, 1, args.T_in).repeat([1,1,1,args.T_out,1])
    
    # IMPORTANT: The model in fourier_3d_clean.py predicts the LAST timestep.
    # So the input 'x' should be the sequence up to T_out, and 'y' is the single frame at T_out.
    train_x = train_a
    train_y = train_u[..., -1] # Target is the last frame
    
    test_x = test_a
    test_y = test_u[..., -1]   # Target is the last frame

    # Create a dedicated normalizer for the last time step for correct decoding
    y_normalizer_last_step = UnitGaussianNormalizer(train_u[..., -1])
    y_normalizer_last_step.to(device)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

    # --- Model Initialization ---
    model = FNO3d(
        modes1=args.modes, 
        modes2=args.modes, 
        modes3=args.modes,
        width=args.width,
        in_channels=args.T_in, # Pass correct in_channels
        out_channels=1
    ).to(device)
    
    print(f"\nModel: Baseline FNO3d (from fourier_3d_clean.py)")
    print(f"Parameters: {count_params(model)}")
    print(f"Hyperparameters: Modes={args.modes}, Width={args.width}, LR={args.learning_rate}")
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    loss_func = LpLoss(size_average=False)
    y_normalizer.to(device)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x) # out shape (B, S1, S2, 1, 1)
            
            # Match dimensions for loss calculation
            loss = loss_func(out.view(x.size(0), S1, S2), y)
            loss.backward()
            
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x) # out shape (B, S1, S2, 1, 1)
                
                # Denormalize for error calculation using the dedicated last-step normalizer
                out_decoded = y_normalizer_last_step.decode(out.view(x.size(0), S1, S2))
                
                # y from the dataloader is already the raw, ground truth last frame
                test_l2 += loss_func(out_decoded, y).item()
        
        train_l2 /= ntrain_actual
        test_l2 /= ntest_actual
        
        t2 = default_timer()
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{args.epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}")
    
    print("--- Training Finished ---")

    if args.model_save_path:
        model_dir = os.path.dirname(args.model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), args.model_save_path)
        print(f"Model saved to {args.model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Baseline 3D FNO.')
    
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat', help='Path to data file.')
    parser.add_argument('--model_save_path', type=str, default='models/fno_baseline_3d.pt', help='Path to save the baseline model.')
    
    parser.add_argument('--ntrain', type=int, default=1000, help='Number of training samples.')
    parser.add_argument('--ntest', type=int, default=200, help='Number of testing samples.')
    
    parser.add_argument('--T_in', type=int, default=10, help='Input time steps.')
    parser.add_argument('--T_out', type=int, default=20, help='Prediction window size.')
    
    # Hyperparameters from fourier_3d.py
    parser.add_argument('--modes', type=int, default=8, help='Fourier modes.')
    parser.add_argument('--width', type=int, default=20, help='Width of the FNO.')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay.')
    parser.add_argument('--scheduler_step', type=int, default=100, help='Scheduler step size.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Scheduler gamma.')
    
    args = parser.parse_args()
    main(args) 