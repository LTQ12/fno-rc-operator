
"""
Trainer for the final proposed model: 
CFT-Driven Dynamic Filter Fourier Neural Operator for 2D Navier-Stokes.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer
import os

# Import the final, low-rank model definition and utilities
from fourier_2d_cft_dynamic_filter import FNO_CFT_LowRank_Filter
from utilities3 import LpLoss, count_params, GaussianNormalizer
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
        data = torch.load(args.data_path)
        if data.dim() > 4: data = data.squeeze()
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        return

    # --- Data Splitting & Normalization ---
    ntrain = args.ntrain
    ntest = args.ntest
    
    x_train = data[:ntrain, ..., :args.T_in]
    y_train = data[:ntrain, ..., args.T_in:args.T_in+args.T_out]
    x_test = data[-ntest:, ..., :args.T_in]
    y_test = data[-ntest:, ..., args.T_in:args.T_in+args.T_out]

    x_normalizer = GaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = GaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    
    x_normalizer.to(device)
    y_normalizer.to(device)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    # --- Model Initialization ---
    model = FNO_CFT_LowRank_Filter(
        modes1=args.modes, 
        modes2=args.modes, 
        width=args.width,
        in_channels=args.T_in,
        out_channels=args.T_out
    ).to(device)
    
    print(f"\nModel: CFT-Driven Low-Rank Filter FNO")
    print(f"Parameters: {count_params(model)}")
    print(f"Hyperparameters: Modes={args.modes}, Width={args.width}, LR={args.learning_rate}")
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
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
            
            out_decoded = y_normalizer.decode(out)
            y_decoded = y_normalizer.decode(y)
            loss = loss_func(out_decoded, y_decoded)
            loss.backward()
            
            # Add gradient clipping for stability
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
                out_decoded = y_normalizer.decode(out)
                y_decoded = y_normalizer.decode(y)
                test_l2 += loss_func(out_decoded, y_decoded).item()
        
        train_l2 /= ntrain
        test_l2 /= ntest
        
        t2 = default_timer()
        if (ep + 1) % 50 == 0:
            print(f"Epoch {ep+1}/{args.epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}")
    
    print("--- Training Finished ---")

    # --- Final Evaluation & Saving ---
    final_test_loss = test_l2 # From the last epoch
    print(f"\nFinal Average Test Error: {final_test_loss:.6f}")

    if args.model_save_path:
        model_dir = os.path.dirname(args.model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        
        torch.save(model.state_dict(), args.model_save_path)
        print(f"Model saved to {args.model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CFT-Driven Low-Rank Dynamic Filter FNO.')
    
    # Paths and data options
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_data_N600_clean.pt', help='Path to data file.')
    parser.add_argument('--model_save_path', type=str, default='models/cft_lowrank_filter_fno.pt', help='Path to save the trained model.')
    
    # Data params from training script
    parser.add_argument('--ntrain', type=int, default=1000, help='Number of training samples.')
    parser.add_argument('--ntest', type=int, default=200, help='Number of testing samples.')
    parser.add_argument('--T_in', type=int, default=10, help='Input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Output time steps.')
    
    # Model hyperparameters aligned with baseline FNO
    parser.add_argument('--modes', type=int, default=16, help='Fourier modes.')
    parser.add_argument('--width', type=int, default=32, help='Width of the FNO.')
    
    # Training hyperparameters aligned with baseline FNO
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    
    args = parser.parse_args()
    main(args) 