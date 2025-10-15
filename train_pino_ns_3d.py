import torch
import numpy as np
import torch.nn as nn
import h5py
from tqdm import tqdm
import torch.nn.functional as F
import os
import scipy.io
from utilities3 import LpLoss, count_params, save_checkpoint
from fourier_3d_clean import FNO3d
from fourier_pino import NavierStokesLoss

# ##############################################################################
# This script trains the standard FNO model on the 3D Navier-Stokes dataset.
# It serves as the baseline for comparing against experimental models.
# ##############################################################################

def main(args):
    # ################################
    # Configurations
    # ################################
    # Path for Google Colab execution from mounted Google Drive
    TRAIN_PATH = args.data_path
    # Model path, saved to Google Drive for persistence
    MODEL_PATH = args.save_path
    
    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Dataset and Loader parameters
    ntrain = 1000
    ntest = 200
    T_in = args.T
    T_out = 1 # Predict the next timestep

    # Model parameters
    modes = args.modes1, args.modes2, args.modes3
    width = args.width

    # Training parameters
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    weight_decay = args.weight_decay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ################################
    # Data Loading and Preparation
    # ################################
    print("Loading data...")
    # This .mat file is in MATLAB v7.3 format, which requires h5py.
    with h5py.File(TRAIN_PATH, 'r') as f:
        # Data in the file is likely stored as (width, height, time, N).
        # We permute it to the standard (N, height, width, time) for PyTorch.
        u = torch.from_numpy(f['u'][()]).float()
        u = u.permute(3, 1, 0, 2)

    # The shape from scipy is (N, H, W, T_total)
    # We need to bring it to (N, H, W, T_in, C) for the model
    # The channel dimension is implicit, so we add it.
    
    train_x = u[:ntrain, ..., :T_in].unsqueeze(-1)
    train_y = u[:ntrain, ..., T_in:T_in+T_out].unsqueeze(-1)
    test_x = u[ntrain:ntrain+ntest, ..., :T_in].unsqueeze(-1)
    test_y = u[ntrain:ntrain+ntest, ..., T_in:T_in+T_out].unsqueeze(-1)
    
    print(f"Data prepared. Train x: {train_x.shape}, Train y: {train_y.shape}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size, shuffle=False
    )
    
    # ################################
    # Model, Optimizer, and Loss
    # ################################
    model = FNO3d(args.modes1, args.modes2, args.modes3, args.width).to(device)
    num_params = count_params(model)
    print(f"Training a PINO model with {num_params/1e6:.2f}M parameters.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss functions
    data_loss_fn = LpLoss(size_average=False)
    # Note: Viscosity is hardcoded based on the dataset name 'ns_V1e-4...'
    pde_loss_fn = NavierStokesLoss(s=args.s, T=args.T, viscosity=args.viscosity, dt=1.0, device=device)

    # ################################
    # Training Loop
    # ################################
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_l2 = 0
        train_pde = 0 # To track the physics loss
        
        # Wrap the loader with tqdm for a progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', dynamic_ncols=True)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(x)
            
            # --- Combined Loss ---
            # 1. Data Fidelity Loss
            data_loss = data_loss_fn(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
            
            # 2. Physics-Informed Loss
            # We need the prediction and the last time-step of the input
            w_pred = out.squeeze(-1).squeeze(-1) # Shape: (batch, s, s)
            w_prev = x[..., -1, :].squeeze(-1)   # Shape: (batch, s, s)
            pde_loss = pde_loss_fn(w_pred, w_prev)
            
            # 3. Total Loss
            loss = data_loss + args.pde_weight * pde_loss
            
            loss.backward()
            optimizer.step()
            
            train_l2 += data_loss.item()
            train_pde += pde_loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Train L2': data_loss.item(), 'PDE Loss': pde_loss.item()})

        scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += data_loss_fn(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()
        
        # Normalize losses by the number of samples/batches for consistent reporting
        train_loss_l2 = train_l2 / ntrain
        train_loss_pde = train_pde / len(train_loader)
        test_loss_l2 = test_l2 / ntest
        
        print(f"Epoch: {epoch} | Train L2: {train_loss_l2:.6f} | Train PDE: {train_loss_pde:.6f} | Test L2: {test_loss_l2:.6f}")
        
    save_checkpoint(MODEL_PATH, model, optimizer)
    print(f"Training complete. PINO model saved to {MODEL_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a PINO model on the 3D Navier-Stokes dataset')
    parser.add_argument('--s', type=int, default=64, help='Spatial resolution')
    parser.add_argument('--T', type=int, default=10, help='Input time steps')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--width', type=int, default=20, help='Width of the FNO layers')
    parser.add_argument('--modes1', type=int, default=8, help='Fourier modes in x')
    parser.add_argument('--modes2', type=int, default=8, help='Fourier modes in y')
    parser.add_argument('--modes3', type=int, default=8, help='Fourier modes in t')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--pde_weight', type=float, default=1.0, help='Weight for the PDE loss term')
    parser.add_argument('--viscosity', type=float, default=1e-4, help='Fluid viscosity')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat', help='Path to the dataset')
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/fno_models/fno_pino.pt', help='Path to save the trained model')
    
    args = parser.parse_args()
    main(args) 