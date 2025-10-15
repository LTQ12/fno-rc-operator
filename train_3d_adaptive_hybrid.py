import torch
import numpy as np
import torch.nn as nn
import h5py
from tqdm import tqdm
import torch.nn.functional as F
import os

from utilities3 import LpLoss, save_checkpoint
from fourier_3d_adaptive_hybrid import FNO_3d_AH

# ##############################################################################
# This script trains the Adaptive Hybrid FNO (FNO-AH) model.
# For this first implementation, it's an "Adaptive FNO" where a gating
# network adaptively modulates the frequency components of the standard FNO path,
# making it data-dependent.
# ##############################################################################

def main():
    # ################################
    # Configurations
    # ################################
    # Path for Google Colab execution from mounted Google Drive
    TRAIN_PATH = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
    # Model path, saved to Google Drive for persistence
    MODEL_PATH = '/content/drive/MyDrive/fno_models/fno_3d_adaptive_hybrid.pt'

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Dataset and Loader parameters
    ntrain = 1000
    ntest = 200
    T_in = 10
    T_out = 1

    # Model parameters
    modes = 8
    width = 20
    
    # Training parameters
    batch_size = 10
    learning_rate = 0.001
    epochs = 50
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ################################
    # Data Loading and Preparation
    # ################################
    print("Loading data...")
    with h5py.File(TRAIN_PATH, 'r') as f:
        # Ground truth shape: (T, x, y, N) -> (50, 64, 64, 10000)
        all_data = torch.from_numpy(f['u'][()]).float()
        # Permute to (N, x, y, T) as expected by the model logic
        all_data = all_data.permute(3, 1, 2, 0)

    all_data = all_data[:ntrain + ntest, ...]

    train_x = all_data[:ntrain, :, :, :T_in].unsqueeze(-1)
    train_y = all_data[:ntrain, :, :, T_in:T_in+T_out].unsqueeze(-1)
    
    test_x = all_data[ntrain:ntrain+ntest, :, :, :T_in].unsqueeze(-1)
    test_y = all_data[ntrain:ntrain+ntest, :, :, T_in:T_in+T_out].unsqueeze(-1)
    
    assert train_x.shape == (ntrain, 64, 64, T_in, 1)

    print(f"Data prepared. Train x: {train_x.shape}, Train y: {train_y.shape}")
    print(f"Test x: {test_x.shape}, Test y: {test_y.shape}")

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
    model = FNO_3d_AH(
        modes1=modes, modes2=modes, modes3=modes,
        width=width,
        in_channels=1,
        out_channels=1
    ).to(device)
    
    print(f"Training FNO-AH with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    myloss = LpLoss(size_average=False)

    # ################################
    # Training Loop
    # ################################
    for ep in range(epochs):
        model.train()
        train_l2 = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", unit="batch"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            loss = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
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
                test_l2 += myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()
        
        train_l2 /= ntrain
        test_l2 /= ntest
        
        print(f"Epoch: {ep+1} | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}")
        
    save_checkpoint(MODEL_PATH, model, optimizer)
    print(f"Training complete. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main() 