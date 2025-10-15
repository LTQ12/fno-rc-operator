"""
Trainer for 2D Fourier Neural Operator

This script is designed to train and evaluate the standard 2D FNO model
for the Darcy Flow problem. It serves as the baseline for comparing
against the new Gated FNO 2D architecture.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer

# Import model definition and utilities
from fourier_2d import FNO2d
from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss, count_params
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

# ###############################################################
# # Main Training Logic
# ###############################################################
def main(args):
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # --- Config ---
    modes = args.modes
    width = args.width
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    # Darcy-Flow specific parameters
    r = 5
    s = int(((421 - 1) / r) + 1) # Resolution of the data

    # --- Data Loading ---
    print("Loading data...")
    try:
        datareader = MatReader(args.data_path)
        
        # Diagnostic print
        print("Shape of 'lognorm_p' field:", datareader.read_field('lognorm_p').shape)
        
        x_train_raw = datareader.read_field('lognorm_p')
        y_train_raw = datareader.read_field('thresh_p')
        
        # Assuming the data is stored as (n_samples, n_x * n_y)
        # We need to reshape it to (n_samples, n_x, n_y)
        # The original paper uses 421x421 resolution
        n_total = x_train_raw.shape[0]
        orig_s = 421 
        
        x_train = x_train_raw[:args.ntrain].reshape(args.ntrain, orig_s, orig_s)[:, ::r, ::r]
        y_train = y_train_raw[:args.ntrain].reshape(args.ntrain, orig_s, orig_s)[:, ::r, ::r]

        x_test_raw = datareader.read_field('lognorm_p')[-args.ntest:]
        y_test_raw = datareader.read_field('thresh_p')[-args.ntest:]
        x_test = x_test_raw.reshape(args.ntest, orig_s, orig_s)[:, ::r, ::r]
        y_test = y_test_raw.reshape(args.ntest, orig_s, orig_s)[:, ::r, ::r]
        
    except FileNotFoundError:
        print(f"Error: Data file not found at '{args.data_path}'.")
        # Attempt to download the canonical dataset if not found
        print("Attempting to download Darcy-Flow dataset...")
        os.system('wget https://github.com/zongyi-li/fourier_neural_operator/raw/master/data/piececonst_r421_N1024_smooth.mat')
        if os.path.exists('piececonst_r421_N1024_smooth.mat'):
             print("Download successful. Please re-run the script with --data_path piececonst_r421_N1024_smooth.mat")
        else:
             print("Download failed. Please manually download the data.")
        return

    # Normalize data
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    # Reshape for model input
    x_train = x_train.reshape(args.ntrain, s, s, 1)
    x_test = x_test.reshape(args.ntest, s, s, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    print("Data loaded and preprocessed successfully.")

    # --- Model, Optimizer, and Loss ---
    model = FNO2d(modes, modes, width).to(device)
    print(f"\n--- Training Standard FNO-2D ---")
    print(f"Total Parameters: {count_params(model)}")
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    loss_func = LpLoss(size_average=False)
    
    y_normalizer.to(device)

    # --- Training Loop ---
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x).squeeze()
            
            # Denormalize for loss calculation
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = loss_func(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        train_l2 /= args.ntrain

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze()
                out = y_normalizer.decode(out)
                test_l2 += loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()
        
        test_l2 /= args.ntest
        t2 = default_timer()
        print(f"Epoch: {ep+1}/{epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.4f} | Test L2: {test_l2:.4f}")

    print("\nTraining finished.")
    
    # Final evaluation on test set
    final_test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze()
            out = y_normalizer.decode(out)
            final_test_loss += loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    final_test_loss /= args.ntest
    print(f"\n--- Final Test Error ---")
    print(f"L2 Relative Error: {final_test_loss:.4f}")
    print("------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a 2D FNO for Darcy Flow.')
    
    # Paths and data options
    parser.add_argument('--data_path', type=str, default='piececonst_r421_N1024_smooth.mat', help='Path to the .mat file for Darcy Flow.')
    parser.add_argument('--ntrain', type=int, default=1000, help='Number of training samples.')
    parser.add_argument('--ntest', type=int, default=100, help='Number of testing samples.')
    
    # Model hyperparameters
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes to use in each dimension.')
    parser.add_argument('--width', type=int, default=32, help='Width of the FNO (number of channels).')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate.')

    args = parser.parse_args()
    main(args) 