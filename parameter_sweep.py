"""
Parameter Sweep for Gated Fourier Neural Operator (G-FNO)

This script systematically tests different hyperparameters for the CFT-based
gating mechanism in the G-FNO to find an optimal balance between
training speed and model accuracy.
"""

import torch
import numpy as np
import argparse
from timeit import default_timer
import pandas as pd

# Import necessary components from the training script
# We will reuse data loading, loss functions, etc.
from train_fno_cft import MatReader, LpLoss, count_params
from fourier_1d_gated import GatedFNO1d

# ###############################################################
# # Simplified Training Function
# ###############################################################
def train_for_sweep(model, config, train_loader, test_loader, device):
    """
    A simplified training loop that runs for a fixed number of epochs
    and returns the average time per epoch and the final test loss.
    """
    epochs = config['epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    loss_func = LpLoss(size_average=False)
    
    total_time = 0
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
            loss.backward()
            optimizer.step()
        t2 = default_timer()
        total_time += (t2 - t1)

    # Final evaluation
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_l2 += loss_func(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()
    
    final_test_loss = test_l2 / len(test_loader.dataset)
    avg_time_per_epoch = total_time / epochs
    
    return avg_time_per_epoch, final_test_loss

# ###############################################################
# # Main Sweep Logic
# ###############################################################
def main(args):
    # --- Device and Base Config ---
    device = torch.device('cpu')
    print("Using CPU for parameter sweep.")
    
    config = {
        'learning_rate': 0.001,
        'epochs': args.epochs,
        'modes': 16,
        'width': 64,
        'batch_size': 20,
        'subsampling_rate': 2**3,
    }
    config['s'] = 2**13 // config['subsampling_rate']

    # --- Data Loading (reusing from train script) ---
    print("Loading data...")
    reader = MatReader('burgers_data_R10.mat')
    x_data = reader.read_field('a')[:, ::config['subsampling_rate']]
    y_data = reader.read_field('u')[:, ::config['subsampling_rate']]
    ntrain, ntest = 1000, 100
    x_train = x_data[:ntrain, :].reshape(ntrain, config['s'], 1)
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :].reshape(ntest, config['s'], 1)
    y_test = y_data[-ntest:, :]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=config['batch_size'], shuffle=False)
    print("Data loaded.")

    # --- Parameter Grid ---
    l_values = [int(l) for l in args.l_values]
    m_values = [int(m) for m in args.m_values]
    
    results = []

    print(f"\n--- Starting Parameter Sweep ---")
    print(f"L (segments) to test: {l_values}")
    print(f"M (Chebyshev order) to test: {m_values}")
    print(f"Epochs per run: {args.epochs}\n")

    # --- Loop through all parameter combinations ---
    for l in l_values:
        for m in m_values:
            torch.manual_seed(0)
            np.random.seed(0)
            
            print(f"Testing: L={l}, M={m}...")
            
            gated_params = {'cft_L_segments': l, 'cft_M_cheb': m}
            model = GatedFNO1d(config['modes'], config['width'], **gated_params).to(device)
            
            avg_time, final_loss = train_for_sweep(model, config, train_loader, test_loader, device)
            
            results.append({
                'L': l,
                'M': m,
                'Avg Time/Epoch (s)': f"{avg_time:.2f}",
                'Final Test Loss': f"{final_loss:.4f}"
            })
            print(f"--> Done. Avg Time: {avg_time:.2f}s/epoch, Final Loss: {final_loss:.4f}\n")

    # --- Display Results ---
    print("--- Sweep Complete. Results: ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a parameter sweep for the G-FNO model.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs for each parameter combination.')
    parser.add_argument('--l_values', nargs='+', default=[5, 10, 20], help='List of L values (segments) to test.')
    parser.add_argument('--m_values', nargs='+', default=[4, 8], help='List of M values (Chebyshev order) to test.')
    
    args = parser.parse_args()
    main(args) 