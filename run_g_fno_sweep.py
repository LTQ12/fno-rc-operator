import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer
import os

from fourier_2d_g_fno import GFNO2d
from utilities3 import GaussianNormalizer, LpLoss, count_params
from Adam import Adam

# ==============================================================================
# G-FNO "Selection Race" Runner
# ==============================================================================
# This script systematically tests different configurations of the G-FNO model
# to find the optimal setup before running a full K-fold cross-validation.
# ==============================================================================


def get_grid(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def run_single_experiment(config):
    """
    Runs a single training and evaluation experiment for one G-FNO configuration.
    """
    print("-" * 60)
    print(f"Running Experiment with Config:")
    print(f"  Slicing Method: {config['slicing_method']}")
    print(f"  CFT Trainable: {config['cft_trainable']}")
    print("-" * 60)

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Data Loading ---
    data = torch.load(config['data_path'])
    train_data = data[:config['ntrain']]
    test_data = data[-config['ntest']:]
    
    train_a = train_data[..., :config['T_in']]
    train_u = train_data[..., config['T_in']:config['T_in'] + config['T_out']]
    test_a = test_data[..., :config['T_in']]
    test_u = test_data[..., config['T_in']:config['T_in'] + config['T_out']]

    a_normalizer = GaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = GaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)
    
    a_normalizer.to(device)
    y_normalizer.to(device)

    grid_train = get_grid(train_a.shape, device)
    train_a = torch.cat((train_a, grid_train), dim=-1)
    grid_test = get_grid(test_a.shape, device)
    test_a = torch.cat((test_a, grid_test), dim=-1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=config['batch_size'], shuffle=False)

    # --- Model Definition ---
    model = GFNO2d(
        modes1=config['modes'],
        modes2=config['modes'],
        width=config['width'],
        in_channels=config['T_in'] + 2,
        out_channels=config['T_out'],
        cft_L=config['cft_L'],
        cft_M=config['cft_M'],
        slicing_method=config['slicing_method'],
        cft_trainable=config['cft_trainable']
    ).to(device)
    
    print(f"Model Parameters: {count_params(model)}")

    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    loss_func = LpLoss(size_average=False)

    # --- Training Loop ---
    for ep in range(config['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            out = y_normalizer.decode(out)
            y_decoded = y_normalizer.decode(y)
            loss = loss_func(out.view(out.size(0), -1), y_decoded.view(y.size(0), -1))
            loss.backward()
            optimizer.step()
        scheduler.step()

    # --- Final Evaluation ---
    model.eval()
    final_test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = y_normalizer.decode(out)
            y_decoded = y_normalizer.decode(y)
            final_test_loss += loss_func(out.view(out.size(0), -1), y_decoded.view(y.size(0), -1)).item()
    
    final_test_loss /= config['ntest']
    print(f"Final Test Error: {final_test_loss:.6f}")
    return final_test_loss


if __name__ == "__main__":
    # --- Base Config ---
    base_config = {
        'data_path': 'data/ns_data_N600_clean.pt',
        'ntrain': 500,
        'ntest': 100,
        'T_in': 10,
        'T_out': 10,
        'modes': 16,
        'width': 32,
        'epochs': 100, # Quick sweep
        'batch_size': 20,
        'learning_rate': 0.001,
        'cft_L': 10, # Using values from previous successful experiments
        'cft_M': 16,
    }

    # --- Experiment Configurations ---
    experiments = {
        "A_Center_Detached": {
            "slicing_method": "center",
            "cft_trainable": False,
        },
        "B_MeanPool_Detached": {
            "slicing_method": "mean_pool",
            "cft_trainable": False,
        },
        "C_Center_Trainable": {
            "slicing_method": "center",
            "cft_trainable": True,
        }
    }

    results = {}
    
    print("========= Starting G-FNO Configuration Sweep =========")
    for name, config_update in experiments.items():
        config = base_config.copy()
        config.update(config_update)
        test_error = run_single_experiment(config)
        results[name] = test_error
    
    print("\n\n========= Sweep Results =========")
    for name, error in results.items():
        print(f"Configuration '{name}': Final Test Error = {error:.6f}")
    
    best_config_name = min(results, key=results.get)
    print(f"\nBest performing configuration: '{best_config_name}'")
    print("=================================") 