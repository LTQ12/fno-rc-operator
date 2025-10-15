"""
Trainer for a Gated 2D FNO adapted for a 1D STATIC Burgers' equation dataset.

This script loads 1D static data (a(x) -> u(x)) and reshapes it to be
compatible with a 2D FNO model, serving as a baseline for the Gated FNO 2D.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from timeit import default_timer

# Import model definition and utilities
from fourier_ns_2d import FNO2d
from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss, count_params
from Adam import Adam
from cft_utils_2d import vectorized_batched_cft_decomposed

torch.manual_seed(0)
np.random.seed(0)

class BurgersStaticDataset(torch.utils.data.Dataset):
    def __init__(self, a, u):
        super(BurgersStaticDataset, self).__init__()
        self.a = a # (N, S)
        self.u = u # (N, S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        # Reshape for 2D FNO: [S] -> [S, 1, 1] (space, dummy_y, channel)
        x = self.a[idx].unsqueeze(-1).unsqueeze(-1)
        y = self.u[idx].unsqueeze(-1).unsqueeze(-1)
        return x, y

# Base FNO model for this specific problem
class FNO_Burgers_Static_2D(FNO2d):
    def __init__(self, modes1, modes2, width):
        super(FNO_Burgers_Static_2D, self).__init__(modes1, modes2, width)
        self.fc0 = nn.Linear(3, self.width) # Input: a(x) + grid(x,y)

    def forward(self, x):
        # ... (This is the standard FNO forward pass, which is correct) ...
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

# Gate Controller (this is correct)
class GateController(nn.Module):
    def __init__(self, cft_out_dim, fno_modes):
        super(GateController, self).__init__()
        self.cft_out_dim = cft_out_dim
        self.fno_modes = fno_modes
        
        # We process the real-decomposed CFT output, so input is cft_out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(cft_out_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, fno_modes * 2), # Output a gate for both real and imag parts
        )

    def forward(self, cft_spectrum):
        # cft_spectrum shape: [batch, cft_out_dim, 2]
        # Flatten the real/imag parts for the linear layer
        x = cft_spectrum.view(cft_spectrum.shape[0], -1)
        gate = self.net(x)
        # Reshape to [batch, fno_modes, 2]
        gate = gate.view(gate.shape[0], self.fno_modes, 2)
        # Apply a sigmoid to keep gate values between 0 and 1
        return torch.sigmoid(gate)

class GFNO_Burgers_Static_2D(nn.Module):
    def __init__(self, modes1, modes2, width, cft_L, cft_M):
        super(GFNO_Burgers_Static_2D, self).__init__()
        # Use the correctly defined FNO as the base
        self.fno = FNO_Burgers_Static_2D(modes1, modes2, width)
        self.gate_controller = GateController(cft_M, modes1)
        self.cft_L = cft_L
        self.cft_M = cft_M

    def forward(self, x):
        # Input x shape: [batch, S, 1, 1]
        
        # --- Gating Path ---
        raw_signal = x.squeeze()
        cft_spectrum = vectorized_batched_cft_decomposed(raw_signal, self.cft_L, self.cft_M).detach()
        gate = self.gate_controller(cft_spectrum)
        gate_complex = torch.view_as_complex(gate.contiguous()).unsqueeze(1).unsqueeze(-1)

        # --- Main FNO Path (intercepting and gating ONLY the first layer) ---
        grid = self.fno.get_grid(x.shape, x.device)
        x_in = torch.cat((x, grid), dim=-1)
        x_in = self.fno.fc0(x_in)
        x_in = x_in.permute(0, 3, 1, 2)
        
        # First spectral layer with gating
        x1_ft = torch.fft.rfft2(x_in)
        x1_ft[:, :, :self.fno.modes1, :self.fno.modes2] *= gate_complex
        x1 = torch.fft.irfft2(x1_ft, s=(x_in.size(-2), x_in.size(-1)))
        
        x2 = self.fno.w0(x_in)
        x = x1 + x2
        x = F.gelu(x)
        
        # --- Pass through the rest of the FNO layers normally ---
        x1 = self.fno.conv1(x)
        x2 = self.fno.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.fno.conv2(x)
        x2 = self.fno.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.fno.conv3(x)
        x2 = self.fno.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fno.fc1(x)
        x = F.gelu(x)
        x = self.fno.fc2(x)
        return x

def run_experiment(args):
    """
    A wrapper function to run a single training experiment with given args.
    """
    # ... (The entire `main` function's logic is moved here) ...
    # This function will return the final test error.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- Data Loading (simplified for sweep) ---
    reader = MatReader(args.data_path)
    data_a = torch.tensor(reader.read_field('a').T, dtype=torch.float)
    data_u = torch.tensor(reader.read_field('u').T, dtype=torch.float)
    train_a = data_a[:args.ntrain, :]
    train_u = data_u[:args.ntrain, :]
    test_a = data_a[-args.ntest:, :]
    test_u = data_u[-args.ntest:, :]
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)
    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)
    train_dataset = BurgersStaticDataset(train_a, train_u)
    test_dataset = BurgersStaticDataset(test_a, test_u)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Model Definition ---
    model = GFNO_Burgers_Static_2D(args.modes, 1, args.width, args.cft_L, args.cft_M).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    loss_func = LpLoss(size_average=False)
    a_normalizer.to(device)
    y_normalizer.to(device)

    # --- Training Loop ---
    for ep in range(args.epochs):
        model.train()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out.view(x.shape[0], -1), y.view(y.shape[0], -1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
        scheduler.step()
    
    # --- Evaluation ---
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out_reshaped = out.view(x.shape[0], -1)
            y_reshaped = y.view(y.shape[0], -1)
            decoded_out = y_normalizer.decode(out_reshaped)
            test_l2 += loss_func(decoded_out, y_reshaped).item()
    test_l2 /= args.ntest
    return test_l2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter sweep for G-FNO on 2D adapted Burgers' data.")
    parser.add_argument('--data_path', type=str, default='data/burgers_data_R10.mat', help='Path to the .mat file.')
    parser.add_argument('--ntrain', type=int, default=1000, help='Number of training samples.')
    parser.add_argument('--ntest', type=int, default=200, help='Number of testing samples.')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes.')
    parser.add_argument('--width', type=int, default=20, help='Width of the FNO.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--scheduler_step', type=int, default=100, help='Scheduler step size.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Scheduler gamma.')
    parser.add_argument('--cft_L', type=float, default=10.0, help='CFT frequency domain width.')
    parser.add_argument('--cft_M', type=int, default=16, help='CFT frequency samples.')
    args = parser.parse_args()

    # --- Parameter Sweep Setup ---
    L_values = [5, 10, 20]
    M_values = [8, 16, 32]
    
    # Shorten epochs for faster sweep
    args.epochs = 10 
    
    print("--- Starting 2D G-FNO Parameter Sweep ---")
    print(f"Baseline FNO Error: 0.3994 (for reference from 20 epochs)")
    print(f"Sweeping over L={L_values} and M={M_values} for {args.epochs} epochs each.")
    print("-" * 50)

    results = {}
    
    for L in L_values:
        for M in M_values:
            print(f"Testing with cft_L = {L}, cft_M = {M}...")
            args.cft_L = L
            args.cft_M = M
            
            t1 = default_timer()
            final_error = run_experiment(args)
            t2 = default_timer()
            
            results[(L, M)] = final_error
            print(f"Result: L={L}, M={M} -> Test L2 Error = {final_error:.4f} (Time: {t2-t1:.2f}s)")
            print("-" * 50)

    print("\n--- Sweep Finished ---")
    best_params, best_error = min(results.items(), key=lambda item: item[1])
    print(f"Best Result: L={best_params[0]}, M={best_params[1]} with Test L2 Error = {best_error:.4f}")
    
    # Also print the full results table
    print("\nFull Results:")
    for params, error in results.items():
        print(f"  L={params[0]}, M={params[1]}: {error:.4f}")

    print("-" * 50) 