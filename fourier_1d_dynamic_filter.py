"""
@author: Zongyi Li
This file is the Gated Fourier Neural Operator with Dynamic Filter Interpolation,
applied to the 1D Burgers' equation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import os
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

def vectorized_batched_cft_decomposed(x, t, freqs, L, M):
    """
    Computes the CFT of a batch of signals using a fully vectorized method,
    decomposed into real and imaginary parts for `torch.compile` optimization.

    This version avoids complex number operations internally to allow for
    better JIT compilation.
    """
    if x.is_complex():
        raise NotImplementedError("Real-decomposed CFT currently only supports real input signals.")

    batch_size, in_channels, n_samples = x.shape
    n_freqs = len(freqs)
    device = x.device

    # Initialize real and imaginary parts of the final spectrum
    total_spectrum_real = torch.zeros(batch_size, in_channels, n_freqs, device=device, dtype=torch.float32)
    total_spectrum_imag = torch.zeros(batch_size, in_channels, n_freqs, device=device, dtype=torch.float32)

    # Precompute constants for the loop
    segment_len = 1.0 / L
    segment_starts = torch.linspace(0, 1, L + 1, device=device)[:-1]

    k = torch.arange(M, device=device)
    cheb_nodes_ref = -torch.cos((2 * k + 1) * torch.pi / (2 * M))  # on [-1, 1]
    T_k_at_nodes = torch.cos(k.unsqueeze(1) * torch.acos(cheb_nodes_ref.unsqueeze(0)))  # T_k(x_m)

    freq_factor = 2 * torch.pi * (segment_len / 2)
    w_prime = freqs.unsqueeze(0) * freq_factor

    # Decompose the quadrature exponential term: exp(-j*w'*x)
    cheb_nodes_grid, w_prime_grid = torch.meshgrid(cheb_nodes_ref, w_prime.squeeze(0), indexing='ij')
    angle_quad = cheb_nodes_grid * w_prime_grid
    exp_term_real = torch.cos(angle_quad)
    exp_term_imag = -torch.sin(angle_quad)

    # Decompose the quadrature weights (integral of T_k(x) * exp(-j*w'*x))
    cheb_weights_real = torch.einsum("km,mf->kf", T_k_at_nodes, exp_term_real) * (segment_len / 2)
    cheb_weights_imag = torch.einsum("km,mf->kf", T_k_at_nodes, exp_term_imag) * (segment_len / 2)

    for i in range(L):
        a = segment_starts[i]
        t_segment = a + (segment_len / 2) * (cheb_nodes_ref + 1)

        # --- Vectorized Linear Interpolation (on real signal) ---
        right_indices = torch.searchsorted(t, t_segment).clamp(max=n_samples - 1)
        left_indices = (right_indices - 1).clamp(min=0)

        t_left = t[left_indices]
        t_right = t[right_indices]

        denom = t_right - t_left
        denom[denom == 0] = 1.0

        w_right = (t_segment - t_left) / denom
        w_left = 1.0 - w_right

        signal_left = x[:, :, left_indices]
        signal_right = x[:, :, right_indices]

        signal_segments = w_left * signal_left + w_right * signal_right

        # --- Apply decomposed quadrature weights to real signal ---
        spectrum_segment_real = torch.einsum("bcm,mf->bcf", signal_segments, cheb_weights_real)
        spectrum_segment_imag = torch.einsum("bcm,mf->bcf", signal_segments, cheb_weights_imag)

        # --- Decompose phase shift and add to total ---
        # exp(-2j*pi*f*a)
        angle_shift = 2 * torch.pi * freqs * a
        exp_shift_real = torch.cos(angle_shift)
        exp_shift_imag = -torch.sin(angle_shift)

        # (A+iB)*(C+iD) = (AC-BD) + i(AD+BC)
        # A=spectrum_segment_real, B=spectrum_segment_imag
        # C=exp_shift_real, D=exp_shift_imag
        total_spectrum_real += (spectrum_segment_real * exp_shift_real) - (spectrum_segment_imag * exp_shift_imag)
        total_spectrum_imag += (spectrum_segment_real * exp_shift_imag) + (spectrum_segment_imag * exp_shift_real)

    return torch.complex(total_spectrum_real, total_spectrum_imag)

################################################################
#  1d fourier layer with dynamic filter interpolation
################################################################
class SpectralConv1d_Interpolate(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d_Interpolate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        # Two sets of expert weights
        self.weights1_A = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights1_B = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d_dynamic(self, input, weights):
        # (batch, in_channel, x), (batch, in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,bioy->box", input, weights)

    def forward(self, x, gate):
        batchsize = x.shape[0]

        # Interpolate the filters using the gate
        # Gate shape: [B, W], we need [B, 1, W, 1] for broadcasting
        gate_reshaped = gate.view(batchsize, 1, self.out_channels, 1)
        weights1 = gate_reshaped * self.weights1_A + (1 - gate_reshaped) * self.weights1_B
        
        # Compute FFT
        x_ft = torch.fft.rfft(x)
        
        # Apply the dynamic filters
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d_dynamic(x_ft[:, :, :self.modes1], weights1)

        # Compute Inverse FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class DynamicFNO1d(nn.Module):
    def __init__(self, modes, width):
        super(DynamicFNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(2, self.width)

        # Gating networks (one for each layer)
        # Using a simple linear layer on CFT magnitudes for gating
        self.gate_net0 = nn.Linear(self.modes1, self.width)
        self.gate_net1 = nn.Linear(self.modes1, self.width)
        self.gate_net2 = nn.Linear(self.modes1, self.width)
        self.gate_net3 = nn.Linear(self.modes1, self.width)

        # Dynamic Spectral Conv layers
        self.conv0 = SpectralConv1d_Interpolate(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d_Interpolate(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d_Interpolate(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d_Interpolate(self.width, self.width, self.modes1)
        
        # Residual connections
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def get_gate_signal(self, x, gate_net):
        n_samples = x.shape[2]
        t_coords = torch.linspace(0, 1, n_samples, device=x.device)
        f_points = torch.fft.rfftfreq(n_samples, d=1/n_samples).to(x.device)
        
        # Use a detached CFT calculation as the basis for the gate
        x_cft = vectorized_batched_cft_decomposed(x=x, t=t_coords, freqs=f_points, L=20, M=8).detach()
        
        # Get magnitude and average over input channels
        cft_mag = torch.abs(x_cft[:, :, :self.modes1])
        cft_mag_avg = torch.mean(cft_mag, dim=1) # Shape: (batch, modes1)
        
        # Pass through a simple linear layer and sigmoid
        gate = torch.sigmoid(gate_net(cft_mag_avg)) # Shape: (batch, width)
        return gate

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Layer 0
        gate0 = self.get_gate_signal(x, self.gate_net0)
        x1 = self.conv0(x, gate0)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 1
        gate1 = self.get_gate_signal(x, self.gate_net1)
        x1 = self.conv1(x, gate1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 2
        gate2 = self.get_gate_signal(x, self.gate_net2)
        x1 = self.conv2(x, gate2)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 3
        gate3 = self.get_gate_signal(x, self.gate_net3)
        x1 = self.conv3(x, gate3)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 100

sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

batch_size = 20
learning_rate = 0.001

epochs = 20
step_size = 50
gamma = 0.5

modes = 16
width = 64

# --- Main Execution Block --- 
if __name__ == "__main__":
    ################################################################
    # read data
    ################################################################
    
    # Data is of the shape (number of samples, grid size)
    print("Loading data...") # Add print statement
    # Ensure the path to the .mat file is correct for your system
    try:
        dataloader = MatReader('/Users/liutaiqian/Downloads/fourier_neural_operator-master/burgers_data_R10.mat')
        x_data = dataloader.read_field('a')[:,::sub]
        y_data = dataloader.read_field('u')[:,::sub]
    except FileNotFoundError:
        print("Error: burgers_data_R10.mat not found.")
        print("Please ensure the path in fourier_1d.py is correct.")
        exit() # Exit if data not found when running directly
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        exit()
    print("Data loaded.")

    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    
    x_train = x_train.reshape(ntrain,s,1)
    x_test = x_test.reshape(ntest,s,1)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # model
    print("Initializing model...")
    model = DynamicFNO1d(modes, width).to(device)
    print(f"Model parameters: {count_params(model)}")
    
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    myloss = LpLoss(size_average=False)

    print(f"Starting training for {epochs} epochs...")
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            # Assume data needs to be on the same device as model
            # x, y = x.to(device), y.to(device) # Add this if using GPU
    
            optimizer.zero_grad()
            out = model(x)
    
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss
    
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
    
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                # x, y = x.to(device), y.to(device) # Add this if using GPU
    
                out = model(x)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    
        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest
    
        t2 = default_timer()
        print(f"Epoch: {ep}, Time: {t2-t1:.4f}, Train MSE: {train_mse:.6f}, Train L2: {train_l2:.6f}, Test L2: {test_l2:.6f}")
    
    print("Training finished.")

    # Prediction and Saving
    print("Generating predictions...")
    # torch.save(model, 'model/ns_fourier_burgers') # Saving model state is often better
    pred = torch.zeros(y_test.shape)
    index = 0
    test_loader_pred = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader_pred:
            # x, y = x.to(device), y.to(device)
            test_l2_sample = 0
            out = model(x).view(-1)
            pred[index] = out
    
            test_l2_sample = myloss(out.view(1, -1), y.view(1, -1)).item()
            # print(f"Sample {index}, Test L2: {test_l2_sample:.6f}") # Optional: print per sample loss
            index = index + 1
    
    # Ensure the pred directory exists
    pred_dir = 'pred'
    os.makedirs(pred_dir, exist_ok=True)
    save_path = os.path.join(pred_dir, 'burger_test.mat')
    try:
        import scipy.io
        scipy.io.savemat(save_path, mdict={'pred': pred.cpu().numpy()})
        print(f"Predictions saved to {save_path}")
    except ImportError:
        print("Warning: scipy.io not found. Cannot save predictions to .mat file.")
    except Exception as e:
        print(f"An error occurred while saving predictions: {e}")

    # Plotting the error for the first test sample
    print("Plotting prediction error for the first test sample...")
    s = x_test.shape[1] # Get the spatial resolution
    x_axis = np.linspace(0, 1, s)
    
    # Select the first sample and convert to numpy
    pred_sample_0 = pred[0].cpu().numpy()
    y_test_sample_0 = y_test[0].cpu().numpy()
    error_sample_0 = np.abs(pred_sample_0 - y_test_sample_0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, y_test_sample_0, label='Ground Truth')
    plt.plot(x_axis, pred_sample_0, '--', label='Prediction')
    plt.plot(x_axis, error_sample_0, ':', label='Absolute Error')
    plt.xlabel("Spatial Coordinate (x)")
    plt.ylabel("Value u(x)")
    plt.title("FNO Prediction vs Ground Truth (Test Sample 0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_save_path = 'fourier_1d_prediction_error.png'
    try:
        plt.savefig(plot_save_path)
        print(f"Error plot saved to {plot_save_path}")
    except Exception as e:
        print(f"An error occurred while saving the plot: {e}")
    plt.close() # Close the plot figure
