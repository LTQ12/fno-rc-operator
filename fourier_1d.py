"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from cft_1d import *
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
from cft_utils import compute_lagrange_coeffs, precompute_factorial_matrix, generate_intervals_and_points
from cft_transform import replace_fft_with_cft, replace_ifft_with_icft
import os # Import os module

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

        # Store CFT/ICFT params (using defaults from cft_transform)
        self.cft_n_intervals = 80  # Increased from default
        self.cft_interp_order = 25  # Increased from default

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        in_channels = x.shape[1]
        n_samples = x.shape[2]
        device = x.device

        # --- Forward CFT --- 
        # Prepare time coordinates (assuming domain [0, 1]) and frequency points once
        t_coords = np.linspace(0, 1, n_samples, endpoint=False) # Use endpoint=False like in test
        sampling_rate = n_samples / 1.0 # Assuming T=1.0
        f_points = np.fft.fftfreq(n_samples, d=1/sampling_rate)
        # We need the unshifted frequencies for ICFT later
        # f_points_unshifted = np.fft.fftfreq(n_samples, d=1/sampling_rate) 
        # Actually, replace_ifft_with_icft expects the shifted freqs
        f_points_shifted = np.fft.fftshift(f_points)

        x_cft_list = []
        for i in range(batchsize):
            sample_cft_list = []
            for j in range(in_channels):
                signal_np = x[i, j, :].detach().cpu().numpy()
                # Call CFT (output is already shifted from the function)
                cft_result_np = replace_fft_with_cft(signal_np, t_coords, 
                                                     n_intervals=self.cft_n_intervals, 
                                                     interp_order=self.cft_interp_order)
                sample_cft_list.append(torch.from_numpy(cft_result_np).to(torch.cfloat))
            x_cft_list.append(torch.stack(sample_cft_list))
        x_cft = torch.stack(x_cft_list).to(device) # Shape: (batch, in_channels, n_samples)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, n_samples, device=device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_cft[:, :, :self.modes1], self.weights1)

        # --- Inverse ICFT --- 
        x_icft_list = []
        # Convert frequency points to NumPy once
        f_points_shifted_np = f_points_shifted
        t_coords_np = t_coords

        for i in range(batchsize):
            sample_icft_list = []
            for j in range(self.out_channels): # Loop over output channels
                coeffs_np = out_ft[i, j, :].detach().cpu().numpy()
                # Call ICFT
                icft_result_np = replace_ifft_with_icft(coeffs_np, f_points_shifted_np, t_coords_np, t_coords_np,
                                                        n_intervals=self.cft_n_intervals, 
                                                        interp_order=self.cft_interp_order)
                sample_icft_list.append(torch.from_numpy(icft_result_np).to(torch.cfloat))
            x_icft_list.append(torch.stack(sample_icft_list))
        
        x_final = torch.stack(x_icft_list).to(device) # Shape: (batch, out_channels, n_samples)

        return x_final.real # Return only the real part

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

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

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
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
    model = FNO1d(modes, width)
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
