"""
Optimized FNO-1D with High-Performance Vectorized CFT/ICFT
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

# Import the NEW high-performance vectorized operators
from fourier_ops_vectorized import vectorized_batched_cft, vectorized_batched_icft

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
#  Optimized 1D Fourier layer with VECTORIZED CFT/ICFT
################################################################
class SpectralConv1dOptimized(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, cft_L_segments=100, cft_M_cheb=20):
        super(SpectralConv1dOptimized, self).__init__()
        """
        High-performance 1D Fourier layer using vectorized CFT/ICFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

        # Store parameters for the vectorized operators
        self.cft_L_segments = cft_L_segments
        self.cft_M_cheb = cft_M_cheb

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        n_samples = x.shape[2]
        device = x.device

        # Prepare coordinates and frequencies as tensors
        t_coords = torch.linspace(0, 1, n_samples, device=device)
        f_points = torch.fft.fftfreq(n_samples, d=1/n_samples).to(device)

        # --- 1. Vectorized Forward CFT ---
        x_cft = vectorized_batched_cft(
            signals=x,
            t_coords=t_coords,
            f_points=f_points,
            L_segments=self.cft_L_segments,
            M_cheb=self.cft_M_cheb
        )

        # --- 2. Multiply relevant Fourier modes ---
        out_ft = torch.zeros(batchsize, self.out_channels, n_samples, device=device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_cft[:, :, :self.modes1], self.weights1)

        # --- 3. Vectorized Inverse ICFT ---
        x_final = vectorized_batched_icft(
            coeffs=out_ft,
            f_points=f_points,
            t_coords=t_coords,
            L_segments=self.cft_L_segments,
            M_cheb=self.cft_M_cheb
        )
        
        return x_final

class FNO1dOptimized(nn.Module):
    def __init__(self, modes, width, cft_L_segments=100, cft_M_cheb=20, **kwargs):
        super(FNO1dOptimized, self).__init__()
        """
        FNO-1D using the high-performance vectorized CFT/ICFT layer.
        """
        self.modes1 = modes
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(2, self.width)

        # Create spectral convolution layers with the vectorized op parameters
        conv_params = {'cft_L_segments': cft_L_segments, 'cft_M_cheb': cft_M_cheb}
        self.conv0 = SpectralConv1dOptimized(self.width, self.width, self.modes1, **conv_params)
        self.conv1 = SpectralConv1dOptimized(self.width, self.width, self.modes1, **conv_params)
        self.conv2 = SpectralConv1dOptimized(self.width, self.width, self.modes1, **conv_params)
        self.conv3 = SpectralConv1dOptimized(self.width, self.width, self.modes1, **conv_params)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch_size, n_samples, 2) - first channel is signal, second is spatial coordinate
        # Add grid coordinates
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # Now shape: (batch_size, n_samples, 3)
        x = self.fc0(x)  # Linear(2, width) - maps 2 channels to width
        x = x.permute(0, 2, 1)  # Shape: (batch_size, width, n_samples)

        # Spectral convolution layers
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

        x = x.permute(0, 2, 1)  # Shape: (batch_size, n_samples, width)
        x = self.fc1(x)  # Linear(width, 128)
        x = F.gelu(x)
        x = self.fc2(x)  # Linear(128, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  Configuration and training setup
################################################################
def create_optimized_fno_config():
    """Create optimized FNO configuration"""
    return {
        'ntrain': 1000,
        'ntest': 100,
        'sub': 2**3,
        'h': 2**13 // 2**3,
        's': 2**13 // 2**3,
        'batch_size': 20,
        'learning_rate': 0.001,
        'epochs': 20,
        'step_size': 50,
        'gamma': 0.5,
        'modes': 16,
        'width': 64,
        # CFT/ICFT parameters
        'cft_L_segments': 100,
        'cft_M_cheb': 20,
    }

def train_optimized_fno():
    """Training function for optimized FNO"""
    config = create_optimized_fno_config()
    
    # Create model
    model = FNO1dOptimized(
        modes=config['modes'],
        width=config['width'],
        cft_L_segments=config['cft_L_segments'],
        cft_M_cheb=config['cft_M_cheb'],
    )
    
    print("Optimized FNO-1D model created with vectorized CFT/ICFT")
    print(f"CFT parameters: L_segments={config['cft_L_segments']}, M_cheb={config['cft_M_cheb']}")
    
    return model, config

if __name__ == "__main__":
    # Test the optimized model
    model, config = train_optimized_fno()
    
    # Create dummy input for testing
    batch_size = 4
    n_samples = 1024
    x = torch.randn(batch_size, n_samples, 2)
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!") 