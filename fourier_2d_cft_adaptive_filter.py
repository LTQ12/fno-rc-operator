import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

from utilities3 import *
from chebyshev import vectorized_batched_cft

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)
np.random.seed(0)

################################################################
# 2d cft (Re-used)
################################################################
def cft2d(x, modes1, modes2, L_segments=10, M_cheb=10):
    B, C, H, W = x.shape
    device = x.device
    
    t_coords_w = torch.linspace(0, 1, W, device=device, dtype=x.dtype)
    f_points_w = torch.fft.rfftfreq(W, d=1.0/W)[:modes2].to(device)
    x_reshaped_w = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
    cft_w_complex = vectorized_batched_cft(x_reshaped_w, t_coords_w, f_points_w, L_segments, M_cheb)
    cft_w_complex = cft_w_complex.view(B, H, C, modes2).permute(0, 2, 1, 3)

    t_coords_h = torch.linspace(0, 1, H, device=device, dtype=x.dtype)
    f_points_h = torch.fft.fftfreq(H, d=1.0/H).to(device)
    h_indices = torch.cat((torch.arange(0, modes1//2), torch.arange(H-(modes1-modes1//2), H))).to(device)
    f_points_h_selected = f_points_h[h_indices]
    x_reshaped_h = cft_w_complex.permute(0, 3, 1, 2).reshape(B * modes2, C, H)

    cft_h_real = vectorized_batched_cft(x_reshaped_h.real, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_h_imag = vectorized_batched_cft(x_reshaped_h.imag, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_hw_complex = cft_h_real + 1j * cft_h_imag
    
    cft_hw_complex = cft_hw_complex.view(B, modes2, C, modes1).permute(0, 2, 3, 1)
    return cft_hw_complex

################################################################
# The Final Architecture: CFT-Adaptive Filter
################################################################
class SpectralConv2d_Adaptive(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, L_segments=4, M_cheb=8):
        super(SpectralConv2d_Adaptive, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Shared, learnable base filter (the core of FNO)
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        # Define CFT parameters before they are used
        self.cft_modes1 = modes1 // 4
        self.cft_modes2 = modes2 // 4
        self.L_segments = L_segments
        self.M_cheb = M_cheb
        cft_flat_dim = self.in_channels * self.cft_modes1 * self.cft_modes2 * 2

        # Lightweight generator for the adaptive "delta" filter components with Dropout
        self.delta_generator_i = nn.Sequential(nn.Linear(cft_flat_dim, self.in_channels * 2), nn.Dropout(0.5), nn.Tanh())
        self.delta_generator_o = nn.Sequential(nn.Linear(cft_flat_dim, self.out_channels * 2), nn.Dropout(0.5), nn.Tanh())
        self.delta_generator_m = nn.Sequential(nn.Linear(cft_flat_dim, self.modes1 * self.modes2 * 2), nn.Dropout(0.5), nn.Tanh())
        
        # Learnable scalers for stability
        self.scaler_i = nn.Parameter(torch.zeros(1))
        self.scaler_o = nn.Parameter(torch.zeros(1))
        self.scaler_m = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Generate the adaptive "delta" filter components from CFT coefficients
        cft_coeffs = cft2d(x, self.cft_modes1, self.cft_modes2, self.L_segments, self.M_cheb)
        cft_flat = torch.view_as_real(cft_coeffs).flatten(1)
        
        # Generate low-rank components
        delta_i = self.scaler_i * torch.view_as_complex(self.delta_generator_i(cft_flat).view(B, self.in_channels, 2))
        delta_o = self.scaler_o * torch.view_as_complex(self.delta_generator_o(cft_flat).view(B, self.out_channels, 2))
        delta_m = self.scaler_m * torch.view_as_complex(self.delta_generator_m(cft_flat).view(B, self.modes1, self.modes2, 2))

        # Synthesize the full delta_weights tensor via outer product (einsum)
        # This is extremely memory efficient as the large tensor is never stored.
        delta_weights = torch.einsum('bi,bo,bxy->bioxy', delta_i, delta_o, delta_m)

        # 2. Combine shared and adaptive filters
        # Use expand to match batch dimension without copying memory
        effective_weights1 = self.weights1.unsqueeze(0).expand(B, -1, -1, -1, -1) + delta_weights
        effective_weights2 = self.weights2.unsqueeze(0).expand(B, -1, -1, -1, -1) + delta_weights

        # 3. Apply the effective dynamic filter
        x_ft = torch.fft.rfft2(x, s=(H, W))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,bioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], effective_weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,bioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], effective_weights2)

        # 4. Return to physical space
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x

class FNO_CFT_AdaptiveFilter(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels):
        super(FNO_CFT_AdaptiveFilter, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9
        self.fc0 = nn.Linear(in_channels + 2, self.width)

        self.conv0 = SpectralConv2d_Adaptive(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_Adaptive(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_Adaptive(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_Adaptive(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

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

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device) 