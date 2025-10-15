import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

from utilities3 import *
from chebyshev import vectorized_batched_cft
from fourier_2d_cft_residual import cft2d

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)
np.random.seed(0)

################################################################
# 3d cft (Hybrid: CFT for spatial, FFT for temporal can be an option)
# For now, full CFT on all three dimensions
################################################################
def cft3d(x, modes1, modes2, modes3, L_segments=6, M_cheb=6):
    """
    Computes the 3D Continuous Fourier Transform of a batch of signals.
    Applies CFT sequentially along the last three dimensions (H, W, D).
    """
    B, C, H, W, D = x.shape
    device = x.device

    # 1. CFT along dimension D (depth)
    t_coords_d = torch.linspace(0, 1, D, device=device, dtype=x.dtype)
    f_points_d = torch.fft.rfftfreq(D, d=1.0/D)[:modes3].to(device)
    # Reshape for CFT: (B*H*W, C, D)
    x_reshaped_d = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, C, D)
    
    # vectorized_batched_cft expects real input, so we process real and imag parts separately if input is complex
    if x.is_complex():
        cft_d_real = vectorized_batched_cft(x_reshaped_d.real, t_coords_d, f_points_d, L_segments, M_cheb)
        cft_d_imag = vectorized_batched_cft(x_reshaped_d.imag, t_coords_d, f_points_d, L_segments, M_cheb)
        cft_d_complex = cft_d_real + 1j * cft_d_imag
    else:
        cft_d_complex = vectorized_batched_cft(x_reshaped_d, t_coords_d, f_points_d, L_segments, M_cheb)
    
    # Reshape back: (B, H, W, C, modes3) -> (B, C, H, W, modes3)
    cft_d_complex = cft_d_complex.view(B, H, W, C, modes3).permute(0, 3, 1, 2, 4)

    # 2. CFT along dimension W (width)
    t_coords_w = torch.linspace(0, 1, W, device=device, dtype=x.dtype)
    f_points_w = torch.fft.rfftfreq(W, d=1.0/W)[:modes2].to(device)
    # Reshape for CFT: (B*H*modes3, C, W)
    x_reshaped_w = cft_d_complex.permute(0, 2, 4, 1, 3).reshape(B * H * modes3, C, W)

    cft_w_real = vectorized_batched_cft(x_reshaped_w.real, t_coords_w, f_points_w, L_segments, M_cheb)
    cft_w_imag = vectorized_batched_cft(x_reshaped_w.imag, t_coords_w, f_points_w, L_segments, M_cheb)
    cft_w_complex = cft_w_real + 1j * cft_w_imag
    
    # Reshape back: (B, H, modes3, C, modes2) -> (B, C, H, modes3, modes2)
    cft_w_complex = cft_w_complex.view(B, H, modes3, C, modes2).permute(0, 3, 1, 2, 4)

    # 3. CFT along dimension H (height)
    t_coords_h = torch.linspace(0, 1, H, device=device, dtype=x.dtype)
    # For a real-to-complex transform, the first dimension is not symmetric.
    # For complex-to-complex, it is. We choose the complex-to-complex FFT frequencies.
    f_points_h = torch.fft.fftfreq(H, d=1.0/H).to(device)
    h_indices = torch.cat((torch.arange(0, modes1//2), torch.arange(H-(modes1-modes1//2), H))).to(device)
    f_points_h_selected = f_points_h[h_indices]
    
    # Reshape for CFT: (B*modes3*modes2, C, H)
    x_reshaped_h = cft_w_complex.permute(0, 3, 4, 1, 2).reshape(B * modes3 * modes2, C, H)

    cft_h_real = vectorized_batched_cft(x_reshaped_h.real, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_h_imag = vectorized_batched_cft(x_reshaped_h.imag, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_hwd_complex = cft_h_real + 1j * cft_h_imag
    
    # Reshape back to final form: (B, C, modes1, modes3, modes2) -> (B, C, modes1, modes2, modes3)
    cft_hwd_complex = cft_hwd_complex.view(B, modes3, modes2, C, modes1).permute(0, 3, 4, 2, 1)

    return cft_hwd_complex

################################################################
# 3D FNO with CFT-based Residual Correction Layer
################################################################
class SpectralConv3d_RC(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, L_segments=6, M_cheb=6):
        super(SpectralConv3d_RC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        # FNO learnable weights
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        
        # CFT residual correction path
        self.cft_modes1 = modes1 // 4
        self.cft_modes2 = modes2 // 4
        self.cft_modes3 = modes3 // 4
        self.L_segments = L_segments
        self.M_cheb = M_cheb
        # 使用仅空间CFT（H/W），为每个时间帧生成校正 → (B,out,D)
        cft_flat_dim_spatial = self.in_channels * self.cft_modes1 * self.cft_modes2 * 2
        self.correction_generator_time = nn.Sequential(
            nn.Linear(cft_flat_dim_spatial, (self.in_channels + self.out_channels) * 2),
            nn.GELU(),
            nn.Linear((self.in_channels + self.out_channels) * 2, self.out_channels)
        )
        # Initialize last layer to zeros for stability
        nn.init.zeros_(self.correction_generator_time[-1].weight)
        nn.init.zeros_(self.correction_generator_time[-1].bias)

        # Optional: enable/disable correction path to save compute on deeper layers
        self.enable_correction = True
        # Learnable scaling to stabilize correction magnitude
        self.correction_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        B, C, H, W, D = x.shape

        # --- Main FNO Path ---
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, H, W, D//2 + 1, dtype=torch.cfloat, device=x.device)

        # Complex multiplication in Fourier space
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x_fno = torch.fft.irfftn(out_ft, s=(H, W, D))

        # --- CFT Residual Correction Path (optional) ---
        self.last_correction = None
        if self.enable_correction:
            # 仅对空间(H/W)做CFT，并逐时间帧生成校正
            # x: (B,C,H,W,D) → (B*D,C,H,W)
            x_slices = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
            cft_hw = cft2d(x_slices, self.cft_modes1, self.cft_modes2, self.L_segments, self.M_cheb) # (B*D,C,m1,m2)
            cft_flat = torch.view_as_real(cft_hw).reshape(B * D, -1)
            corr_time = self.correction_generator_time(cft_flat) # (B*D, out)
            corr_time = corr_time.view(B, D, self.out_channels).permute(0, 2, 1) # (B,out,D)
            correction = corr_time.view(B, self.out_channels, 1, 1, D)
            self.last_correction = correction
            return x_fno + self.correction_scale * correction
        else:
            return x_fno

################################################################
# 3D FNO-RC Model
################################################################
class FNO_RC_3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=32, in_channels=4, out_channels=1,
                 num_correction_layers=2, cft_L=6, cft_M=6, correction_scale_init=0.1): # configurable RC
        super(FNO_RC_3D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # Padding for non-periodic boundaries
        self.num_correction_layers = num_correction_layers

        # Input: (u(x,y,t), x, y, t) -> 4 channels
        self.fc0 = nn.Linear(in_channels + 3, self.width)

        self.conv0 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3, L_segments=cft_L, M_cheb=cft_M)
        self.conv1 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3, L_segments=cft_L, M_cheb=cft_M)
        self.conv2 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3, L_segments=cft_L, M_cheb=cft_M)
        self.conv3 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3, L_segments=cft_L, M_cheb=cft_M)

        # 启用前 num_correction_layers 个 RC，其余关闭，并设置初始缩放
        layers = [self.conv0, self.conv1, self.conv2, self.conv3]
        for idx, layer in enumerate(layers):
            layer.enable_correction = (idx < self.num_correction_layers)
            with torch.no_grad():
                layer.correction_scale.data = torch.tensor(correction_scale_init, dtype=torch.float32)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # Append coordinates

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3) # (B, C, H, W, D)
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding]) # Pad all 3 spatial dimensions

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

        x = x[..., :-self.padding, :-self.padding, :-self.padding] # Remove padding
        x = x.permute(0, 2, 3, 4, 1) # (B, H, W, D, C) where D=T_in, C=width

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x 