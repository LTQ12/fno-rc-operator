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
# 3D CFT - 修复版本
################################################################
def cft3d_fixed(x, modes1, modes2, modes3, L_segments=8, M_cheb=8):
    """修复的3D CFT实现 - 确保输出维度正确"""
    B, C, H, W, T = x.shape
    device = x.device

    # 1. FFT along temporal dimension (T)
    x_ft = torch.fft.rfftn(x, dim=(-1,))
    
    # Select modes for temporal dimension
    x_ft_filtered = torch.zeros(B, C, H, W, modes3, dtype=torch.cfloat, device=device)
    x_ft_filtered[..., :modes3] = x_ft[..., :modes3]

    # --- Now apply 2D CFT to the (B, C*modes3, H, W) tensor ---
    x_reshaped = x_ft_filtered.permute(0, 1, 4, 2, 3).reshape(B, C * modes3, H, W)

    # 2. CFT along width dimension (W)
    t_coords_w = torch.linspace(0, 1, W, device=device, dtype=x.dtype)
    f_points_w = torch.fft.rfftfreq(W, d=1.0/W)[:modes2].to(device)
    cft_w_input = x_reshaped.permute(0, 2, 1, 3).reshape(B * H, C * modes3, W)
    
    cft_w_real = vectorized_batched_cft(cft_w_input.real, t_coords_w, f_points_w, L_segments, M_cheb)
    cft_w_imag = vectorized_batched_cft(cft_w_input.imag, t_coords_w, f_points_w, L_segments, M_cheb)
    cft_w_complex = cft_w_real + 1j * cft_w_imag
    cft_w_complex = cft_w_complex.view(B, H, C * modes3, modes2).permute(0, 2, 1, 3)

    # 3. CFT along height dimension (H)
    t_coords_h = torch.linspace(0, 1, H, device=device, dtype=x.dtype)
    f_points_h = torch.fft.fftfreq(H, d=1.0/H).to(device)
    h_indices = torch.cat((torch.arange(0, modes1//2), torch.arange(H-(modes1-modes1//2), H))).to(device)
    f_points_h_selected = f_points_h[h_indices]
    cft_h_input = cft_w_complex.permute(0, 3, 1, 2).reshape(B * modes2, C * modes3, H)

    cft_h_real = vectorized_batched_cft(cft_h_input.real, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_h_imag = vectorized_batched_cft(cft_h_input.imag, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_hw_complex = cft_h_real + 1j * cft_h_imag
    
    # Reshape back to (B, C, modes1, modes2, modes3)
    cft_hwt_complex = cft_hw_complex.view(B, modes2, C, modes3, modes1).permute(0, 2, 4, 1, 3)
    return cft_hwt_complex

################################################################
# 完全重新设计的3D FNO-RC架构
################################################################
class SpectralConv3d_RC_Fixed(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_RC_Fixed, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        # 主FNO路径的权重
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

        # 🔧 关键修复1: CFT使用更多模态以捕获更丰富的信息
        self.cft_modes1 = max(2, modes1 // 2)  # 4 而不是 2
        self.cft_modes2 = max(2, modes2 // 2)  # 4 而不是 2
        self.cft_modes3 = max(2, modes3 // 2)  # 4 而不是 2
        
        # 🔧 关键修复2: CFT残差修正网络 - 输出空间分辨率的修正
        cft_feature_dim = self.in_channels * self.cft_modes1 * self.cft_modes2 * self.cft_modes3 * 2
        
        # 使用卷积网络而不是全连接网络，保持空间结构
        self.cft_projection = nn.Sequential(
            nn.Linear(cft_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        
        # 空间感知的残差生成器
        self.residual_conv = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, self.out_channels, kernel_size=1)
        )
        
        # 🔧 关键修复3: 残差权重 - 学习如何组合主路径和CFT路径
        self.residual_weight = nn.Parameter(torch.zeros(1))
        
        # 零初始化确保训练稳定性
        nn.init.zeros_(self.residual_conv[-1].weight)
        nn.init.zeros_(self.residual_conv[-1].bias)

    def forward(self, x):
        B, C, H, W, T = x.shape

        # 🔧 主FNO路径 - 保持不变
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, H, W, T//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x_fno = torch.fft.irfftn(out_ft, s=(H, W, T))

        # 🔧 CFT残差修正路径 - 完全重新设计
        try:
            # 计算CFT特征
            cft_coeffs = cft3d_fixed(x, self.cft_modes1, self.cft_modes2, self.cft_modes3)
            cft_flat = torch.view_as_real(cft_coeffs).flatten(1)  # [B, feature_dim]
            
            # 投影到中间特征
            cft_features = self.cft_projection(cft_flat)  # [B, 64]
            
            # 重塑为3D张量以便卷积处理
            # 这里我们需要将1D特征重塑为3D，可以通过重复或学习的方式
            cft_3d = cft_features.view(B, 64, 1, 1, 1).repeat(1, 1, H, W, T)
            
            # 通过3D卷积生成空间感知的残差
            residual = self.residual_conv(cft_3d)  # [B, out_channels, H, W, T]
            
            # 🔧 关键修复4: 学习的残差权重组合
            return x_fno + self.residual_weight * residual
            
        except Exception as e:
            print(f"CFT路径失败，使用主路径: {e}")
            return x_fno

################################################################
# 修复的FNO_RC_3D模型
################################################################
class FNO_RC_3D_Fixed(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels):
        super(FNO_RC_3D_Fixed, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = 6 
        self.fc0 = nn.Linear(self.in_channels + 3, self.width)

        # 使用修复的频谱卷积层
        self.conv0 = SpectralConv3d_RC_Fixed(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_RC_Fixed(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_RC_Fixed(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_RC_Fixed(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, T_in+3, H, W, T_out] -> [B, C, H, W, T]
        x = F.pad(x, [0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.bn3(x1 + x2)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        
        # 强制处理5维输入 [B, H, W, T_out, T_in]，生成5维网格 [B, H, W, T_out, 3]
        if len(shape) >= 5:
            size_t = shape[3]  # T_out
            # 生成网格坐标
            gridx = torch.linspace(0, 1, size_x, device=device, dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(batchsize, 1, size_y, size_t, 1)
            
            gridy = torch.linspace(0, 1, size_y, device=device, dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(batchsize, size_x, 1, size_t, 1)
            
            gridz = torch.zeros(batchsize, size_x, size_y, size_t, 1, device=device, dtype=torch.float)
            
            return torch.cat((gridx, gridy, gridz), dim=-1)  # [B, H, W, T_out, 3]
        else:
            # 对于4维输入 [B, H, W, ?]，生成4维网格 [B, H, W, 3]
            gridx = torch.linspace(0, 1, size_x, device=device, dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
            
            gridy = torch.linspace(0, 1, size_y, device=device, dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
            
            gridz = torch.zeros(batchsize, size_x, size_y, 1, device=device, dtype=torch.float)
            
            return torch.cat((gridx, gridy, gridz), dim=-1)  # [B, H, W, 3]
