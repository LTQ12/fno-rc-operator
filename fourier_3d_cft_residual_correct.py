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
# 3D CFT - 简化版本（使用FFT近似）
################################################################
def cft3d_simple(x, modes1, modes2, modes3):
    """简化的3D CFT，使用FFT近似以提高速度"""
    B, C, H, W, T = x.shape
    device = x.device
    
    # 使用FFT快速提取频域特征，近似CFT效果
    x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
    
    # 只取低频部分，模拟CFT的连续性捕获
    h_modes = min(modes1, H//2)
    w_modes = min(modes2, W//2) 
    t_modes = min(modes3, T//2 + 1)
    
    # 提取对应的频域特征
    features = x_ft[:, :, :h_modes, :w_modes, :t_modes]
    
    return features

################################################################
# 正确的3D FNO-RC架构 - 基于fourier_3d_clean.py + 2D CFT逻辑
################################################################
class SpectralConv3d_RC_Correct(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_RC_Correct, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        # 🔧 主FNO路径 - 完全复制fourier_3d_clean.py的简单结构
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

        # 🔧 CFT残差修正路径 - 完全复制2D成功逻辑
        self.cft_modes1 = modes1 // 4  # 复制2D的//4比例
        self.cft_modes2 = modes2 // 4
        self.cft_modes3 = modes3 // 4
        
        cft_flat_dim = self.in_channels * self.cft_modes1 * self.cft_modes2 * self.cft_modes3 * 2
        
        # 完全复制2D的correction_generator结构
        self.correction_generator = nn.Sequential(
            nn.Linear(cft_flat_dim, (self.in_channels + self.out_channels) * 2),
            nn.GELU(),
            nn.Linear((self.in_channels + self.out_channels) * 2, self.out_channels)
        )
        
        # 完全复制2D的零初始化策略
        nn.init.zeros_(self.correction_generator[-1].weight)
        nn.init.zeros_(self.correction_generator[-1].bias)

    def forward(self, x):
        B, C, H, W, T = x.shape

        # 🔧 主FNO路径 - 完全复制fourier_3d_clean.py的逻辑
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, H, W, T//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # 只有一个权重矩阵，不是4个！
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights)
        
        x_fno = torch.fft.irfftn(out_ft, s=(H, W, T))

        # 🔧 CFT残差修正路径 - 完全复制2D成功逻辑
        try:
            cft_coeffs = cft3d_simple(x, self.cft_modes1, self.cft_modes2, self.cft_modes3)
            cft_flat = torch.view_as_real(cft_coeffs).flatten(1)
            correction = self.correction_generator(cft_flat)  # (B, out_channels)
            
            # 完全复制2D的广播方式，扩展到3D
            correction = correction.view(B, self.out_channels, 1, 1, 1)
            
            return x_fno + correction
            
        except Exception as e:
            print(f"CFT路径失败: {e}")
            return x_fno

################################################################
# 正确的3D FNO-RC模型 - 基于fourier_3d_clean.py结构
################################################################
class FNO_RC_3D_Correct(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels):
        super(FNO_RC_3D_Correct, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = 8  # 复制fourier_3d_clean.py的padding
        
        # 完全复制fourier_3d_clean.py的结构
        self.fc0 = nn.Linear(self.in_channels + 3, self.width)  # +3 for grid coordinates

        self.conv0 = SpectralConv3d_RC_Correct(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_RC_Correct(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_RC_Correct(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_RC_Correct(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def forward(self, x):
        # 添加网格坐标 - 复制fourier_3d_clean.py的逻辑
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, H, W, T, C] -> [B, C, H, W, T]
        x = F.pad(x, [0, self.padding])

        # 4层FNO-RC块 - 复制fourier_3d_clean.py的结构
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

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # [B, C, H, W, T] -> [B, H, W, T, C]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        """生成3D网格坐标 - 复制fourier_3d_clean.py的逻辑"""
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
