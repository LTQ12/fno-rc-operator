"""
可靠的3D FNO基线实现
基于fourier_3d_cft_residual.py的架构，但移除CFT部分，确保完全一致的对比
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

import sys
sys.path.append('/content/fourier_neural_operator-master')
from utilities3 import *

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)
np.random.seed(0)

################################################################
# 3D Fourier层 - 与FNO-RC完全一致的实现
################################################################

class SpectralConv3d(nn.Module):
    """标准3D频谱卷积层，与FNO-RC中的主路径完全一致"""
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        # 与FNO-RC完全相同的权重初始化
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W, T = x.shape

        # 与FNO-RC主路径完全相同的FFT处理
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, H, W, T//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # 四个象限的频域操作，与FNO-RC完全一致
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        
        # 逆FFT回到物理空间
        x_fno = torch.fft.irfftn(out_ft, s=(H, W, T))
        
        # 基线FNO只返回主路径结果，没有残差修正
        return x_fno

class FNO3d_Baseline(nn.Module):
    """3D FNO基线模型，与FNO-RC架构完全一致，但没有CFT残差修正"""
    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels):
        super(FNO3d_Baseline, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = 6  # 与FNO-RC相同的padding
        
        # 与FNO-RC完全相同的网络结构
        self.fc0 = nn.Linear(self.in_channels + 3, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        # 与FNO-RC相同的BatchNorm
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        # 与FNO-RC相同的输出层
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def forward(self, x):
        # 与FNO-RC完全相同的预处理
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])

        # 与FNO-RC相同的4层结构
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

        # 与FNO-RC相同的后处理
        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        """与FNO-RC完全相同的网格生成"""
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        # 第三个空间维度的占位符
        gridz = torch.zeros(batchsize, size_x, size_y, 1, device=device)
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
