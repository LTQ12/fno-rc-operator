#!/usr/bin/env python3
"""
第一阶段3D对比实验: FNO-3D vs B-DeepONet-3D vs FNO-RC-3D
独立版本 - 所有依赖都内嵌在此文件中
数据: 3D Navier-Stokes (ns_V1e-4_N10000_T30.mat)
直接上传到Colab根目录即可运行
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.distributions import Normal
import math

# 添加项目根目录到路径
import sys
sys.path.append('/content/fourier_neural_operator-master')
from utilities3 import LpLoss, count_params, MatReader, UnitGaussianNormalizer
from Adam import Adam
from chebyshev import vectorized_batched_cft

torch.manual_seed(42)
np.random.seed(42)

################################################################
# 数据质量检查函数
################################################################
def check_3d_data_quality():
    """检查3D数据的质量、形状、统计特性等"""
    print("🔍 3D Navier-Stokes数据质量检查")
    print("=" * 60)
    
    data_path = "/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat"
    
    try:
        reader = MatReader(data_path)
        u_field = reader.read_field('u')
        print(f"✅ 数据形状: {u_field.shape}")
        print(f"✅ 数据类型: {u_field.dtype}")
        
    except Exception as e:
        print(f"❌ MatReader加载失败: {e}")
        try:
            data = loadmat(data_path)
            print(f"✅ 文件中的键: {list(data.keys())}")
            for key in data.keys():
                if not key.startswith('__'):
                    field_data = data[key]
                    print(f"✅ 字段 '{key}': 形状={field_data.shape}, 类型={field_data.dtype}")
                    if 'u' in key.lower():
                        u_field = field_data
                        break
        except Exception as e2:
            print(f"❌ scipy.io加载也失败: {e2}")
            return None
    
    print(f"\n📊 数据统计分析")
    print(f"数据维度: {len(u_field.shape)}D")
    print(f"数据形状: {u_field.shape}")
    print(f"数据大小: {int(u_field.numel()):,} 个元素")
    # 计算内存占用
    if hasattr(u_field, 'nbytes'):
        memory_gb = u_field.nbytes / (1024**3)
    else:
        memory_gb = int(u_field.numel()) * 4 / (1024**3)  # 假设float32
    print(f"内存占用: {memory_gb:.2f} GB")
    
    print(f"\n数值范围:")
    # 获取统计值，兼容tensor和numpy
    if hasattr(u_field, 'min') and callable(u_field.min):
        min_val = float(u_field.min())
        max_val = float(u_field.max())
        mean_val = float(u_field.mean())
        std_val = float(u_field.std())
    else:
        min_val = float(np.min(u_field))
        max_val = float(np.max(u_field))
        mean_val = float(np.mean(u_field))
        std_val = float(np.std(u_field))
    
    print(f"  最小值: {min_val:.6f}")
    print(f"  最大值: {max_val:.6f}")
    print(f"  均值: {mean_val:.6f}")
    print(f"  标准差: {std_val:.6f}")
    
    # 转换为numpy进行NaN/Inf检查
    if hasattr(u_field, 'numpy'):
        u_field_np = u_field.numpy()
    else:
        u_field_np = np.array(u_field)
    
    nan_count = np.isnan(u_field_np).sum()
    inf_count = np.isinf(u_field_np).sum()
    print(f"\n数据质量:")
    print(f"  NaN数量: {nan_count}")
    print(f"  Inf数量: {inf_count}")
    print(f"  质量状态: {'✅ 良好' if nan_count == 0 and inf_count == 0 else '❌ 有问题'}")
    
    if len(u_field.shape) == 4:
        N, H, W, T = u_field.shape
        print(f"\n📐 维度解析:")
        print(f"  样本数 (N): {N}")
        print(f"  空间高度 (H): {H}")
        print(f"  空间宽度 (W): {W}")
        print(f"  时间步数 (T): {T}")
    
    print("✅ 3D数据质量检查完成！")
    return u_field

################################################################
# 3D CFT函数
################################################################
def cft3d(x, modes1, modes2, modes3, L_segments=8, M_cheb=8):
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
# 3D FNO基线模型
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
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W, T = x.shape

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, H, W, T//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        
        x_fno = torch.fft.irfftn(out_ft, s=(H, W, T))
        return x_fno

class FNO3d_Baseline(nn.Module):
    """3D FNO基线模型，与FNO-RC架构完全一致，但没有CFT残差修正"""
    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels, T_in=10, T_out=20):
        super(FNO3d_Baseline, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T_in = T_in
        self.T_out = T_out
        self.padding = 6
        
        self.fc0 = nn.Linear(self.in_channels + 3, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
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
        x = x.permute(0, 4, 1, 2, 3)
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
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float, device=device)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)

################################################################
# FNO-RC 3D模型
################################################################
class SpectralConv3d_RC(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_RC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

        # CFT残差修正路径
        self.cft_modes1 = modes1 // 2
        self.cft_modes2 = modes2 // 2
        self.cft_modes3 = modes3 // 2
        cft_flat_dim = self.in_channels * self.cft_modes1 * self.cft_modes2 * self.cft_modes3 * 2

        self.correction_generator = nn.Sequential(
            nn.Linear(cft_flat_dim, (self.in_channels + self.out_channels) * 2),
            nn.GELU(),
            nn.Linear((self.in_channels + self.out_channels) * 2, self.out_channels)
        )
        nn.init.zeros_(self.correction_generator[-1].weight)
        nn.init.zeros_(self.correction_generator[-1].bias)

    def forward(self, x):
        B, C, H, W, T = x.shape

        # 主FNO路径
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, H, W, T//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x_fno = torch.fft.irfftn(out_ft, s=(H, W, T))

        # CFT残差修正路径
        cft_coeffs = cft3d(x, self.cft_modes1, self.cft_modes2, self.cft_modes3)
        cft_flat = torch.view_as_real(cft_coeffs).flatten(1)
        correction = self.correction_generator(cft_flat)
        correction = correction.view(B, self.out_channels, 1, 1, 1)

        return x_fno + correction

class FNO_RC_3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels):
        super(FNO_RC_3D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = 6
        self.fc0 = nn.Linear(self.in_channels + 3, self.width)

        self.conv0 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
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
        x = x.permute(0, 4, 1, 2, 3)
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
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float, device=device)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)

################################################################
# 简化的B-DeepONet 3D模型
################################################################
class BDeepONet3D_Simplified(nn.Module):
    def __init__(self, modes1=8, modes2=8, modes3=8, width=20, in_channels=10, out_channels=1):
        super(BDeepONet3D_Simplified, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2  
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 简化的分支网络
        self.branch_net = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(width, width * 2),
            nn.ReLU(),
            nn.Linear(width * 2, width)
        )
        
        # 简化的主干网络
        self.trunk_net = nn.Sequential(
            nn.Linear(3, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width)
        )
        
        self.output_layer = nn.Linear(width, out_channels)

    def forward(self, x):
        if x.dim() == 5:
            x = x.permute(0, 4, 1, 2, 3)  # (batch, channels, height, width, time)
        
        batch_size = x.size(0)
        H, W, T = x.size(2), x.size(3), x.size(4)
        
        # 分支网络处理
        branch_out = self.branch_net(x)  # (batch, width)
        
        # 创建查询网格点
        device = x.device
        h_coords = torch.linspace(0, 1, H, device=device)
        w_coords = torch.linspace(0, 1, W, device=device)
        t_coords = torch.linspace(0, 1, T, device=device)
        
        # 创建网格
        grid_h, grid_w, grid_t = torch.meshgrid(h_coords, w_coords, t_coords, indexing='ij')
        coords = torch.stack([grid_h.flatten(), grid_w.flatten(), grid_t.flatten()], dim=-1)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, H*W*T, 3)
        
        # 主干网络处理
        trunk_out = self.trunk_net(coords)  # (batch, H*W*T, width)
        
        # DeepONet点积
        branch_out = branch_out.unsqueeze(1)  # (batch, 1, width)
        output = torch.sum(branch_out * trunk_out, dim=-1)  # (batch, H*W*T)
        
        # 重塑为3D输出
        output = output.view(batch_size, H, W, T)
        
        # 只取最后一个时间步作为预测
        output = output[..., -1].unsqueeze(-1)  # (batch, H, W, 1)
        
        return output
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float, device=device)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)

################################################################
# 实验跟踪器
################################################################
class ExperimentTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.results = {}
        self.start_time = datetime.now()
        os.makedirs(save_dir, exist_ok=True)
        
    def log_model_info(self, model_name, model, args):
        self.results[model_name] = {
            'model_info': {
                'parameters': count_params(model),
                'modes': args.modes,
                'width': args.width,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs,
                'batch_size': args.batch_size
            },
            'training_history': {
                'train_loss': [],
                'test_loss': [],
                'epoch_times': []
            },
            'final_results': {}
        }
        
    def log_epoch(self, model_name, epoch, train_loss, test_loss, epoch_time):
        self.results[model_name]['training_history']['train_loss'].append(train_loss)
        self.results[model_name]['training_history']['test_loss'].append(test_loss)
        self.results[model_name]['training_history']['epoch_times'].append(epoch_time)
        
    def log_final_results(self, model_name, final_train_loss, final_test_loss, total_time):
        self.results[model_name]['final_results'] = {
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'total_training_time': total_time,
            'avg_epoch_time': np.mean(self.results[model_name]['training_history']['epoch_times'])
        }
        
    def save_results(self):
        if 'FNO_Baseline' in self.results and 'FNO_RC' in self.results:
            baseline_error = self.results['FNO_Baseline']['final_results']['final_test_loss']
            fno_rc_error = self.results['FNO_RC']['final_results']['final_test_loss']
            improvement = (baseline_error - fno_rc_error) / baseline_error * 100
            self.results['comparison'] = {
                'fno_rc_vs_baseline_improvement': f"{improvement:.2f}%"
            }
        
        results_file = os.path.join(self.save_dir, 'phase1_3d_comparison_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\n📊 实验结果已保存到: {results_file}")
        
    def plot_training_curves(self):
        plt.figure(figsize=(15, 5))
        
        # 训练损失
        plt.subplot(1, 3, 1)
        for model_name in self.results.keys():
            if model_name != 'comparison':
                train_losses = self.results[model_name]['training_history']['train_loss']
                epochs = range(1, len(train_losses) + 1)
                plt.plot(epochs, train_losses, label=model_name, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training L2 Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 测试损失
        plt.subplot(1, 3, 2)
        for model_name in self.results.keys():
            if model_name != 'comparison':
                test_losses = self.results[model_name]['training_history']['test_loss']
                epochs = range(1, len(test_losses) + 1)
                plt.plot(epochs, test_losses, label=model_name, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Test L2 Loss')
        plt.title('Test Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 最终对比
        plt.subplot(1, 3, 3)
        model_names = []
        test_errors = []
        for model_name in self.results.keys():
            if model_name != 'comparison':
                model_names.append(model_name)
                test_errors.append(self.results[model_name]['final_results']['final_test_loss'])
        
        bars = plt.bar(model_names, test_errors, 
                      color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        plt.xlabel('Model')
        plt.ylabel('Final Test L2 Loss')
        plt.title('Final Performance Comparison')
        plt.yscale('log')
        
        for bar, error in zip(bars, test_errors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{error:.6f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.save_dir, 'phase1_3d_training_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 训练曲线已保存到: {plot_file}")
        plt.close()

################################################################
# 训练函数
################################################################
def train_model(model, model_name, train_loader, test_loader, args, device, tracker, y_normalizer):
    print(f"\n🚀 开始训练 {model_name}")
    print("=" * 60)
    
    tracker.log_model_info(model_name, model, args)
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    loss_func = LpLoss(size_average=False)
    
    print(f"📊 模型参数量: {count_params(model):,}")
    print(f"🔧 超参数: modes={args.modes}, width={args.width}, lr={args.learning_rate}")
    
    model_start_time = default_timer()
    
    for ep in range(args.epochs):
        model.train()
        epoch_start = default_timer()
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(x).squeeze(-1)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze(-1)
                
                out_decoded = y_normalizer.decode(out)
                y_decoded = y_normalizer.decode(y)
                test_l2 += loss_func(out_decoded, y_decoded).item()
        
        train_l2 /= args.ntrain
        test_l2 /= args.ntest
        
        epoch_time = default_timer() - epoch_start
        tracker.log_epoch(model_name, ep + 1, train_l2, test_l2, epoch_time)
        
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1:3d}/{args.epochs} | Time: {epoch_time:.2f}s | "
                  f"Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}")
    
    total_time = default_timer() - model_start_time
    
    final_train_loss = tracker.results[model_name]['training_history']['train_loss'][-1]
    final_test_loss = tracker.results[model_name]['training_history']['test_loss'][-1]
    tracker.log_final_results(model_name, final_train_loss, final_test_loss, total_time)
    
    print(f"✅ {model_name} 训练完成!")
    print(f"   最终测试误差: {final_test_loss:.6f}")
    print(f"   总训练时间: {total_time:.1f}秒")
    
    return model

################################################################
# 主函数
################################################################
def main():
    class Args:
        def __init__(self):
            self.data_path = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
            self.save_dir = '/content/drive/MyDrive/FNO_RC_Experiments/phase1_3d_comparison'
            self.ntrain = 40  # 实际可用样本数，留10个做测试
            self.ntest = 10
            self.T_in = 10
            self.T_out = 20
            self.modes = 8
            self.width = 20
            self.epochs = 200
            self.batch_size = 10
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.scheduler_step = 100
            self.scheduler_gamma = 0.5
    
    args = Args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("🚀 开始3D实验")
    tracker = ExperimentTracker(args.save_dir)
    
    # 数据加载
    print("📂 加载数据...")
    try:
        reader = MatReader(args.data_path)
        u_field = reader.read_field('u')
        
        min_time_steps = args.T_in + args.T_out
        if u_field.shape[-1] < min_time_steps:
            print(f"❌ 时间步不足: 需要{min_time_steps}, 实际{u_field.shape[-1]}")
            return
            
        train_a = u_field[:args.ntrain, ..., :args.T_in]
        train_u = u_field[:args.ntrain, ..., args.T_in:args.T_in + args.T_out]
        test_a = u_field[-args.ntest:, ..., :args.T_in]
        test_u = u_field[-args.ntest:, ..., args.T_in:args.T_in + args.T_out]
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)
    
    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)
    y_normalizer.to(device)
    
    # 3D数据预处理: [N, H, W, T] -> [N, H, W, T, 1] (时间作为第3个空间维度，通道=1)
    train_a = train_a.unsqueeze(-1)  # [40, 64, 64, 10, 1]
    train_u = train_u.unsqueeze(-1)  # [40, 64, 64, 20, 1] 
    test_a = test_a.unsqueeze(-1)    # [10, 64, 64, 10, 1]
    test_u = test_u.unsqueeze(-1)    # [10, 64, 64, 20, 1]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), 
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), 
        batch_size=args.batch_size, shuffle=False
    )
    
    # 模型定义 - 输入输出都是1个通道，但需要处理时间维度变化
    models = {
        'FNO_Baseline': FNO3d_Baseline(
            args.modes, args.modes, args.modes, args.width, 
            in_channels=1, out_channels=1, T_in=args.T_in, T_out=args.T_out
        ).to(device),
        
        'B_DeepONet': BDeepONet3D_Simplified(
            args.modes, args.modes, args.modes, args.width,
            in_channels=1, out_channels=1, T_in=args.T_in, T_out=args.T_out
        ).to(device),
        
        'FNO_RC': FNO_RC_3D(
            args.modes, args.modes, args.modes, args.width,
            in_channels=1, out_channels=1, T_in=args.T_in, T_out=args.T_out
        ).to(device)
    }
    
    # 训练所有模型
    trained_models = {}
    for model_name, model in models.items():
        trained_models[model_name] = train_model(
            model, model_name, train_loader, test_loader, args, device, tracker, y_normalizer
        )
    
    # 保存结果和绘制图表
    tracker.save_results()
    tracker.plot_training_curves()
    
    # 输出最终对比结果
    print("\n🏆 实验结果:")
    
    results_summary = []
    for model_name in ['FNO_Baseline', 'B_DeepONet', 'FNO_RC']:
        if model_name in tracker.results:
            result = tracker.results[model_name]['final_results']
            results_summary.append({
                'model': model_name,
                'test_error': result['final_test_loss'],
                'params': tracker.results[model_name]['model_info']['parameters'],
                'time': result['total_training_time']
            })
    
    results_summary.sort(key=lambda x: x['test_error'])
    
    baseline_error = None
    for result in results_summary:
        if result['model'] == 'FNO_Baseline':
            baseline_error = result['test_error']
            improvement = "基线"
        else:
            if baseline_error:
                improvement = f"{(baseline_error - result['test_error']) / baseline_error * 100:+.1f}%"
            else:
                improvement = "N/A"
        
        print(f"{result['model']}: {result['test_error']:.6f} ({improvement})")
    
    print("🎉 实验完成！")

if __name__ == "__main__":
    main()
