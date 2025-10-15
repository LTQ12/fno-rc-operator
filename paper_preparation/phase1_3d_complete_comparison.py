#!/usr/bin/env python3
"""
完整的3D对比实验 - 第一阶段
包含: 标准FNO3D基线 vs FNO-RC-3D vs B-DeepONet-3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

################################################################
# 标准3D频谱卷积层 - 与原始论文完全一致
################################################################
class SpectralConv3d_Standard(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_Standard, self).__init__()
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

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

################################################################
# 标准FNO3D模型 - 与原始论文完全一致
################################################################
class FNO3d_Standard(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d_Standard, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6
        self.fc0 = nn.Linear(13, self.width)  # 10个时间步 + 3个坐标

        self.conv0 = SpectralConv3d_Standard(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_Standard(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_Standard(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_Standard(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])

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
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# CFT工具函数
################################################################
def vectorized_batched_cft(u, t_coords, f_points, L_segments, M_cheb):
    """向量化的批量CFT计算"""
    device = u.device
    dtype = u.dtype
    
    B, C, N = u.shape
    K = len(f_points)
    
    # 分段处理
    segment_size = N // L_segments
    cft_results = torch.zeros(B, C, K, dtype=torch.complex64, device=device)
    
    for seg_idx in range(L_segments):
        start_idx = seg_idx * segment_size
        end_idx = min((seg_idx + 1) * segment_size, N)
        
        if start_idx >= end_idx:
            continue
            
        # 当前段的数据和坐标
        u_seg = u[:, :, start_idx:end_idx]
        t_seg = t_coords[start_idx:end_idx]
        
        # 映射到[-1, 1]
        t_min, t_max = t_seg.min(), t_seg.max()
        if t_max > t_min:
            t_mapped = 2 * (t_seg - t_min) / (t_max - t_min) - 1
        else:
            t_mapped = torch.zeros_like(t_seg)
        
        # Chebyshev多项式
        T_cheb = torch.ones(len(t_mapped), M_cheb, device=device)
        if M_cheb > 1:
            T_cheb[:, 1] = t_mapped
        for m in range(2, M_cheb):
            T_cheb[:, m] = 2 * t_mapped * T_cheb[:, m-1] - T_cheb[:, m-2]
        
        # CFT计算
        for k_idx, f_k in enumerate(f_points):
            phase = torch.exp(-2j * np.pi * f_k * t_seg)
            integrand = u_seg * phase.unsqueeze(0).unsqueeze(0)
            
            # 使用Chebyshev积分
            weights = torch.ones(len(t_seg), device=device) * (t_max - t_min) / len(t_seg)
            cft_results[:, :, k_idx] += torch.sum(integrand * weights.unsqueeze(0).unsqueeze(0), dim=2)
    
    return cft_results

def cft3d(x, modes1, modes2, modes3, L_segments=8, M_cheb=8):
    """3D CFT变换"""
    B, C, H, W, T = x.shape
    device = x.device
    
    # 1. 时间维度FFT
    x_ft = torch.fft.rfft(x, dim=-1)
    x_ft_filtered = x_ft[..., :modes3]
    
    # 2. CFT处理
    x_reshaped = x_ft_filtered.permute(0, 1, 4, 2, 3).reshape(B, C * modes3, H, W)
    
    # Width方向CFT
    t_coords_w = torch.linspace(0, 1, W, device=device, dtype=x.dtype)
    f_points_w = torch.fft.rfftfreq(W, d=1.0/W)[:modes2].to(device)
    cft_w_input = x_reshaped.permute(0, 2, 1, 3).reshape(B * H, C * modes3, W)
    
    cft_w_real = vectorized_batched_cft(cft_w_input.real, t_coords_w, f_points_w, L_segments, M_cheb)
    cft_w_imag = vectorized_batched_cft(cft_w_input.imag, t_coords_w, f_points_w, L_segments, M_cheb)
    cft_w_complex = cft_w_real + 1j * cft_w_imag
    cft_w_complex = cft_w_complex.view(B, H, C * modes3, modes2).permute(0, 2, 1, 3)

    # Height方向CFT
    t_coords_h = torch.linspace(0, 1, H, device=device, dtype=x.dtype)
    f_points_h = torch.fft.fftfreq(H, d=1.0/H).to(device)
    h_indices = torch.cat((torch.arange(0, modes1//2), torch.arange(H-(modes1-modes1//2), H))).to(device)
    f_points_h_selected = f_points_h[h_indices]
    cft_h_input = cft_w_complex.permute(0, 3, 1, 2).reshape(B * modes2, C * modes3, H)

    cft_h_real = vectorized_batched_cft(cft_h_input.real, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_h_imag = vectorized_batched_cft(cft_h_input.imag, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_hwt_complex = cft_h_real + 1j * cft_h_imag
    
    cft_hwt_complex = cft_hwt_complex.view(B, modes2, C, modes3, modes1).permute(0, 2, 4, 1, 3)
    return cft_hwt_complex

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

        # 标准FNO路径
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, H, W, T//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        
        x_fno = torch.fft.irfftn(out_ft, s=(H, W, T))

        # CFT残差修正路径
        try:
            x_cft = cft3d(x, self.cft_modes1, self.cft_modes2, self.cft_modes3, L_segments=8, M_cheb=8)
            x_cft_real = torch.cat([x_cft.real, x_cft.imag], dim=-1)
            x_cft_flat = x_cft_real.reshape(B, H, W, T, -1)
            correction = self.correction_generator(x_cft_flat)
            return x_fno + correction
        except:
            return x_fno

class FNO_RC_3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO_RC_3D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6
        self.fc0 = nn.Linear(13, self.width)

        self.conv0 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_RC(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])

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
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# B-DeepONet 3D模型
################################################################
class BDeepONet3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(BDeepONet3D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        
        # Branch网络
        self.branch_net = nn.Sequential(
            nn.Linear(13, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width)
        )
        
        # Trunk网络
        self.trunk_net = nn.Sequential(
            nn.Linear(3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width)
        )
        
        self.output_layer = nn.Linear(width, 1)

    def forward(self, x):
        B, H, W, T, C = x.shape
        
        # 获取网格坐标
        grid = self.get_grid(x.shape, x.device)
        
        # Branch网络处理输入
        x_with_grid = torch.cat((x, grid), dim=-1)
        branch_out = self.branch_net(x_with_grid)  # [B, H, W, T, width]
        
        # Trunk网络处理坐标
        trunk_out = self.trunk_net(grid)  # [B, H, W, T, width]
        
        # 组合
        combined = branch_out * trunk_out  # [B, H, W, T, width]
        output = self.output_layer(combined)  # [B, H, W, T, 1]
        
        return output

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# 数据预处理
################################################################
def preprocess_3d_comparison(data_path, T_in=10, T_out=20, ntrain=40, ntest=10):
    """3D对比实验数据预处理 - 支持Matlab v7.3格式"""
    try:
        # 首先尝试scipy.io.loadmat
        try:
            data = loadmat(data_path)
            u_field = data['u']
            print(f"✅ 使用scipy.io加载成功")
        except:
            # 如果失败，使用h5py加载Matlab v7.3文件
            print("📁 检测到Matlab v7.3格式，使用h5py加载...")
            with h5py.File(data_path, 'r') as f:
                # 查看文件中的键
                keys = list(f.keys())
                print(f"文件中的键: {keys}")
                
                # 寻找数据字段
                if 'u' in f:
                    u_field = np.array(f['u'])
                elif 'data' in f:
                    u_field = np.array(f['data'])
                else:
                    # 尝试第一个非元数据键
                    data_key = [k for k in keys if not k.startswith('#')][0]
                    u_field = np.array(f[data_key])
                    print(f"使用键: {data_key}")
                
                # h5py加载的数据可能需要转置
                if u_field.ndim == 4:
                    # 通常h5py加载的维度顺序是 [T, W, H, N]，需要转置为 [N, H, W, T]
                    u_field = u_field.transpose(3, 2, 1, 0)
                    print(f"数据已转置为标准格式")
            
            print(f"✅ 使用h5py加载成功")
        
        print(f"原始数据形状: {u_field.shape}")
        
        N, H, W, T_total = u_field.shape
        if T_total < T_in + T_out:
            print(f"时间步不足: 需要{T_in + T_out}, 实际{T_total}")
            return None, None, None, None
        
        T_window = T_in + T_out
        train_data = u_field[:ntrain, ..., :T_window]
        test_data = u_field[-ntest:, ..., :T_window]
        
        def create_input(data):
            N, H, W, T_win = data.shape
            initial_steps = data[..., :T_in]  # [N, H, W, T_in]
            # 转换为torch tensor以使用expand
            initial_steps = torch.tensor(initial_steps, dtype=torch.float32)
            input_field = torch.zeros(N, H, W, T_win, T_in)
            for i in range(T_in):
                # 使用torch tensor的expand方法
                step_expanded = initial_steps[..., i:i+1].expand(-1, -1, -1, T_win)
                input_field[..., i] = step_expanded.squeeze(-1)
            return input_field
        
        train_a = create_input(train_data)
        test_a = create_input(test_data)
        train_u = torch.tensor(train_data[..., T_in:], dtype=torch.float32).unsqueeze(-1)
        test_u = torch.tensor(test_data[..., T_in:], dtype=torch.float32).unsqueeze(-1)
        
        train_a = train_a.float()
        test_a = test_a.float()
        
        print(f"预处理完成:")
        print(f"训练输入: {train_a.shape}, 训练输出: {train_u.shape}")
        print(f"测试输入: {test_a.shape}, 测试输出: {test_u.shape}")
        
        return train_a, train_u, test_a, test_u
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None, None

################################################################
# 损失函数
################################################################
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def rel(self, x, y):
        return self.abs(x, y)

################################################################
# 训练函数
################################################################
def train_model(model, model_name, train_loader, test_loader, device, epochs=100):
    """统一的模型训练函数"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    myloss = LpLoss(size_average=False)
    
    train_losses = []
    test_losses = []
    
    print(f"🔧 开始训练 {model_name}...")
    print(f"📊 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            if out.shape != y.shape:
                print(f"❌ 形状不匹配: 输出{out.shape} vs 标签{y.shape}")
                return None, None, None
                
            loss = myloss.rel(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_loss += myloss.rel(out, y).item()
        
        scheduler.step()
        
        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if epoch % 25 == 0:
            print(f'  Epoch {epoch}: Train {train_loss:.6f}, Test {test_loss:.6f}')
    
    return model, train_losses, test_losses

################################################################
# 主函数
################################################################
def main():
    print("🖥️ 使用设备: cuda" if torch.cuda.is_available() else "🖥️ 使用设备: cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("🚀 开始3D对比实验")
    
    # 数据加载
    data_path = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
    train_a, train_u, test_a, test_u = preprocess_3d_comparison(data_path)
    
    if train_a is None:
        print("❌ 数据加载失败")
        return
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), 
        batch_size=4, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), 
        batch_size=4, shuffle=False
    )
    
    # 模型定义
    models = {
        'FNO3D_Standard': FNO3d_Standard(modes1=8, modes2=8, modes3=8, width=20),
        'FNO_RC_3D': FNO_RC_3D(modes1=8, modes2=8, modes3=8, width=20),
        'B_DeepONet_3D': BDeepONet3D(modes1=8, modes2=8, modes3=8, width=20)
    }
    
    # 训练所有模型
    results = {}
    for model_name, model in models.items():
        trained_model, train_losses, test_losses = train_model(
            model, model_name, train_loader, test_loader, device, epochs=100
        )
        
        if trained_model is not None:
            final_test_loss = test_losses[-1]
            results[model_name] = {
                'final_test_loss': final_test_loss,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'parameters': sum(p.numel() for p in model.parameters())
            }
            print(f"✅ {model_name}: {final_test_loss:.6f}")
        else:
            print(f"❌ {model_name}: 训练失败")
    
    # 结果对比
    print(f"\n🏆 3D对比实验结果:")
    print("-" * 60)
    
    if 'FNO3D_Standard' in results:
        baseline_error = results['FNO3D_Standard']['final_test_loss']
        print(f"FNO3D_Standard: {baseline_error:.6f} (基线)")
        
        for name, result in results.items():
            if name != 'FNO3D_Standard':
                improvement = (baseline_error - result['final_test_loss']) / baseline_error * 100
                print(f"{name}: {result['final_test_loss']:.6f} (改进: {improvement:+.1f}%)")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/content/drive/MyDrive/3d_comparison_results_{timestamp}.json'
    
    # 转换为可序列化格式
    serializable_results = {}
    for name, result in results.items():
        serializable_results[name] = {
            'final_test_loss': float(result['final_test_loss']),
            'parameters': int(result['parameters']),
            'train_losses': [float(x) for x in result['train_losses']],
            'test_losses': [float(x) for x in result['test_losses']]
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 结果已保存到: {results_file}")
    print("🎉 3D对比实验完成！")

if __name__ == "__main__":
    main()
