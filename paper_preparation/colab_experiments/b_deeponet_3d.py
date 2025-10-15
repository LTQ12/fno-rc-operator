"""
B-DeepONet 3D实现 - 贝叶斯深度算子网络
基于2023年最新的B-DeepONet论文，适配3D Navier-Stokes问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import math

import sys
sys.path.append('/content/fourier_neural_operator-master')
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

################################################################
# B-DeepONet 3D实现
################################################################

class BranchNet3D(nn.Module):
    """分支网络 - 处理输入函数"""
    def __init__(self, input_dim, branch_dim, hidden_dim=128):
        super(BranchNet3D, self).__init__()
        self.input_dim = input_dim
        self.branch_dim = branch_dim
        
        # 3D卷积层处理空间-时间输入
        self.conv1 = nn.Conv3d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool3d(1)  # 全局平均池化
        
        # 全连接层
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, branch_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch, time_steps, height, width, depth) -> (batch, time_steps, height, width, depth)
        # 需要调整为卷积格式: (batch, channels, depth, height, width)
        if len(x.shape) == 5:
            x = x.permute(0, 1, 4, 2, 3)  # (B, T, D, H, W)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 全局平均池化
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class TrunkNet3D(nn.Module):
    """主干网络 - 处理查询点坐标"""
    def __init__(self, coord_dim, trunk_dim, hidden_dim=128):
        super(TrunkNet3D, self).__init__()
        self.coord_dim = coord_dim  # 通常是3 (x, y, t)
        self.trunk_dim = trunk_dim
        
        # 多层感知机
        self.fc1 = nn.Linear(coord_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, trunk_dim)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, coords):
        # coords shape: (batch, num_points, coord_dim)
        x = F.relu(self.fc1(coords))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

class BayesianLayer(nn.Module):
    """贝叶斯线性层 - 引入不确定性量化"""
    def __init__(self, in_features, out_features):
        super(BayesianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重的均值和方差参数
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # 偏置的均值和方差参数
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
        
        # 先验分布参数
        self.prior_sigma = 1.0

    def forward(self, x):
        # 重参数化技巧采样权重
        weight_sigma = torch.log(1 + torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """计算KL散度正则化项"""
        weight_sigma = torch.log(1 + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
        
        # 权重的KL散度
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_sigma ** 2) / (self.prior_sigma ** 2) - 
            torch.log(weight_sigma ** 2) + torch.log(self.prior_sigma ** 2) - 1
        )
        
        # 偏置的KL散度
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_sigma ** 2) / (self.prior_sigma ** 2) - 
            torch.log(bias_sigma ** 2) + torch.log(self.prior_sigma ** 2) - 1
        )
        
        return weight_kl + bias_kl

class BDeepONet3D(nn.Module):
    """3D贝叶斯深度算子网络"""
    def __init__(self, input_dim=10, coord_dim=3, hidden_dim=128, output_dim=1):
        super(BDeepONet3D, self).__init__()
        self.input_dim = input_dim
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 分支网络和主干网络的输出维度
        self.branch_dim = hidden_dim
        self.trunk_dim = hidden_dim
        
        # 网络组件
        self.branch_net = BranchNet3D(input_dim, self.branch_dim, hidden_dim)
        self.trunk_net = TrunkNet3D(coord_dim, self.trunk_dim, hidden_dim)
        
        # 贝叶斯输出层
        self.bayesian_layer = BayesianLayer(self.branch_dim, output_dim)
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, u, coords):
        """
        前向传播
        u: 输入函数 (batch, spatial_dims..., time_steps)
        coords: 查询点坐标 (batch, num_points, coord_dim)
        """
        # 分支网络处理输入函数
        branch_out = self.branch_net(u)  # (batch, branch_dim)
        
        # 主干网络处理查询点
        trunk_out = self.trunk_net(coords)  # (batch, num_points, trunk_dim)
        
        # DeepONet的核心：点积操作
        # branch_out: (batch, branch_dim) -> (batch, 1, branch_dim)
        # trunk_out: (batch, num_points, trunk_dim)
        branch_out = branch_out.unsqueeze(1)  # (batch, 1, branch_dim)
        
        # 点积 + 贝叶斯层
        if self.branch_dim == self.trunk_dim:
            # 标准点积
            dot_product = torch.sum(branch_out * trunk_out, dim=-1)  # (batch, num_points)
        else:
            # 如果维度不匹配，使用贝叶斯层进行变换
            dot_product = self.bayesian_layer(branch_out).squeeze(-1)  # (batch, 1)
            dot_product = dot_product.expand(-1, trunk_out.size(1))  # (batch, num_points)
        
        # 添加偏置
        output = dot_product + self.bias
        
        return output.unsqueeze(-1)  # (batch, num_points, output_dim)
    
    def kl_loss(self):
        """计算KL散度损失"""
        return self.bayesian_layer.kl_divergence()

class BDeepONet3D_Simplified(nn.Module):
    """简化版B-DeepONet，更适合与FNO对比"""
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
        
        # 输出层
        self.output_layer = nn.Linear(width, out_channels)
        
        # 贝叶斯不确定性参数
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (batch, height, width, time, channels)
        """
        # 调整输入格式
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
        """生成网格坐标（兼容性方法）"""
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        gridz = torch.zeros(batchsize, size_x, size_y, 1, device=device)
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
