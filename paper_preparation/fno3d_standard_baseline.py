#!/usr/bin/env python3
"""
标准FNO3D基线 - 严格按照原始论文实现
基于: https://arxiv.org/pdf/2010.08895.pdf Section 5.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from scipy.io import loadmat
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
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

################################################################
# 标准FNO3D模型 - 与原始论文完全一致
################################################################
class FNO3d_Standard(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d_Standard, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(13, self.width)
        # input channel is 13: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d_Standard(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_Standard(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_Standard(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_Standard(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        # 注意: 原始版本没有BatchNorm

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2  # 简单相加，没有BatchNorm
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
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
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
# 标准3D数据预处理 - 按照原始论文的方式
################################################################
def preprocess_3d_standard(data_path, T_in=10, T_out=20, ntrain=40, ntest=10):
    """
    标准3D数据预处理，按照原始FNO3D论文的方式
    输入: 前T_in个时间步 + 坐标
    输出: 后T_out个时间步
    但在同一个时间网格上进行预测
    """
    try:
        data = loadmat(data_path)
        u_field = data['u']  # [N, H, W, T]
        print(f"原始数据形状: {u_field.shape}")
        
        N, H, W, T_total = u_field.shape
        
        # 确保有足够的时间步
        if T_total < T_in + T_out:
            print(f"时间步不足: 需要{T_in + T_out}, 实际{T_total}")
            return None, None, None, None
        
        # 按照原始论文的方式：在同一时间网格上预测
        # 输入: 前T_in步的解 + 在整个时间网格上重复
        # 输出: 后T_out步的解
        
        # 选择时间窗口
        T_window = T_in + T_out  # 总时间窗口
        
        train_data = u_field[:ntrain, ..., :T_window]  # [40, 64, 64, 30]
        test_data = u_field[-ntest:, ..., :T_window]   # [10, 64, 64, 30]
        
        # 创建输入：前T_in步在整个时间网格上重复
        def create_input(data):
            N, H, W, T_win = data.shape
            # 取前T_in步
            initial_steps = data[..., :T_in]  # [N, H, W, T_in]
            
            # 在时间维度上扩展到T_window
            # 方法：将前T_in步重复到整个时间窗口
            input_field = torch.zeros(N, H, W, T_win, T_in)
            for i in range(T_in):
                input_field[..., i] = initial_steps[..., i:i+1].expand(-1, -1, -1, T_win)
            
            return input_field  # [N, H, W, T_window, T_in]
        
        train_a = create_input(train_data)  # [40, 64, 64, 30, 10]
        test_a = create_input(test_data)    # [10, 64, 64, 30, 10]
        
        # 输出：后T_out步
        train_u = torch.tensor(train_data[..., T_in:], dtype=torch.float32).unsqueeze(-1)  # [40, 64, 64, 20, 1]
        test_u = torch.tensor(test_data[..., T_in:], dtype=torch.float32).unsqueeze(-1)    # [10, 64, 64, 20, 1]
        
        train_a = train_a.float()
        test_a = test_a.float()
        
        print(f"标准预处理结果:")
        print(f"训练输入: {train_a.shape} (前{T_in}步在时间网格上重复)")
        print(f"训练输出: {train_u.shape} (后{T_out}步)")
        print(f"测试输入: {test_a.shape}")
        print(f"测试输出: {test_u.shape}")
        
        return train_a, train_u, test_a, test_u
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None, None

################################################################
# 损失函数 - 从utilities3导入
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
def train_standard_fno3d(model, train_loader, test_loader, device, epochs=200):
    """标准FNO3D训练"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    myloss = LpLoss(size_average=False)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            # 确保输出和标签形状匹配
            if out.shape != y.shape:
                print(f"形状不匹配: 输出{out.shape} vs 标签{y.shape}")
                return None, None, None
                
            loss = myloss.rel(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 测试
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
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Train {train_loss:.6f}, Test {test_loss:.6f}')
    
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
    
    print("🚀 开始标准FNO3D基线实验")
    
    # 数据加载
    data_path = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
    train_a, train_u, test_a, test_u = preprocess_3d_standard(data_path)
    
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
    
    # 标准FNO3D模型
    model = FNO3d_Standard(modes1=8, modes2=8, modes3=8, width=20)
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print("🔧 开始训练标准FNO3D...")
    trained_model, train_losses, test_losses = train_standard_fno3d(
        model, train_loader, test_loader, device, epochs=200
    )
    
    if trained_model is not None:
        final_test_loss = test_losses[-1]
        print(f"✅ 标准FNO3D最终测试误差: {final_test_loss:.6f}")
        
        # 保存结果
        results = {
            'model': 'Standard_FNO3D',
            'final_test_loss': final_test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        # 可选：保存到文件
        # with open('/content/drive/MyDrive/standard_fno3d_results.json', 'w') as f:
        #     json.dump(results, f, indent=2)
        
    else:
        print("❌ 训练失败")
    
    print("🎉 标准FNO3D基线实验完成！")

if __name__ == "__main__":
    main()
