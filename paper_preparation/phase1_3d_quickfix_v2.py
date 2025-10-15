#!/usr/bin/env python3
"""
快速修复版本 - 3D实验
解决损失计算中的张量大小不匹配问题
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
# 3D数据预处理 - 简化版本
################################################################
def preprocess_3d_data(data_path, T_in=10, T_out=20, ntrain=40, ntest=10):
    """简化的3D数据预处理，确保输入输出维度匹配"""
    try:
        data = loadmat(data_path)
        u_field = data['u']  # [N, H, W, T]
        print(f"原始数据形状: {u_field.shape}")
        
        # 确保有足够的时间步
        if u_field.shape[-1] < T_in + T_out:
            print(f"时间步不足: 需要{T_in + T_out}, 实际{u_field.shape[-1]}")
            return None, None, None, None
        
        # 数据切片 - 使用相同的时间窗口进行训练
        train_data = u_field[:ntrain, ..., :T_in+T_out]  # [40, 64, 64, 30]
        test_data = u_field[-ntest:, ..., :T_in+T_out]   # [10, 64, 64, 30]
        
        # 分离输入输出 - 但使用滑动窗口方式
        train_a = train_data[..., :T_in]      # [40, 64, 64, 10]
        train_u = train_data[..., T_in:T_in+T_out]  # [40, 64, 64, 20]
        test_a = test_data[..., :T_in]        # [10, 64, 64, 10]  
        test_u = test_data[..., T_in:T_in+T_out]    # [10, 64, 64, 20]
        
        # 转换为张量
        train_a = torch.tensor(train_a, dtype=torch.float32)
        train_u = torch.tensor(train_u, dtype=torch.float32)
        test_a = torch.tensor(test_a, dtype=torch.float32)
        test_u = torch.tensor(test_u, dtype=torch.float32)
        
        print(f"训练输入: {train_a.shape}, 训练输出: {train_u.shape}")
        print(f"测试输入: {test_a.shape}, 测试输出: {test_u.shape}")
        
        return train_a, train_u, test_a, test_u
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None, None

################################################################
# 简化的3D FNO模型 - 直接处理时序预测
################################################################
class Simple3DFNO(nn.Module):
    """简化的3D FNO模型，直接处理时序预测"""
    def __init__(self, T_in=10, T_out=20, modes=8, width=20):
        super(Simple3DFNO, self).__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.modes = modes
        self.width = width
        
        # 编码器：将输入时序映射到隐藏状态
        self.encoder = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # 频谱层
        self.spectral = nn.ModuleList([
            nn.Conv3d(width, width, kernel_size=1) for _ in range(4)
        ])
        
        # 解码器：从隐藏状态生成输出时序
        self.decoder = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, 1, kernel_size=3, padding=1)
        )
        
        # 时间维度映射
        self.time_mapper = nn.Linear(T_in, T_out)
        
    def forward(self, x):
        # x: [B, H, W, T_in] -> [B, 1, H, W, T_in]
        B, H, W, T = x.shape
        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)  # [B, 1, T_in, H, W]
        
        # 编码
        x = self.encoder(x)  # [B, width, T_in, H, W]
        
        # 频谱处理
        for layer in self.spectral:
            x = F.gelu(layer(x) + x)
        
        # 时间维度映射
        x = x.permute(0, 1, 3, 4, 2)  # [B, width, H, W, T_in]
        x = self.time_mapper(x)       # [B, width, H, W, T_out]
        x = x.permute(0, 1, 4, 2, 3)  # [B, width, T_out, H, W]
        
        # 解码
        x = self.decoder(x)  # [B, 1, T_out, H, W]
        
        # 输出格式：[B, H, W, T_out]
        x = x.squeeze(1).permute(0, 2, 3, 1)
        
        return x

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
def train_model(model, train_loader, test_loader, device, epochs=50):
    """简化的训练函数"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
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
                return None
                
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
        
        if epoch % 10 == 0:
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
    
    print("🚀 开始3D实验（简化版）")
    
    # 数据加载
    data_path = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
    train_a, train_u, test_a, test_u = preprocess_3d_data(data_path)
    
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
    
    # 模型训练
    models = {
        'Simple_FNO': Simple3DFNO(T_in=10, T_out=20, modes=8, width=20)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n🔧 训练 {name}...")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        trained_model, train_losses, test_losses = train_model(
            model, train_loader, test_loader, device, epochs=50
        )
        
        if trained_model is not None:
            final_test_loss = test_losses[-1]
            results[name] = final_test_loss
            print(f"✅ {name}: {final_test_loss:.6f}")
        else:
            print(f"❌ {name}: 训练失败")
    
    # 结果总结
    print(f"\n🎯 实验结果:")
    for name, loss in results.items():
        print(f"{name}: {loss:.6f}")
    
    print("🎉 实验完成！")

if __name__ == "__main__":
    main()
