#!/usr/bin/env python3
"""
3D对比实验 - 简化版，减少错误
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
import os
from datetime import datetime

################################################################
# 简化的数据加载
################################################################
def load_3d_data_simple(data_path):
    """简化的3D数据加载，减少错误"""
    print("📁 加载3D数据...")
    
    with h5py.File(data_path, 'r') as f:
        u_field = np.array(f['u'])  # 直接使用'u'键
        # 转置为标准格式 [N, H, W, T]
        u_field = u_field.transpose(3, 2, 1, 0)
        print(f"数据形状: {u_field.shape}")
        
        # 取前40个样本，确保有足够时间步
        N, H, W, T = u_field.shape
        ntrain, ntest = 30, 10
        T_in, T_out = 10, 20
        
        if T < T_in + T_out:
            print(f"❌ 时间步不足: 需要{T_in + T_out}, 实际{T}")
            return None
        
        # 简化的数据处理
        train_data = u_field[:ntrain]
        test_data = u_field[ntrain:ntrain+ntest]
        
        # 输入：前10步，输出：第10-30步
        train_a = torch.tensor(train_data[..., :T_in], dtype=torch.float32)
        train_u = torch.tensor(train_data[..., T_in:T_in+T_out], dtype=torch.float32)
        test_a = torch.tensor(test_data[..., :T_in], dtype=torch.float32)
        test_u = torch.tensor(test_data[..., T_in:T_in+T_out], dtype=torch.float32)
        
        print(f"训练输入: {train_a.shape}, 训练输出: {train_u.shape}")
        return train_a, train_u, test_a, test_u

################################################################
# 简化的3D FNO模型
################################################################
class SimpleFNO3D(nn.Module):
    def __init__(self, width=32):
        super(SimpleFNO3D, self).__init__()
        self.width = width
        
        # 简化架构
        self.encoder = nn.Sequential(
            nn.Conv3d(1, width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, 1, 3, padding=1)
        )
        
        # 时间映射
        self.time_proj = nn.Linear(10, 20)
    
    def forward(self, x):
        # x: [B, H, W, T_in] -> [B, 1, H, W, T_in]
        B, H, W, T_in = x.shape
        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)  # [B, 1, T_in, H, W]
        
        # 编码
        x = self.encoder(x)  # [B, width, T_in, H, W]
        
        # 时间维度处理
        x = x.permute(0, 1, 3, 4, 2)  # [B, width, H, W, T_in]
        x = self.time_proj(x)         # [B, width, H, W, T_out]
        x = x.permute(0, 1, 4, 2, 3)  # [B, width, T_out, H, W]
        
        # 解码
        x = self.decoder(x)  # [B, 1, T_out, H, W]
        
        # 输出: [B, H, W, T_out]
        return x.squeeze(1).permute(0, 2, 3, 1)

################################################################
# 简化的FNO-RC模型
################################################################
class SimpleFNO_RC(nn.Module):
    def __init__(self, width=32):
        super(SimpleFNO_RC, self).__init__()
        self.width = width
        
        # 主路径
        self.main_path = SimpleFNO3D(width)
        
        # 残差路径
        self.residual_path = nn.Sequential(
            nn.Conv3d(1, width//2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(width//2, 1, 3, padding=1)
        )
        
        self.time_proj_res = nn.Linear(10, 20)
    
    def forward(self, x):
        # 主路径
        main_out = self.main_path(x)
        
        # 残差路径
        B, H, W, T_in = x.shape
        x_res = x.unsqueeze(1).permute(0, 1, 4, 2, 3)  # [B, 1, T_in, H, W]
        x_res = self.residual_path(x_res)  # [B, 1, T_in, H, W]
        x_res = x_res.permute(0, 1, 3, 4, 2)  # [B, 1, H, W, T_in]
        x_res = self.time_proj_res(x_res)     # [B, 1, H, W, T_out]
        x_res = x_res.squeeze(1).permute(0, 3, 1, 2)  # [B, T_out, H, W]
        x_res = x_res.permute(0, 2, 3, 1)  # [B, H, W, T_out]
        
        return main_out + x_res

################################################################
# 损失函数
################################################################
class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
    
    def forward(self, pred, true):
        return torch.mean((pred - true) ** 2) / torch.mean(true ** 2)

################################################################
# 训练函数
################################################################
def train_simple(model, train_loader, test_loader, device, epochs=50):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = SimpleLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        if epoch % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    test_loss += criterion(pred, y).item()
            
            print(f'Epoch {epoch}: Train {train_loss/len(train_loader):.6f}, Test {test_loss/len(test_loader):.6f}')
    
    return train_loss/len(train_loader), test_loss/len(test_loader)

################################################################
# 主函数
################################################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 设备: {device}")
    
    # 数据加载
    data_path = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
    data = load_3d_data_simple(data_path)
    if data is None:
        return
    
    train_a, train_u, test_a, test_u = data
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), 
        batch_size=2, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), 
        batch_size=2, shuffle=False
    )
    
    # 模型对比
    models = {
        'SimpleFNO3D': SimpleFNO3D(width=32),
        'SimpleFNO_RC': SimpleFNO_RC(width=32)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n🔧 训练 {name}...")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        train_loss, test_loss = train_simple(model, train_loader, test_loader, device, epochs=50)
        results[name] = test_loss
        print(f"✅ {name}: {test_loss:.6f}")
    
    # 结果对比
    print(f"\n🏆 结果对比:")
    baseline = results['SimpleFNO3D']
    for name, loss in results.items():
        if name != 'SimpleFNO3D':
            improvement = (baseline - loss) / baseline * 100
            print(f"{name}: {loss:.6f} (改进: {improvement:+.1f}%)")
        else:
            print(f"{name}: {loss:.6f} (基线)")
    
    print("🎉 实验完成！")

if __name__ == "__main__":
    main()
