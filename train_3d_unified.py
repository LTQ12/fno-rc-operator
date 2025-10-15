%%writefile train_3d_unified.py

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
import time

# 确保可以导入本地模块
import sys
if '/content/' not in sys.path:
    sys.path.append('/content/')

from fourier_3d_clean import FNO3d # <--- 使用 fourier_3d_clean.py
from fourier_3d_cft_residual import FNO_RC_3D
from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss
from Adam import Adam

def train_model(args):
    # ... (函数体和之前完全一样, 这里省略)
    """
    统一的训练脚本，用于训练 FNO 基线模型或 FNO-RC 模型。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Training on {device} ---")
    print(f"Model Type: {args.model_type.upper()}")

    # 1. 加载数据和准备 DataLoader
    print("Loading data...")
    try:
        reader = MatReader(args.data_path)
        u_field = reader.read_field('u')
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {args.data_path}")
        print("Please ensure your Google Drive is mounted and the path is correct.")
        return

    ntrain = args.ntrain
    ntest = args.ntest
    
    # 提取输入和输出序列
    train_a = u_field[:ntrain, ..., :args.T_in]
    train_u = u_field[:ntrain, ..., args.T_in:args.T_in + args.T_out]

    # 数据归一化
    a_normalizer = UnitGaussianNormalizer(train_a)
    y_normalizer = UnitGaussianNormalizer(train_u)
    
    train_a_norm = a_normalizer.encode(train_a)
    train_u_norm = y_normalizer.encode(train_u)

    # 增加坐标网格
    grid = get_grid(train_a.shape, device) # (B, H, W, D, 3)
    # 将 grid 扩展到 batch size
    grid_repeated = grid.repeat(train_a.shape[0], 1, 1, 1, 1)

    # 合并输入和坐标
    # train_a_norm: (B, H, W, D) -> (B, H, W, D, 1)
    train_input = torch.cat((train_a_norm.unsqueeze(-1), grid_repeated.cpu()), dim=-1) # (B, H, W, D, 4)

    train_dataset = TensorDataset(train_input, train_u_norm)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Data loaded. Training with {ntrain} samples.")

    # 2. 初始化模型、优化器和调度器
    if args.model_type == 'fno_rc':
        model = FNO_RC_3D(args.modes, args.modes, args.modes, args.width, in_channels=4, out_channels=args.T_out).to(device)
    elif args.model_type == 'fno':
        model = FNO3d(args.modes, args.modes, args.modes, args.width, in_channels=4).to(device)
        # 动态添加 fc1 和 fc2 以匹配输出维度
        model.fc1 = nn.Linear(args.width, 128)
        model.fc2 = nn.Linear(128, args.T_out)
    else:
        raise ValueError("Invalid model_type specified. Choose 'fno' or 'fno_rc'.")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_func = LpLoss(size_average=True)
    
    print(f"Model initialized: {args.model_type.upper()}. Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. 训练循环
    print("\nStarting training loop...")
    start_time = time.time()
    for ep in range(args.epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            
            # 确保输出和目标的形状匹配 (B, H, W, T_out)
            loss = loss_func(out, y)
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{args.epochs} | Loss: {avg_epoch_loss:.6f}")

    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time/60:.2f} minutes.")

    # 4. 保存模型
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

def get_grid(shape, device):
    batchsize, size_x, size_y, size_z = 1, shape[1], shape[2], shape[3] # Grid is batch-independent
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified 3D FNO Model Trainer.')
    
    # ... (参数部分和之前完全一样, 这里省略)
    parser.add_argument('--model_type', type=str, required=True, choices=['fno', 'fno_rc'],
                        help="Type of model to train: 'fno' or 'fno_rc'.")
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat',
                        help='Path to the .mat file for 3D Navier-Stokes data.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the trained model weights.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.') # Increased epochs for better convergence
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.') # Smaller batch size for larger models
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer.')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for LR scheduler.') # Adjusted scheduler step
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for LR scheduler.')
    parser.add_argument('--ntrain', type=int, default=40, help='Number of training samples.')
    parser.add_argument('--ntest', type=int, default=10, help='Number of testing samples.')
    parser.add_argument('--T_in', type=int, default=10, help='Input time steps.')
    parser.add_argument('--T_out', type=int, default=20, help='Output time steps.')
    parser.add_argument('--modes', type=int, default=8, help='Fourier modes.')
    parser.add_argument('--width', type=int, default=32, help='Width of the FNO layers.') # Using the new default width
    
    args = parser.parse_args()
    train_model(args)
