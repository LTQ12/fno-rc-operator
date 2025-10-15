#!/usr/bin/env python3
"""
测试修复架构的3D对比实验
关键修复：
1. CFT使用更多模态 (4x4x4 而不是 2x2x2)
2. 空间感知的残差修正 (而不是全局标量)
3. 学习的残差权重组合
4. 更鲁棒的CFT实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
import os
from datetime import datetime
from timeit import default_timer

# 导入修复的模型
from fourier_3d_clean import FNO3d
from fourier_3d_cft_residual_fixed import FNO_RC_3D_Fixed
from utilities3 import LpLoss, count_params, UnitGaussianNormalizer
from Adam import Adam

################################################################
# 数据加载 - 复用之前的函数
################################################################
def load_3d_data_efficient(data_path, ntrain=800, ntest=100, T_in=10, T_out=20):
    """内存高效的数据加载"""
    print(f"📁 正在加载数据: {data_path}")
    
    try:
        print("⏳ 正在读取.mat文件 (MATLAB v7.3格式)...")
        with h5py.File(data_path, 'r') as f:
            u_field = f['u'][:]  # [T, H, W, N] -> [50, 64, 64, 10000]
        
        print(f"✅ 原始数据形状: {u_field.shape}")
        
        # 转换为tensor并调整格式: [T, H, W, N] -> [N, H, W, T]
        print("🔄 正在转换数据格式...")
        u_field = torch.from_numpy(u_field).float()
        u_field = u_field.permute(3, 1, 2, 0)  # [10000, 64, 64, 50]
        print(f"✅ 转换后数据形状: {u_field.shape}")
        
        # 控制样本数量以节省内存
        total_samples = u_field.shape[0]
        ntrain_actual = min(total_samples - 150, ntrain)
        ntest_actual = min(total_samples - ntrain_actual, ntest)
        
        print(f"📊 总样本数: {total_samples}, 训练: {ntrain_actual}, 测试: {ntest_actual}")
        
        # 检查时间步
        if u_field.shape[-1] < T_in + T_out:
            print(f"❌ 时间步不足: 需要 {T_in + T_out}, 实际 {u_field.shape[-1]}")
            return None
        
        print("🔄 正在分割数据...")
        train_a = u_field[:ntrain_actual, ..., :T_in]
        train_u = u_field[:ntrain_actual, ..., T_in:T_in + T_out]
        
        test_a = u_field[-ntest_actual:, ..., :T_in]
        test_u = u_field[-ntest_actual:, ..., T_in:T_in + T_out]

        print(f"✅ 数据形状: train_a: {train_a.shape}, train_u: {train_u.shape}")
        print(f"✅ 最终样本数: ntrain={ntrain_actual}, ntest={ntest_actual}")
        
        # 释放原始数据
        del u_field
        import gc
        gc.collect()
        
        return train_a, train_u, test_a, test_u, ntrain_actual, ntest_actual
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def preprocess_3d_data_sequence(train_a, train_u, test_a, test_u, T_in, T_out, device):
    """完整序列预测的数据预处理"""
    
    S1, S2 = train_a.shape[1], train_a.shape[2]
    ntrain_actual, ntest_actual = train_a.shape[0], test_a.shape[0]
    
    # 标准化输入
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    # 标准化完整序列输出
    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u_normalized = y_normalizer.encode(train_u)
    y_normalizer.to(device)
    
    # 数据格式转换
    train_a_fno = train_a.reshape(ntrain_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    test_a_fno = test_a.reshape(ntest_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    
    train_a_rc = train_a.reshape(ntrain_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    test_a_rc = test_a.reshape(ntest_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    
    print(f"预处理完成（修复架构版本）:")
    print(f"  FNO输入: {train_a_fno.shape}, 目标: {train_u_normalized.shape}")
    print(f"  FNO_RC_Fixed输入: {train_a_rc.shape}, 目标: {train_u_normalized.shape}")
    
    return (train_a_fno, train_u_normalized, test_a_fno, test_u, 
            train_a_rc, test_a_rc, y_normalizer, S1, S2)

################################################################
# 训练函数
################################################################
def train_model_fixed(model, model_name, train_loader, test_loader, y_normalizer, device, ntrain_actual, ntest_actual, epochs=100):
    """修复架构的训练函数"""
    print(f"\n🔧 Training {model_name} (Fixed Architecture)...")
    print(f"Parameters: {count_params(model):,}")
    
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_func = LpLoss(size_average=False)
    
    train_losses = []
    test_losses = []
    
    # 早停机制
    best_test_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if model_name == 'FNO3d':
                # FNO3d: 添加网格坐标
                grid = create_grid_for_fno3d(x.shape, device)
                x_with_grid = torch.cat((x, grid), dim=-1)
                out = model(x_with_grid).squeeze(-1)
            elif model_name == 'FNO_RC_3D_Fixed':
                # 修复的FNO_RC_3D
                out = model(x).squeeze(-1)
                
                # 监控CFT路径的学习情况
                if ep % 20 == 0 and batch_idx == 0:
                    # 检查残差权重的学习情况
                    residual_weights = [layer.residual_weight.item() for layer in [model.conv0, model.conv1, model.conv2, model.conv3]]
                    print(f"  残差权重: {residual_weights}")
                    print(f"  输出范围: min={out.min().item():.6f}, max={out.max().item():.6f}, std={out.std().item():.6f}")
            
            # 完整序列损失计算
            loss_normalized = loss_func(out, y)
            loss_normalized.backward()
            optimizer.step()
            
            # 记录损失 - 使用真实尺度数据
            with torch.no_grad():
                out_decoded = y_normalizer.decode(out)
                y_decoded = y_normalizer.decode(y)
                loss_real_scale = loss_func(out_decoded, y_decoded)
                train_l2 += loss_real_scale.item()
        
        scheduler.step()
        
        # 测试
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                if model_name == 'FNO3d':
                    grid = create_grid_for_fno3d(x.shape, device)
                    x_with_grid = torch.cat((x, grid), dim=-1)
                    out = model(x_with_grid).squeeze(-1)
                elif model_name == 'FNO_RC_3D_Fixed':
                    out = model(x).squeeze(-1)
                
                # 完整序列测试损失
                out_decoded = y_normalizer.decode(out)
                y_decoded = y_normalizer.decode(y)
                test_l2 += loss_func(out_decoded, y_decoded).item()
        
        train_l2 /= ntrain_actual
        test_l2 /= ntest_actual
        
        train_losses.append(train_l2)
        test_losses.append(test_l2)
        
        # 早停检查
        if test_l2 < best_test_loss:
            best_test_loss = test_l2
            patience_counter = 0
        else:
            patience_counter += 1
            
        if ep % 10 == 0:
            print(f'Epoch {ep+1}/{epochs}: Train {train_l2:.6f}, Test {test_l2:.6f}')
        
        # 早停
        if patience_counter >= patience:
            print(f"🛑 早停: 测试损失在{patience}个epoch内没有改善")
            break
    
    return model, train_losses, test_losses

def create_grid_for_fno3d(shape, device):
    """为FNO3d创建网格坐标"""
    B, H, W, T_dim, _ = shape
    
    h_coords = torch.linspace(0, 1, H, device=device)
    w_coords = torch.linspace(0, 1, W, device=device)
    t_coords = torch.linspace(0, 1, T_dim, device=device)
    
    hh, ww, tt = torch.meshgrid(h_coords, w_coords, t_coords, indexing='ij')
    grid = torch.stack([hh, ww, tt], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
    
    return grid

################################################################
# 主函数
################################################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("🔧 设置Colab环境...")
    
    # 参数设置
    data_path = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
    ntrain, ntest = 600, 80        # 进一步减少样本数以节省内存
    T_in, T_out = 10, 20           # 保持完整序列预测
    modes = 8
    width = 20
    batch_size = 6                 # 进一步减小batch_size
    epochs = 100
    
    print("🚀 开始3D对比实验 - 修复架构版本")
    print(f"📋 实验参数: epochs={epochs}, batch_size={batch_size}")
    print(f"📊 数据参数: ntrain={ntrain}, ntest={ntest}, T_in={T_in}, T_out={T_out}")
    print("🎯 关键修复: CFT空间感知残差修正 + 学习权重组合")
    
    # 数据加载
    print("\n📁 步骤1: 数据加载")
    data = load_3d_data_efficient(data_path, ntrain, ntest, T_in, T_out)
    if data is None:
        return
    
    train_a, train_u, test_a, test_u, ntrain_actual, ntest_actual = data
    
    # 数据预处理
    print("\n🔄 步骤2: 数据预处理")
    processed_data = preprocess_3d_data_sequence(train_a, train_u, test_a, test_u, T_in, T_out, device)
    (train_a_fno, train_u, test_a_fno, test_u, 
     train_a_rc, test_a_rc, y_normalizer, S1, S2) = processed_data
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a_fno, train_u), 
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a_fno, test_u), 
        batch_size=batch_size, shuffle=False
    )
    
    # 模型定义 - 对比原始FNO和修复的FNO_RC
    models = {
        'FNO3d': FNO3d(modes, modes, modes, width, in_dim=13, out_dim=1),
        'FNO_RC_3D_Fixed': FNO_RC_3D_Fixed(modes, modes, modes, width, in_channels=T_in, out_channels=1),
    }
    
    print(f"📊 模型配置:")
    for name, model in models.items():
        print(f"  {name}: {count_params(model):,} 参数")
    
    # 训练所有模型
    print("\n🏋️ 步骤3: 开始训练模型")
    results = {}
    for model_name, model in models.items():
        trained_model, train_losses, test_losses = train_model_fixed(
            model, model_name, train_loader, test_loader, y_normalizer, device, ntrain_actual, ntest_actual, epochs
        )
        
        final_test_loss = test_losses[-1]
        results[model_name] = {
            'final_test_loss': final_test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'parameters': count_params(model)
        }
        print(f"✅ {model_name}: {final_test_loss:.6f}")
    
    # 结果对比
    print(f"\n🏆 3D实验结果 (修复架构 - {epochs} epochs):")
    print("-" * 60)
    
    # 按性能排序显示
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_loss'])
    
    for i, (name, result) in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"{rank} {name}: {result['final_test_loss']:.6f} ({result['parameters']:,} 参数)")
    
    # 计算改进百分比
    if len(sorted_results) >= 2:
        best_name, best_result = sorted_results[0]
        baseline_name, baseline_result = sorted_results[1]
        improvement = (baseline_result['final_test_loss'] - best_result['final_test_loss']) / baseline_result['final_test_loss'] * 100
        print(f"\n📈 {best_name} 相对于 {baseline_name} 改进: {improvement:.2f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = '/content/drive/MyDrive/FNO_RC_Experiments/'
    os.makedirs(results_dir, exist_ok=True)
    results_file = f'{results_dir}/fixed_architecture_3d_comparison_{timestamp}.json'
    
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
    print("🎉 修复架构3D对比实验完成！")
    print("\n📝 架构修复要点:")
    print("   ✅ CFT使用4×4×4模态 (而不是2×2×2)")
    print("   ✅ 空间感知的3D卷积残差修正")
    print("   ✅ 学习的残差权重组合")
    print("   ✅ 更鲁棒的CFT实现和错误处理")

if __name__ == "__main__":
    main()
