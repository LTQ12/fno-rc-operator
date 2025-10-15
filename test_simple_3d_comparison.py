#!/usr/bin/env python3
"""
简化的3D对比实验 - 直接复制2D成功的思路
关键简化：
1. 去掉复杂的学习权重
2. 直接使用2D成功的全局标量修正（虽然不完美，但至少能工作）
3. 增加CFT模态数量
4. 简化架构，确保梯度传播
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

# 导入正确结构的模型
from fourier_3d_clean import FNO3d
from fourier_3d_cft_residual_correct import FNO_RC_3D_Correct
from utilities3 import LpLoss, count_params, UnitGaussianNormalizer, GaussianNormalizer
from Adam import Adam

################################################################
# 复用数据加载函数
################################################################
def load_3d_data_efficient(data_path, ntrain=600, ntest=80, T_in=10, T_out=20):
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
    
    # 🔧 关键修复：使用GaussianNormalizer而不是UnitGaussianNormalizer
    # 这保持了空间相关性，对CFT至关重要！
    a_normalizer = GaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    # 标准化完整序列输出
    y_normalizer = GaussianNormalizer(train_u)
    train_u_normalized = y_normalizer.encode(train_u)
    y_normalizer.to(device)
    
    # 数据格式转换
    train_a_fno = train_a.reshape(ntrain_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    test_a_fno = test_a.reshape(ntest_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    
    train_a_rc = train_a.reshape(ntrain_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    test_a_rc = test_a.reshape(ntest_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    
    print(f"预处理完成（简化架构版本）:")
    print(f"  FNO输入: {train_a_fno.shape}, 目标: {train_u_normalized.shape}")
    print(f"  FNO_RC_Simple输入: {train_a_rc.shape}, 目标: {train_u_normalized.shape}")
    
    return (train_a_fno, train_u_normalized, test_a_fno, test_u, 
            train_a_rc, test_a_rc, y_normalizer, S1, S2)

################################################################
# 简化的训练函数
################################################################
def train_model_simple(model, model_name, train_loader, test_loader, y_normalizer, device, ntrain_actual, ntest_actual, epochs=100):
    """简化的训练函数 - 重点监控CFT路径是否工作"""
    print(f"\n🔧 Training {model_name} (Simple Architecture)...")
    print(f"Parameters: {count_params(model):,}")
    
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 🔧 使用成功3D实验的学习率
    
    # 🔧 完全复制成功3D实验的调度器：StepLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
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
            elif model_name == 'FNO_RC_3D_Simple':
                # 简化的FNO_RC_3D - 完全复制成功实验的简洁方式
                out = model(x).squeeze(-1)
                
                # 🔧 增强监控：每5个epoch检查FNO-RC状态
                if ep % 5 == 0 and batch_idx == 0:
                    print(f"  🔍 Epoch {ep} 诊断:")
                    print(f"    输出范围: min={out.min().item():.6f}, max={out.max().item():.6f}, std={out.std().item():.6f}")
                    
                    # 检查是否出现异常值
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        print(f"    ❌ 检测到NaN或Inf！")
                    
                    if out.abs().max() > 100:
                        print(f"    ⚠️  输出值过大，可能发散！")
                    
                    # 🔧 CFT路径详细监控
                    with torch.no_grad():
                        try:
                            # 启用监控模式
                            for layer in [model.conv0, model.conv1, model.conv2, model.conv3]:
                                layer._monitor_cft = True
                            
                            # 运行一次前向传播以收集监控数据
                            x_test = x[:1]
                            _ = model(x_test)
                            
                            # 检查各层CFT路径的活跃度
                            cft_active_layers = 0
                            total_correction_magnitude = 0.0
                            
                            for i, layer in enumerate([model.conv0, model.conv1, model.conv2, model.conv3]):
                                if hasattr(layer, '_last_correction_magnitude'):
                                    correction_mag = layer._last_correction_magnitude
                                    cft_input_mag = layer._last_cft_input_magnitude
                                    
                                    if correction_mag > 1e-6:  # CFT路径有显著输出
                                        cft_active_layers += 1
                                        total_correction_magnitude += correction_mag
                                    
                                    if i == 0:  # 只打印第一层的详细信息
                                        print(f"    CFT路径: 输入幅度={cft_input_mag:.6f}, 修正幅度={correction_mag:.6f}")
                            
                            print(f"    CFT活跃层数: {cft_active_layers}/4, 总修正幅度: {total_correction_magnitude:.6f}")
                            
                            # 诊断CFT修正是否合理
                            if total_correction_magnitude > 10:
                                print(f"    ⚠️  CFT修正幅度过大，可能破坏训练！")
                            elif total_correction_magnitude < 0.001:
                                print(f"    ⚠️  CFT修正幅度过小，可能没有效果！")
                            else:
                                print(f"    ✅ CFT修正幅度适中")
                            
                            # 关闭监控模式
                            for layer in [model.conv0, model.conv1, model.conv2, model.conv3]:
                                delattr(layer, '_monitor_cft')
                                
                        except Exception as e:
                            print(f"    ❌ CFT监控失败: {e}")
            
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
                elif model_name == 'FNO_RC_3D_Simple':
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
    
    # 参数设置 - 使用成功实验的配置
    data_path = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
    ntrain, ntest = 1000, 200       # 🔧 恢复成功实验的样本数
    T_in, T_out = 10, 20            # 保持完整序列预测
    modes = 8
    width = 20
    batch_size = 10                 # 🔧 恢复成功实验的batch_size
    epochs = 100                    # 🔧 设置为100 epochs
    
    print("🚀 开始FNO-RC专项调试实验")
    print(f"📋 实验参数: epochs={epochs}, batch_size={batch_size}")
    print(f"📊 数据参数: ntrain={ntrain}, ntest={ntest}, T_in={T_in}, T_out={T_out}")
    print("🎯 调试策略: 屏蔽FNO训练，专注FNO-RC问题诊断")
    print("📈 FNO基准: 已知稳定在测试误差~0.44")
    print("🔧 FNO-RC配置: 最简单稳定版本")
    print("  - CFT模态比例: modes//4")
    print("  - 网络结构: 2层correction_generator") 
    print("  - 修正方式: 全局标量修正")
    print("  - 初始化: 零初始化")
    print("  - 标准化: GaussianNormalizer")
    print("⚡ 目标：让FNO-RC测试误差 < 0.44（即超越FNO）")
    
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
    
    # 🔧 专注调试FNO-RC：暂时屏蔽FNO训练
    models = {
        # 'FNO3d': FNO3d(modes, modes, modes, width, in_dim=13, out_dim=1),  # 暂时屏蔽，专注FNO-RC调试
        'FNO_RC_3D_Correct': FNO_RC_3D_Correct(modes, modes, modes, width, in_channels=T_in, out_channels=1),
    }

    print(f"📊 模型配置:")
    for name, model in models.items():
        print(f"  {name}: {count_params(model):,} 参数")

    # 训练所有模型
    print("\n🏋️ 步骤3: 开始训练模型")
    results = {}
    for model_name, model in models.items():
        trained_model, train_losses, test_losses = train_model_simple(
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

    # FNO-RC专项调试结果
    print(f"\n🏆 FNO-RC专项调试结果 (正确架构 - {epochs} epochs):")
    print("-" * 60)

    for name, result in results.items():
        print(f"🔧 {name}: {result['final_test_loss']:.6f} ({result['parameters']:,} 参数)")
        
        # 与已知FNO基准对比 (约0.44)
        fno_baseline = 0.44  # 已知的FNO3d基准性能
        if result['final_test_loss'] < fno_baseline:
            improvement = (fno_baseline - result['final_test_loss']) / fno_baseline * 100
            print(f"🎉 成功！FNO-RC ({result['final_test_loss']:.6f}) 优于FNO基准 ({fno_baseline:.6f})")
            print(f"📈 相对于FNO基准改进: {improvement:.2f}%")
            
            if improvement > 35:
                print(f"🏆 优秀！接近目标43.76%改进！")
            elif improvement > 20:
                print(f"🔥 良好进展！")
            elif improvement > 5:
                print(f"⚡ 有改进但需要进一步优化")
        else:
            degradation = (result['final_test_loss'] - fno_baseline) / fno_baseline * 100
            print(f"❌ FNO-RC ({result['final_test_loss']:.6f}) 劣于FNO基准 ({fno_baseline:.6f})")
            print(f"📉 性能下降: {degradation:.2f}%")
            print("🔧 需要进一步调试架构问题")
        
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = '/content/drive/MyDrive/FNO_RC_Experiments/'
    os.makedirs(results_dir, exist_ok=True)
    results_file = f'{results_dir}/correct_architecture_3d_comparison_{timestamp}.json'

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
    print("🎉 正确架构3D对比实验完成！")
    print("\n📝 正确架构要点:")
    print("   ✅ 基于fourier_3d_clean.py的成熟FNO结构")
    print("   ✅ 单一权重矩阵（不是4个）")
    print("   ✅ 完全复制2D成功的CFT残差逻辑")
    print("   ✅ 简化CFT计算（FFT近似）")
    print("   ✅ 零初始化确保训练稳定性")

if __name__ == "__main__":
    main()
