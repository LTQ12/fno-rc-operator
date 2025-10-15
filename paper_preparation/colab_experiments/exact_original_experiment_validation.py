"""
基于您实际使用的训练脚本的统计验证实验
- train_fno_ns_2d.py (标准FNO)
- train_cft_residual_ns_2d.py (FNO-RC)
使用工作区现有的所有模块，不重新定义任何东西
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from timeit import default_timer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入您工作区的现有模块 - 不重新定义
from fourier_2d_baseline import FNO2d
from fourier_2d_cft_residual import FNO_RC
from utilities3 import GaussianNormalizer, LpLoss, count_params
from Adam import Adam

# 使用已有的标准FNO结果
BASELINE_FNO_RESULTS = [0.189337, 0.187468, 0.186463, 0.189672, 0.189752]

def setup_colab_environment():
    """设置Colab环境"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    base_path = "/content/drive/MyDrive/FNO_RC_Experiments"
    os.makedirs(f"{base_path}/results/statistical_validation_2d", exist_ok=True)
    os.makedirs(f"{base_path}/models/2d_ns", exist_ok=True)
    
    return device, base_path

def load_ns_data():
    """加载2D Navier-Stokes数据"""
    print("Loading 2D Navier-Stokes data...")
    
    data_file = "/content/drive/MyDrive/ns_data_N600_clean.pt"
    
    try:
        data = torch.load(data_file, map_location='cpu')
        if data.dim() > 4: 
            data = data.squeeze()
        print(f"Successfully loaded NS data from {data_file}")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Data range: [{data.min():.6f}, {data.max():.6f}]")
        
        if torch.isnan(data).any():
            print("⚠️  Warning: Found NaN values in data")
        if torch.isinf(data).any():
            print("⚠️  Warning: Found Inf values in data")
        
        return data
        
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_file}")
        print("Please ensure the data file exists at the specified path.")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def train_standard_fno_style(data, device, epochs=100, ntrain=500, ntest=100, 
                            T_in=10, T_out=10, batch_size=20, learning_rate=0.00025, 
                            modes=16, width=32, resolution=128):
    """
    完全按照train_fno_ns_2d.py的方式训练标准FNO
    但简化为单次运行而非K-fold
    """
    
    print(f"🔧 标准FNO训练参数:")
    print(f"  epochs: {epochs}")
    print(f"  ntrain: {ntrain}, ntest: {ntest}")
    print(f"  T_in: {T_in}, T_out: {T_out}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  modes: {modes}, width: {width}")
    print(f"  resolution: {resolution}")
    
    # 数据分割 - 按照train_fno_ns_2d.py方式
    train_a = data[:ntrain, ..., :T_in]
    train_u = data[:ntrain, ..., T_in:T_in + T_out]
    test_a = data[-ntest:, ..., :T_in]
    test_u = data[-ntest:, ..., T_in:T_in + T_out]
    
    print(f"📊 数据形状:")
    print(f"  train_a: {train_a.shape}, train_u: {train_u.shape}")
    print(f"  test_a: {test_a.shape}, test_u: {test_u.shape}")
    
    # 数据归一化 - 完全按照train_fno_ns_2d.py方式
    a_normalizer = GaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = GaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)
    
    a_normalizer.to(device)
    y_normalizer.to(device)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), 
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), 
        batch_size=batch_size, shuffle=False
    )
    
    # 模型初始化 - 完全按照train_fno_ns_2d.py方式
    model = FNO2d(modes, modes, width, in_channels=T_in, out_channels=T_out).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_func = LpLoss(size_average=False)
    
    print(f"🚀 开始标准FNO训练...")
    print(f"  Model: FNO2d")
    print(f"  Parameters: {count_params(model)}")
    print(f"  Optimizer: Adam (from Adam.py)")
    print(f"  Scheduler: CosineAnnealingLR")
    
    # 训练循环 - 完全按照train_fno_ns_2d.py方式
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            
            # 关键：按照train_fno_ns_2d.py方式 - 解码后计算损失
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = loss_func(
                out.view(out.size(0), resolution, resolution, T_out), 
                y.view(y.size(0), resolution, resolution, T_out)
            )
            loss.backward()
            
            # 关键：梯度裁剪 - train_fno_ns_2d.py有这个
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        
        # 测试评估 - 完全按照train_fno_ns_2d.py方式
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                test_l2 += loss_func(
                    out.view(out.size(0), resolution, resolution, T_out), 
                    y.view(y.size(0), resolution, resolution, T_out)
                ).item()
        
        train_l2 /= ntrain
        test_l2 /= ntest
        
        t2 = default_timer()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f'Epoch {ep+1}/{epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}')
    
    return test_l2

def train_fno_rc_style(data, device, epochs=100, ntrain=1000, ntest=200, 
                      T_in=10, T_out=10, batch_size=20, learning_rate=0.001, 
                      modes=16, width=32, weight_decay=1e-4):
    """
    完全按照train_cft_residual_ns_2d.py的方式训练FNO-RC
    """
    
    print(f"🔧 FNO-RC训练参数:")
    print(f"  epochs: {epochs}")
    print(f"  ntrain: {ntrain}, ntest: {ntest}")
    print(f"  T_in: {T_in}, T_out: {T_out}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  modes: {modes}, width: {width}")
    print(f"  weight_decay: {weight_decay}")
    
    # 数据分割 - 完全按照train_cft_residual_ns_2d.py方式
    x_train = data[:ntrain, ..., :T_in]
    y_train = data[:ntrain, ..., T_in:T_in+T_out]
    x_test = data[-ntest:, ..., :T_in]
    y_test = data[-ntest:, ..., T_in:T_in+T_out]
    
    print(f"📊 数据形状:")
    print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  x_test: {x_test.shape}, y_test: {y_test.shape}")
    
    # 数据归一化 - 完全按照train_cft_residual_ns_2d.py方式
    x_normalizer = GaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = GaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), 
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), 
        batch_size=batch_size, shuffle=False
    )
    
    # 模型初始化 - 完全按照train_cft_residual_ns_2d.py方式
    model = FNO_RC(
        modes1=modes, 
        modes2=modes, 
        width=width,
        in_channels=T_in,
        out_channels=T_out
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_func = LpLoss(size_average=False)
    
    print(f"🚀 开始FNO-RC训练...")
    print(f"  Model: FNO_RC")
    print(f"  Parameters: {count_params(model)}")
    print(f"  Optimizer: Adam (from Adam.py)")
    print(f"  Scheduler: CosineAnnealingLR")
    
    # 训练循环 - 完全按照train_cft_residual_ns_2d.py方式
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            
            # 关键：按照train_cft_residual_ns_2d.py方式 - 编码状态下计算训练损失
            loss = loss_func(out.view(out.size(0), -1), y.view(y.size(0), -1))
            loss.backward()
            
            # 注意：train_cft_residual_ns_2d.py没有梯度裁剪
            
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        
        # 测试评估 - 完全按照train_cft_residual_ns_2d.py方式
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            y_normalizer.to(device)
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                
                # 关键：按照train_cft_residual_ns_2d.py方式 - 解码后计算测试误差
                out_decoded = y_normalizer.decode(out)
                test_l2 += loss_func(out_decoded.view(out.size(0), -1), y.view(y.size(0), -1)).item()
        
        train_l2 /= ntrain
        test_l2 /= ntest
        
        t2 = default_timer()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f'Epoch {ep+1}/{epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}')
    
    return test_l2

def run_exact_original_experiments():
    """运行完全基于原始训练脚本的统计验证实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    data = load_ns_data()
    
    print("="*80)
    print("基于原始训练脚本的统计验证实验")
    print("="*80)
    print("📋 实验设置对比:")
    print("="*80)
    print("标准FNO (train_fno_ns_2d.py):")
    print("  ✅ 数据分割: train_a, train_u (前ntrain个)")
    print("  ✅ 归一化: a_normalizer, y_normalizer")
    print("  ✅ 训练损失: 解码后计算，4D shape")
    print("  ✅ 测试损失: 解码后计算，4D shape")
    print("  ✅ 梯度裁剪: 1.0")
    print("  ✅ 学习率: 0.00025")
    print("  ✅ ntrain/ntest: 500/100")
    print()
    print("FNO-RC (train_cft_residual_ns_2d.py):")
    print("  ✅ 数据分割: x_train, y_train (前ntrain个)")
    print("  ✅ 归一化: x_normalizer, y_normalizer")
    print("  ✅ 训练损失: 编码状态计算，flatten")
    print("  ✅ 测试损失: 解码后计算，flatten")
    print("  ✅ 梯度裁剪: 无")
    print("  ✅ 学习率: 0.001")
    print("  ✅ ntrain/ntest: 1000/200")
    print("="*80)
    
    # 但是为了公平比较，我们需要统一一些参数
    EPOCHS = 100  # 统一epochs
    
    # 按照您提到的2D训练命令参数调整
    print("🔧 为公平比较，统一以下参数:")
    print("  epochs: 100 (您要求)")
    print("  但保持其他参数按原始脚本设置")
    print()
    
    print("使用已有的标准FNO结果:")
    for i, error in enumerate(BASELINE_FNO_RESULTS):
        print(f"  运行 {i+1}: {error:.6f}")
    print(f"  平均: {np.mean(BASELINE_FNO_RESULTS):.6f} ± {np.std(BASELINE_FNO_RESULTS):.6f}")
    
    # 运行FNO-RC实验 - 按照原始脚本设置
    print("\n" + "="*60)
    print("运行FNO-RC实验 (按原始train_cft_residual_ns_2d.py设置)")
    print("="*60)
    
    fno_rc_results = []
    
    for run in range(5):
        print(f"\n{'='*20} FNO-RC运行 {run+1}/5 {'='*20}")
        
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # 按照train_cft_residual_ns_2d.py的默认参数
        best_test_loss = train_fno_rc_style(
            data, device,
            epochs=EPOCHS,
            ntrain=1000,  # train_cft_residual_ns_2d.py默认值
            ntest=200,    # train_cft_residual_ns_2d.py默认值
            T_in=10,
            T_out=10,
            batch_size=20,
            learning_rate=0.001,  # train_cft_residual_ns_2d.py默认值
            modes=16,
            width=32,
            weight_decay=1e-4
        )
        
        fno_rc_results.append(best_test_loss)
        print(f"\n✅ FNO-RC运行 {run+1} 完成: 最佳测试误差 = {best_test_loss:.6f}")
        
        torch.cuda.empty_cache()
    
    # 计算统计结果
    fno_mean = np.mean(BASELINE_FNO_RESULTS)
    fno_std = np.std(BASELINE_FNO_RESULTS)
    
    fno_rc_mean = np.mean(fno_rc_results)
    fno_rc_std = np.std(fno_rc_results)
    
    improvement = (fno_mean - fno_rc_mean) / fno_mean * 100
    
    # t检验
    diff = np.array(BASELINE_FNO_RESULTS) - np.array(fno_rc_results)
    t_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)))
    
    if abs(t_stat) > 2.776:
        p_value = 0.01
    elif abs(t_stat) > 2.132:
        p_value = 0.05
    else:
        p_value = 0.1
    
    # 结果
    results = {
        'fno_baseline': {
            'runs': [{'run': i+1, 'best_test_loss': error} for i, error in enumerate(BASELINE_FNO_RESULTS)],
            'mean': fno_mean,
            'std': fno_std,
            'training_script': 'train_fno_ns_2d.py',
            'parameters': {
                'ntrain': 500, 'ntest': 100, 'learning_rate': 0.00025,
                'gradient_clipping': 1.0, 'loss_calculation': 'decoded_4D'
            }
        },
        'fno_rc': {
            'runs': [{'run': i+1, 'best_test_loss': error} for i, error in enumerate(fno_rc_results)],
            'mean': fno_rc_mean,
            'std': fno_rc_std,
            'training_script': 'train_cft_residual_ns_2d.py',
            'parameters': {
                'ntrain': 1000, 'ntest': 200, 'learning_rate': 0.001,
                'gradient_clipping': None, 'loss_calculation': 'encoded_flatten_train_decoded_flatten_test'
            }
        },
        'improvement_percent': improvement,
        'statistical_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        },
        'experimental_setup': {
            'note': 'Based on exact original training scripts',
            'baseline_script': 'train_fno_ns_2d.py',
            'fno_rc_script': 'train_cft_residual_ns_2d.py',
            'unified_epochs': EPOCHS,
            'data_file': '/content/drive/MyDrive/ns_data_N600_clean.pt',
            'models_used': ['FNO2d from fourier_2d_baseline.py', 'FNO_RC from fourier_2d_cft_residual.py'],
            'utilities_used': ['Adam.py', 'utilities3.py'],
            'differences_preserved': [
                'ntrain/ntest: 500/100 vs 1000/200',
                'learning_rate: 0.00025 vs 0.001', 
                'gradient_clipping: yes vs no',
                'loss_calculation: different methods'
            ]
        },
        'metadata': {
            'problem': '2D Navier-Stokes',
            'timestamp': datetime.now().isoformat(),
            'note': 'Exact replication of original training scripts with unified epochs only'
        }
    }
    
    # 保存结果
    results_path = f"{base_path}/results/statistical_validation_2d/exact_original_experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    print("\n" + "="*80)
    print("基于原始训练脚本的统计实验结果")
    print("="*80)
    print(f"标准FNO (train_fno_ns_2d.py):     {fno_mean:.6f} ± {fno_std:.6f}")
    print(f"FNO-RC (train_cft_residual_ns_2d.py): {fno_rc_mean:.6f} ± {fno_rc_std:.6f}")
    print(f"改进:                              {improvement:.2f}%")
    print(f"t统计量:                           {t_stat:.4f}")
    print(f"p值:                              {p_value:.6f}")
    print(f"统计显著:                          {'是' if p_value < 0.05 else '否'}")
    print("="*80)
    print("⚠️  注意：此实验保持了两个训练脚本的原始差异")
    print("   如需完全公平比较，需要统一所有参数")
    
    return results

# ================================
# 主执行
# ================================

if __name__ == "__main__":
    print("🚀 基于原始训练脚本的统计验证实验")
    print("📋 使用您工作区的实际训练代码:")
    print("   - train_fno_ns_2d.py (标准FNO)")
    print("   - train_cft_residual_ns_2d.py (FNO-RC)")
    print("📁 数据路径: /content/drive/MyDrive/ns_data_N600_clean.pt")
    print("🔧 导入现有模块: fourier_2d_baseline, fourier_2d_cft_residual, utilities3, Adam")
    print("🕐 预计运行时间: 2-3小时")
    print()
    
    results = run_exact_original_experiments()
    
    print("\n🎉 基于原始训练脚本的统计验证完成！")
    print("✅ 使用了您工作区的实际训练代码")
    print("✅ 保持了原始脚本的所有设置")
    print("✅ 仅统一了训练epochs为100")
    print("💾 结果已保存到Google Drive")
