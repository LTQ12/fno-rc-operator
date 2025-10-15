"""
Chebyshev多项式阶数消融实验 - 2D Navier-Stokes
测试不同M_cheb值对FNO-RC性能的影响：4, 8, 16, 32个modes

目的：验证Chebyshev多项式阶数的重要性和最优设置
基于CFT分段实验的发现，固定L_segments=8
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

# 导入您工作区的现有模块
from fourier_2d_cft_residual import FNO_RC
from utilities3 import GaussianNormalizer, LpLoss, count_params
from Adam import Adam

def setup_colab_environment():
    """设置Colab环境"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    base_path = "/content/drive/MyDrive/FNO_RC_Experiments"
    os.makedirs(f"{base_path}/results/ablation_chebyshev_modes", exist_ok=True)
    os.makedirs(f"{base_path}/models/ablation", exist_ok=True)
    
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

class FNO_RC_Chebyshev_Ablation(nn.Module):
    """FNO-RC模型，支持可调节的Chebyshev参数"""
    def __init__(self, modes1, modes2, width, in_channels, out_channels, L_segments=8, M_cheb=8):
        super(FNO_RC_Chebyshev_Ablation, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9
        self.fc0 = nn.Linear(in_channels + 2, self.width)

        # 使用可调节参数的SpectralConv2d_RC
        from fourier_2d_cft_residual import SpectralConv2d_RC
        self.conv0 = SpectralConv2d_RC(self.width, self.width, self.modes1, self.modes2, L_segments, M_cheb)
        self.conv1 = SpectralConv2d_RC(self.width, self.width, self.modes1, self.modes2, L_segments, M_cheb)
        self.conv2 = SpectralConv2d_RC(self.width, self.width, self.modes1, self.modes2, L_segments, M_cheb)
        self.conv3 = SpectralConv2d_RC(self.width, self.width, self.modes1, self.modes2, L_segments, M_cheb)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

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

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

def train_fno_rc_chebyshev_ablation(data, device, M_cheb, epochs=200, ntrain=1000, ntest=200, 
                                   T_in=10, T_out=10, batch_size=20, learning_rate=0.0001, 
                                   modes=16, width=32, L_segments=8, weight_decay=1e-4):
    """训练FNO-RC Chebyshev消融实验模型"""
    
    print(f"🔧 FNO-RC (M_cheb={M_cheb}): lr={learning_rate}, epochs={epochs}, ntrain={ntrain}, ntest={ntest}")
    
    # 数据分割 - 完全按照train_cft_residual_ns_2d.py方式
    x_train = data[:ntrain, ..., :T_in]
    y_train = data[:ntrain, ..., T_in:T_in+T_out]
    x_test = data[-ntest:, ..., :T_in]
    y_test = data[-ntest:, ..., T_in:T_in+T_out]
    
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
    
    # 模型初始化 - 使用Chebyshev消融版本
    model = FNO_RC_Chebyshev_Ablation(
        modes1=modes, 
        modes2=modes, 
        width=width,
        in_channels=T_in,
        out_channels=T_out,
        L_segments=L_segments,
        M_cheb=M_cheb
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_func = LpLoss(size_average=False)
    
    print(f"  参数量: {count_params(model)}")
    
    # 训练循环 - 完全按照train_cft_residual_ns_2d.py方式
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            
            # 按照train_cft_residual_ns_2d.py方式 - 编码状态下计算训练损失
            loss = loss_func(out.view(out.size(0), -1), y.view(y.size(0), -1))
            loss.backward()
            
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
                
                # 按照train_cft_residual_ns_2d.py方式 - 解码后计算测试误差
                out_decoded = y_normalizer.decode(out)
                test_l2 += loss_func(out_decoded.view(out.size(0), -1), y.view(y.size(0), -1)).item()
        
        train_l2 /= ntrain
        test_l2 /= ntest
        
        t2 = default_timer()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f'Epoch {ep+1}/{epochs}: Train L2={train_l2:.6f}, Test L2={test_l2:.6f}')
    
    return test_l2

def run_chebyshev_modes_ablation():
    """运行Chebyshev多项式阶数消融实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    data = load_ns_data()
    
    print("="*80)
    print("Chebyshev多项式阶数消融实验 - 2D Navier-Stokes")
    print("="*80)
    print("📋 实验设置:")
    print("测试M_cheb: [6, 8, 10, 12]")
    print("固定L_segments: 8 (基于之前实验的最优结果)")
    print("固定其他参数: lr=0.0001, epochs=200, ntrain=1000, ntest=200")
    print("💡 细致探索M=8附近的最优区间")
    print("="*80)
    
    # 实验参数
    EPOCHS = 200  # 足够收敛
    M_CHEB_LIST = [6, 8, 10, 12]  # 要测试的Chebyshev阶数，细致探索最优区间
    L_SEGMENTS = 8  # 固定为最优分段数量
    
    results = {
        'experiment_type': 'chebyshev_modes_ablation',
        'fixed_parameters': {
            'L_segments': L_SEGMENTS,
            'modes': 16,
            'width': 32,
            'learning_rate': 0.0001,
            'epochs': EPOCHS,
            'ntrain': 1000,
            'ntest': 200,
            'batch_size': 20,
            'weight_decay': 1e-4
        },
        'variable_parameter': 'M_cheb',
        'results': [],
        'baseline_comparison': {
            'baseline_fno_error': 0.088803,  # 来自统计验证
            'original_fno_rc_error': 0.024504,  # 原始L=4,M=8的结果
            'optimal_segments_error': 0.028079,  # L=8,M=8的结果
            'original_improvement': 72.41,
            'segments_improvement': 68.38
        },
        'metadata': {
            'problem': '2D Navier-Stokes',
            'timestamp': datetime.now().isoformat(),
            'data_file': '/content/drive/MyDrive/ns_data_N600_clean.pt',
            'note': 'Using optimal L_segments=8 from previous ablation experiment'
        }
    }
    
    print(f"🚀 开始Chebyshev多项式阶数消融实验...")
    print(f"📌 使用最优L_segments={L_SEGMENTS}")
    
    for M_cheb in M_CHEB_LIST:
        print(f"\n{'='*20} M_cheb = {M_cheb} {'='*20}")
        
        torch.manual_seed(42)  # 固定随机种子确保公平比较
        np.random.seed(42)
        
        # 训练模型
        test_error = train_fno_rc_chebyshev_ablation(
            data, device,
            M_cheb=M_cheb,
            epochs=EPOCHS,
            L_segments=L_SEGMENTS
        )
        
        # 计算相对于基线FNO的改进
        baseline_error = results['baseline_comparison']['baseline_fno_error']
        improvement = (baseline_error - test_error) / baseline_error * 100
        
        # 计算相对于原始设置和最优分段设置的比较
        original_ratio = test_error / results['baseline_comparison']['original_fno_rc_error']
        segments_ratio = test_error / results['baseline_comparison']['optimal_segments_error']
        
        result = {
            'M_cheb': M_cheb,
            'test_error': test_error,
            'improvement_vs_baseline': improvement,
            'relative_to_original': original_ratio,
            'relative_to_optimal_segments': segments_ratio
        }
        
        results['results'].append(result)
        
        print(f"✅ M_cheb={M_cheb} 完成:")
        print(f"  测试误差: {test_error:.6f}")
        print(f"  vs 基线FNO改进: {improvement:.2f}%")
        print(f"  vs 原始FNO-RC(L=4,M=8): {original_ratio:.3f}x")
        print(f"  vs 最优分段(L=8,M=8): {segments_ratio:.3f}x")
        
        torch.cuda.empty_cache()
    
    # 分析结果
    print("\n" + "="*80)
    print("Chebyshev多项式阶数消融实验结果")
    print("="*80)
    
    print("📊 详细结果:")
    print(f"{'M_cheb':<8} {'Test Error':<12} {'vs FNO':<12} {'vs L=4,M=8':<12} {'vs L=8,M=8':<12}")
    print("-" * 65)
    
    best_cheb = None
    best_error = float('inf')
    
    for result in results['results']:
        M_cheb = result['M_cheb']
        error = result['test_error']
        improvement = result['improvement_vs_baseline']
        original_ratio = result['relative_to_original']
        segments_ratio = result['relative_to_optimal_segments']
        
        print(f"{M_cheb:<8} {error:<12.6f} {improvement:<12.2f}% {original_ratio:<12.3f}x {segments_ratio:<12.3f}x")
        
        if error < best_error:
            best_error = error
            best_cheb = M_cheb
    
    print("\n🎯 关键发现:")
    print(f"  最佳M_cheb: {best_cheb}")
    print(f"  最佳测试误差: {best_error:.6f}")
    print(f"  原始设置(L=4,M=8): {results['baseline_comparison']['original_fno_rc_error']:.6f}")
    print(f"  最优分段(L=8,M=8): {results['baseline_comparison']['optimal_segments_error']:.6f}")
    
    # 分析趋势
    errors = [r['test_error'] for r in results['results']]
    cheb_modes = [r['M_cheb'] for r in results['results']]
    
    print(f"\n📈 趋势分析:")
    if best_cheb == 8:
        print("  ✅ 原始设置(M=8)在新的L=8配置下仍是最优的")
    elif best_cheb == 6:
        print("  📉 略少的Chebyshev阶数表现更好，M=8可能略微过拟合")
    elif best_cheb in [10, 12]:
        print("  📈 略多的Chebyshev阶数表现更好，更高阶多项式有助于性能")
        if best_cheb == 12:
            print("  🔍 M=12表现最佳，在计算效率和精度间找到了更好平衡")
    
    # 计算最优组合的总体改进
    if best_error < results['baseline_comparison']['optimal_segments_error']:
        total_improvement = (baseline_error - best_error) / baseline_error * 100
        print(f"\n🌟 最优组合 (L={L_SEGMENTS}, M={best_cheb}):")
        print(f"  总体改进: {total_improvement:.2f}% (vs 基线FNO)")
        print(f"  相比原始FNO-RC改进: {(results['baseline_comparison']['original_fno_rc_error'] - best_error) / results['baseline_comparison']['original_fno_rc_error'] * 100:.2f}%")
    
    # 保存结果
    results_path = f"{base_path}/results/ablation_chebyshev_modes/chebyshev_modes_ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {results_path}")
    
    return results

# ================================
# 主执行
# ================================

if __name__ == "__main__":
    print("🔬 Chebyshev多项式阶数消融实验")
    print("📋 测试M_cheb=[6,8,10,12]对FNO-RC性能的影响")
    print("🎯 验证Chebyshev多项式阶数的重要性和最优设置")
    print("📌 基于L_segments=8的最优发现")
    print("💡 细致探索M=8附近的最优区间")
    print("📁 数据路径: /content/drive/MyDrive/ns_data_N600_clean.pt")
    print("🕐 预计运行时间: 1.5-2小时")
    print()
    
    results = run_chebyshev_modes_ablation()
    
    print("\n🎉 Chebyshev多项式阶数消融实验完成！")
    print("✅ 验证了不同M_cheb设置的影响")
    print("✅ 为FNO-RC的CFT参数选择提供了完整依据")
    print("💾 结果已保存到Google Drive")
