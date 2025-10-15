"""
CFT分段数量消融实验 - 2D Navier-Stokes
测试不同L_segments值对FNO-RC性能的影响：1, 2, 4, 8个segments

目的：验证CFT分段数量的重要性和最优设置
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
    os.makedirs(f"{base_path}/results/ablation_cft_segments", exist_ok=True)
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

class FNO_RC_Ablation(nn.Module):
    """FNO-RC模型，支持可调节的CFT参数"""
    def __init__(self, modes1, modes2, width, in_channels, out_channels, L_segments=4, M_cheb=8):
        super(FNO_RC_Ablation, self).__init__()
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

def train_fno_rc_ablation(data, device, L_segments, epochs=200, ntrain=1000, ntest=200, 
                         T_in=10, T_out=10, batch_size=20, learning_rate=0.0001, 
                         modes=16, width=32, M_cheb=8, weight_decay=1e-4):
    """训练FNO-RC消融实验模型"""
    
    print(f"🔧 FNO-RC (L_segments={L_segments}): lr={learning_rate}, epochs={epochs}, ntrain={ntrain}, ntest={ntest}")
    
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
    
    # 模型初始化 - 使用消融版本
    model = FNO_RC_Ablation(
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

def run_cft_segments_ablation():
    """运行CFT分段数量消融实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    data = load_ns_data()
    
    print("="*80)
    print("CFT分段数量消融实验 - 2D Navier-Stokes")
    print("="*80)
    print("📋 实验设置:")
    print("测试L_segments: [2, 4, 8, 16]")
    print("固定M_cheb: 8")
    print("固定其他参数: lr=0.0001, epochs=200, ntrain=1000, ntest=200")
    print("="*80)
    
    # 实验参数
    EPOCHS = 200  # 足够收敛
    L_SEGMENTS_LIST = [2, 4, 8, 16]  # 要测试的分段数量
    M_CHEB = 8  # 固定Chebyshev阶数
    
    results = {
        'experiment_type': 'cft_segments_ablation',
        'fixed_parameters': {
            'M_cheb': M_CHEB,
            'modes': 16,
            'width': 32,
            'learning_rate': 0.0001,
            'epochs': EPOCHS,
            'ntrain': 1000,
            'ntest': 200,
            'batch_size': 20,
            'weight_decay': 1e-4
        },
        'variable_parameter': 'L_segments',
        'results': [],
        'baseline_comparison': {
            'baseline_fno_error': 0.088803,  # 来自统计验证
            'original_fno_rc_error': 0.024504,  # L_segments=4的结果
            'original_improvement': 72.41
        },
        'metadata': {
            'problem': '2D Navier-Stokes',
            'timestamp': datetime.now().isoformat(),
            'data_file': '/content/drive/MyDrive/ns_data_N600_clean.pt'
        }
    }
    
    print(f"🚀 开始CFT分段数量消融实验...")
    
    for L_segments in L_SEGMENTS_LIST:
        print(f"\n{'='*20} L_segments = {L_segments} {'='*20}")
        
        torch.manual_seed(42)  # 固定随机种子确保公平比较
        np.random.seed(42)
        
        # 训练模型
        test_error = train_fno_rc_ablation(
            data, device,
            L_segments=L_segments,
            epochs=EPOCHS,
            M_cheb=M_CHEB
        )
        
        # 计算相对于基线FNO的改进
        baseline_error = results['baseline_comparison']['baseline_fno_error']
        improvement = (baseline_error - test_error) / baseline_error * 100
        
        result = {
            'L_segments': L_segments,
            'test_error': test_error,
            'improvement_vs_baseline': improvement,
            'relative_to_original': test_error / results['baseline_comparison']['original_fno_rc_error']
        }
        
        results['results'].append(result)
        
        print(f"✅ L_segments={L_segments} 完成:")
        print(f"  测试误差: {test_error:.6f}")
        print(f"  vs 基线FNO改进: {improvement:.2f}%")
        print(f"  vs 原始FNO-RC(L=4): {result['relative_to_original']:.3f}x")
        
        torch.cuda.empty_cache()
    
    # 分析结果
    print("\n" + "="*80)
    print("CFT分段数量消融实验结果")
    print("="*80)
    
    print("📊 详细结果:")
    print(f"{'L_segments':<12} {'Test Error':<12} {'vs FNO':<12} {'vs L=4':<12}")
    print("-" * 50)
    
    best_segments = None
    best_error = float('inf')
    
    for result in results['results']:
        L_seg = result['L_segments']
        error = result['test_error']
        improvement = result['improvement_vs_baseline']
        relative = result['relative_to_original']
        
        print(f"{L_seg:<12} {error:<12.6f} {improvement:<12.2f}% {relative:<12.3f}x")
        
        if error < best_error:
            best_error = error
            best_segments = L_seg
    
    print("\n🎯 关键发现:")
    print(f"  最佳L_segments: {best_segments}")
    print(f"  最佳测试误差: {best_error:.6f}")
    print(f"  原始设置(L=4): {[r for r in results['results'] if r['L_segments']==4][0]['test_error']:.6f}")
    
    # 分析趋势
    errors = [r['test_error'] for r in results['results']]
    segments = [r['L_segments'] for r in results['results']]
    
    print(f"\n📈 趋势分析:")
    if best_segments == 4:
        print("  ✅ 原始设置(L=4)是最优的")
    elif best_segments == 2:
        print("  📉 更少的分段数量表现更好，可能原始设置过于复杂")
    elif best_segments in [8, 16]:
        print("  📈 更多的分段数量表现更好，更精细分段有助于性能")
        if best_segments == 16:
            print("  🔍 L=16表现最佳，可能需要测试更大的分段数量")
    
    # 保存结果
    results_path = f"{base_path}/results/ablation_cft_segments/cft_segments_ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {results_path}")
    
    return results

# ================================
# 主执行
# ================================

if __name__ == "__main__":
    print("🔬 CFT分段数量消融实验")
    print("📋 测试L_segments=[2,4,8,16]对FNO-RC性能的影响")
    print("🎯 验证CFT分段策略的重要性和最优设置")
    print("📁 数据路径: /content/drive/MyDrive/ns_data_N600_clean.pt")
    print("🕐 预计运行时间: 1.5-2小时")
    print()
    
    results = run_cft_segments_ablation()
    
    print("\n🎉 CFT分段数量消融实验完成！")
    print("✅ 验证了不同L_segments设置的影响")
    print("✅ 为FNO-RC的CFT参数选择提供了依据")
    print("💾 结果已保存到Google Drive")
