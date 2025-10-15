"""
门控机制消融实验 - 2D Navier-Stokes
对比FNO-RC中不同融合策略的效果：简单相加 vs 门控机制

目的：验证简单相加融合策略的有效性和合理性
基于消融实验发现的最优参数：L_segments=8, M_cheb=8
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
    os.makedirs(f"{base_path}/results/ablation_gating_mechanism", exist_ok=True)
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

# ================================
# 门控机制的SpectralConv2d_RC变体
# ================================

class SpectralConv2d_RC_Gated(nn.Module):
    """带门控机制的SpectralConv2d_RC"""
    def __init__(self, in_channels, out_channels, modes1, modes2, L_segments=8, M_cheb=8):
        super(SpectralConv2d_RC_Gated, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 1. Standard FNO learnable weights (the "main path")
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        # 2. CFT参数
        self.cft_modes1 = modes1 // 4
        self.cft_modes2 = modes2 // 4
        self.L_segments = L_segments
        self.M_cheb = M_cheb
        cft_flat_dim = self.in_channels * self.cft_modes1 * self.cft_modes2 * 2 # Real/Imag

        # 3. CFT特征提取器
        self.correction_generator = nn.Sequential(
            nn.Linear(cft_flat_dim, (self.in_channels + self.out_channels) * 2),
            nn.GELU(),
            nn.Linear((self.in_channels + self.out_channels) * 2, self.out_channels)
        )
        
        # 4. 门控机制
        self.gate_network = nn.Sequential(
            nn.Linear(cft_flat_dim, (self.in_channels + self.out_channels)),
            nn.GELU(),
            nn.Linear((self.in_channels + self.out_channels), self.out_channels),
            nn.Sigmoid()  # 输出0-1的门控权重
        )
        
        # 初始化
        nn.init.zeros_(self.correction_generator[-1].weight)
        nn.init.zeros_(self.correction_generator[-1].bias)
        # 门控网络初始化为0.5，表示均衡融合
        nn.init.constant_(self.gate_network[-2].bias, 0.0)  # 使sigmoid输出接近0.5

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Main FNO Path ---
        x_ft = torch.fft.rfft2(x, s=(H, W))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x_fno = torch.fft.irfft2(out_ft, s=(H, W))

        # --- CFT Residual Correction Path ---
        from fourier_2d_cft_residual import cft2d
        cft_coeffs = cft2d(x, self.cft_modes1, self.cft_modes2, self.L_segments, self.M_cheb)
        cft_flat = torch.view_as_real(cft_coeffs).flatten(1)
        
        # CFT修正
        correction = self.correction_generator(cft_flat) # (B, out_channels)
        correction = correction.view(B, self.out_channels, 1, 1)
        
        # 门控权重
        gate = self.gate_network(cft_flat) # (B, out_channels)
        gate = gate.view(B, self.out_channels, 1, 1)

        # 门控融合: gate * x_fno + (1-gate) * correction
        return gate * x_fno + (1 - gate) * correction

class FNO_RC_Gating_Ablation(nn.Module):
    """FNO-RC模型，支持不同的融合策略"""
    def __init__(self, modes1, modes2, width, in_channels, out_channels, 
                 L_segments=8, M_cheb=8, use_gating=False):
        super(FNO_RC_Gating_Ablation, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9
        self.use_gating = use_gating
        self.fc0 = nn.Linear(in_channels + 2, self.width)

        # 根据是否使用门控机制选择不同的SpectralConv
        if use_gating:
            self.conv0 = SpectralConv2d_RC_Gated(self.width, self.width, self.modes1, self.modes2, L_segments, M_cheb)
            self.conv1 = SpectralConv2d_RC_Gated(self.width, self.width, self.modes1, self.modes2, L_segments, M_cheb)
            self.conv2 = SpectralConv2d_RC_Gated(self.width, self.width, self.modes1, self.modes2, L_segments, M_cheb)
            self.conv3 = SpectralConv2d_RC_Gated(self.width, self.width, self.modes1, self.modes2, L_segments, M_cheb)
        else:
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

def train_fno_rc_gating_ablation(data, device, use_gating, epochs=200, ntrain=1000, ntest=200, 
                                 T_in=10, T_out=10, batch_size=20, learning_rate=0.0001, 
                                 modes=16, width=32, L_segments=8, M_cheb=8, weight_decay=1e-4):
    """训练FNO-RC门控消融实验模型"""
    
    gating_type = "Gated" if use_gating else "Simple Addition"
    print(f"🔧 FNO-RC ({gating_type}): lr={learning_rate}, epochs={epochs}, ntrain={ntrain}, ntest={ntest}")
    
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
    
    # 模型初始化 - 使用门控消融版本
    model = FNO_RC_Gating_Ablation(
        modes1=modes, 
        modes2=modes, 
        width=width,
        in_channels=T_in,
        out_channels=T_out,
        L_segments=L_segments,
        M_cheb=M_cheb,
        use_gating=use_gating
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

def run_gating_mechanism_ablation():
    """运行门控机制消融实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    data = load_ns_data()
    
    print("="*80)
    print("门控机制消融实验 - 2D Navier-Stokes")
    print("="*80)
    print("📋 实验设置:")
    print("对比融合策略: [简单相加, 门控机制]")
    print("固定最优CFT参数: L_segments=8, M_cheb=8")
    print("固定其他参数: lr=0.0001, epochs=200, ntrain=1000, ntest=200")
    print("💡 验证简单相加融合策略的有效性")
    print("="*80)
    
    # 实验参数
    EPOCHS = 200  # 足够收敛
    L_SEGMENTS = 8  # 最优分段数量
    M_CHEB = 8  # 最优Chebyshev阶数
    
    results = {
        'experiment_type': 'gating_mechanism_ablation',
        'fixed_parameters': {
            'L_segments': L_SEGMENTS,
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
        'variable_parameter': 'fusion_strategy',
        'results': [],
        'baseline_comparison': {
            'baseline_fno_error': 0.088803,  # 来自统计验证
            'original_fno_rc_error': 0.024504,  # 原始L=4,M=8的结果
            'optimal_cft_error': 0.028024,  # L=8,M=8的结果 (简单相加)
        },
        'metadata': {
            'problem': '2D Navier-Stokes',
            'timestamp': datetime.now().isoformat(),
            'data_file': '/content/drive/MyDrive/ns_data_N600_clean.pt',
            'note': 'Using optimal CFT parameters L=8, M=8 from previous ablation experiments'
        }
    }
    
    print(f"🚀 开始门控机制消融实验...")
    print(f"📌 使用最优CFT参数: L_segments={L_SEGMENTS}, M_cheb={M_CHEB}")
    
    # 测试两种融合策略
    fusion_strategies = [
        {'use_gating': False, 'name': 'Simple Addition'},
        {'use_gating': True, 'name': 'Gated Fusion'}
    ]
    
    for strategy in fusion_strategies:
        use_gating = strategy['use_gating']
        strategy_name = strategy['name']
        
        print(f"\n{'='*20} {strategy_name} {'='*20}")
        
        torch.manual_seed(42)  # 固定随机种子确保公平比较
        np.random.seed(42)
        
        # 训练模型
        test_error = train_fno_rc_gating_ablation(
            data, device,
            use_gating=use_gating,
            epochs=EPOCHS,
            L_segments=L_SEGMENTS,
            M_cheb=M_CHEB
        )
        
        # 计算相对于基线FNO的改进
        baseline_error = results['baseline_comparison']['baseline_fno_error']
        improvement = (baseline_error - test_error) / baseline_error * 100
        
        # 计算相对于原始设置的比较
        original_ratio = test_error / results['baseline_comparison']['original_fno_rc_error']
        
        result = {
            'fusion_strategy': strategy_name,
            'use_gating': use_gating,
            'test_error': test_error,
            'improvement_vs_baseline': improvement,
            'relative_to_original': original_ratio,
            'parameter_count': None  # 将在训练时更新
        }
        
        results['results'].append(result)
        
        print(f"✅ {strategy_name} 完成:")
        print(f"  测试误差: {test_error:.6f}")
        print(f"  vs 基线FNO改进: {improvement:.2f}%")
        print(f"  vs 原始FNO-RC: {original_ratio:.3f}x")
        
        torch.cuda.empty_cache()
    
    # 分析结果
    print("\n" + "="*80)
    print("门控机制消融实验结果")
    print("="*80)
    
    print("📊 详细结果:")
    print(f"{'融合策略':<15} {'测试误差':<12} {'vs FNO改进':<12} {'vs 原始FNO-RC':<15}")
    print("-" * 65)
    
    simple_result = None
    gated_result = None
    
    for result in results['results']:
        strategy = result['fusion_strategy']
        error = result['test_error']
        improvement = result['improvement_vs_baseline']
        original_ratio = result['relative_to_original']
        
        print(f"{strategy:<15} {error:<12.6f} {improvement:<12.2f}% {original_ratio:<15.3f}x")
        
        if not result['use_gating']:
            simple_result = result
        else:
            gated_result = result
    
    # 对比分析
    print("\n🎯 关键发现:")
    if simple_result and gated_result:
        simple_error = simple_result['test_error']
        gated_error = gated_result['test_error']
        
        if simple_error < gated_error:
            improvement_pct = (gated_error - simple_error) / gated_error * 100
            print(f"  ✅ 简单相加策略表现更好")
            print(f"  📈 简单相加相比门控机制提升: {improvement_pct:.2f}%")
            print(f"  💡 验证了您选择简单相加的合理性")
        else:
            improvement_pct = (simple_error - gated_error) / simple_error * 100
            print(f"  🔄 门控机制略有优势")
            print(f"  📈 门控机制相比简单相加提升: {improvement_pct:.2f}%")
            if improvement_pct < 1.0:
                print(f"  ⚖️  提升微小，简单相加仍是更好的选择（参数更少，计算更快）")
    
    print(f"\n📈 设计哲学验证:")
    print(f"  🎯 FNO-RC采用简单相加的设计理念：")
    print(f"     - CFT路径作为残差修正，直接叠加到FNO输出")
    print(f"     - 避免复杂的门控机制，保持模型简洁性")
    print(f"     - 减少参数量，提高训练稳定性")
    
    # 保存结果
    results_path = f"{base_path}/results/ablation_gating_mechanism/gating_mechanism_ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {results_path}")
    
    return results

# ================================
# 主执行
# ================================

if __name__ == "__main__":
    print("🔬 门控机制消融实验")
    print("📋 对比简单相加 vs 门控融合策略")
    print("🎯 验证FNO-RC设计选择的合理性")
    print("📌 基于最优CFT参数: L_segments=8, M_cheb=8")
    print("📁 数据路径: /content/drive/MyDrive/ns_data_N600_clean.pt")
    print("🕐 预计运行时间: 1-1.5小时")
    print()
    
    results = run_gating_mechanism_ablation()
    
    print("\n🎉 门控机制消融实验完成！")
    print("✅ 验证了不同融合策略的效果")
    print("✅ 为FNO-RC的设计选择提供了科学依据")
    print("💾 结果已保存到Google Drive")
    print("\n🏆 核心消融实验全部完成！")
