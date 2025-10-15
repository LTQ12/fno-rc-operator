"""
FNO-RC 消融实验
专为Google Colab环境设计
测试CFT segments、Chebyshev modes、门控机制的影响
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
import pickle
import h5py
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 导入之前定义的模型组件
from statistical_validation_experiments import (
    setup_colab_environment, load_burgers_data, prepare_data_loaders,
    LpLoss, SpectralConv1d, FNOLayer1d, StandardFNO1d
)

# ================================
# 改进的CFT层实现
# ================================

class CFTLayer1d(nn.Module):
    """改进的CFT层实现，支持不同参数配置"""
    def __init__(self, in_channels, out_channels, modes, segments=4):
        super().__init__()
        self.modes = modes
        self.segments = segments
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Chebyshev权重
        self.chebyshev_weights = nn.Parameter(
            torch.randn(segments, modes, in_channels, out_channels) * 0.02
        )
        
        # 保形映射参数
        self.conformal_params = nn.Parameter(torch.tensor([1.0, 0.5]))
        
    def chebyshev_transform(self, x, n_modes):
        """Chebyshev变换"""
        batch_size, channels, length = x.shape
        
        # 归一化到[-1,1]
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        
        # Chebyshev多项式展开
        coeffs = []
        T_prev_prev = torch.ones_like(x_norm)
        T_prev = x_norm
        
        for n in range(n_modes):
            if n == 0:
                T_n = torch.ones_like(x_norm)
            elif n == 1:
                T_n = x_norm
            else:
                T_n = 2 * x_norm * T_prev - T_prev_prev
                T_prev_prev = T_prev
                T_prev = T_n
                
            coeff = torch.mean(x_norm * T_n, dim=-1, keepdim=True)
            coeffs.append(coeff)
        
        return torch.cat(coeffs, dim=-1)
    
    def forward(self, x):
        batch_size, channels, length = x.shape
        
        # 分段处理
        segment_size = length // self.segments
        results = []
        
        for seg in range(self.segments):
            start_idx = seg * segment_size
            end_idx = (seg + 1) * segment_size if seg < self.segments - 1 else length
            x_segment = x[:, :, start_idx:end_idx]
            
            # Chebyshev变换
            coeffs = self.chebyshev_transform(x_segment, self.modes)
            
            # 应用学习权重
            coeffs_filtered = torch.einsum('bci,mico->bco', coeffs, self.chebyshev_weights[seg])
            
            # 重构信号
            reconstructed = torch.tanh(coeffs_filtered.mean(dim=-1, keepdim=True).expand(-1, -1, x_segment.shape[-1]))
            results.append(reconstructed)
        
        return torch.cat(results, dim=-1)

# ================================
# 不同配置的FNO-RC模型
# ================================

class ConfigurableFNORC1d(nn.Module):
    """可配置的1D FNO-RC模型"""
    def __init__(self, modes=16, width=64, num_layers=4, 
                 cft_segments=4, cft_modes=8, use_gating=True):
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.use_gating = use_gating
        
        # 输入嵌入
        self.fc0 = nn.Linear(2, self.width)
        
        # FNO主路径
        self.fno_layers = nn.ModuleList([FNOLayer1d(modes, width) for _ in range(num_layers)])
        
        # CFT残差路径
        self.cft_layers = nn.ModuleList([CFTLayer1d(width, width, cft_modes, cft_segments) for _ in range(num_layers)])
        
        # 门控机制（可选）
        if use_gating:
            self.gate_layers = nn.ModuleList([nn.Linear(2*width, width) for _ in range(num_layers)])
        else:
            self.gate_layers = None
        
        # 输出层
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # 激活函数
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x shape: (batch, seq_len, 2) -> (a(x), x)
        batch_size = x.shape[0]
        
        # 输入嵌入
        x = self.fc0(x)  # (batch, seq_len, width)
        x = x.permute(0, 2, 1)  # (batch, width, seq_len)
        
        # 逐层处理
        for i in range(self.num_layers):
            # FNO路径
            x_fno = self.fno_layers[i](x)
            x_fno = self.activation(x_fno)
            
            # CFT路径
            x_cft = self.cft_layers[i](x)
            
            if self.use_gating and self.gate_layers is not None:
                # 门控融合
                x_concat = torch.cat([x_fno, x_cft], dim=1)  # (batch, 2*width, seq_len)
                x_concat = x_concat.permute(0, 2, 1)  # (batch, seq_len, 2*width)
                
                gate = torch.sigmoid(self.gate_layers[i](x_concat))  # (batch, seq_len, width)
                gate = gate.permute(0, 2, 1)  # (batch, width, seq_len)
                
                # 残差连接
                x = x_fno + gate * x_cft
            else:
                # 直接相加（无门控）
                x = x_fno + 0.1 * x_cft  # 添加小权重避免训练不稳定
        
        # 输出层
        x = x.permute(0, 2, 1)  # (batch, seq_len, width)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)  # (batch, seq_len, 1)
        
        return x

# ================================
# 训练函数
# ================================

def train_model_quick(model, train_loader, test_loader, device, epochs=200, lr=0.001):
    """快速训练模型（用于消融实验）"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = LpLoss(size_average=True)
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 测试阶段
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += loss_fn(output, target).item()
            
            test_loss /= len(test_loader)
            best_test_loss = min(best_test_loss, test_loss)
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch:3d}: Test Loss = {test_loss:.6f}')
        
        scheduler.step()
    
    return best_test_loss

# ================================
# 消融实验主函数
# ================================

def run_ablation_experiments():
    """运行消融实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    train_a, train_u, test_a, test_u = load_burgers_data()
    train_loader, test_loader = prepare_data_loaders(train_a, train_u, test_a, test_u)
    
    results = {
        'baseline_fno': {},
        'cft_segments': {},
        'chebyshev_modes': {},
        'gating_ablation': {},
        'metadata': {
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
            'epochs_per_experiment': 200
        }
    }
    
    print("="*60)
    print("开始消融实验")
    print("="*60)
    
    # 1. 基线FNO
    print("\n1. 基线FNO实验")
    print("-" * 40)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = StandardFNO1d(modes=16, width=64, num_layers=4)
    best_loss = train_model_quick(model, train_loader, test_loader, device, epochs=200)
    results['baseline_fno'] = {
        'test_error': best_loss,
        'config': 'Standard FNO baseline'
    }
    print(f"基线FNO: {best_loss:.6f}")
    
    del model
    torch.cuda.empty_cache()
    
    # 2. CFT分段数量消融
    print("\n2. CFT分段数量消融实验")
    print("-" * 40)
    
    segments_to_test = [1, 2, 4, 8]
    
    for segments in segments_to_test:
        print(f"测试 {segments} 个segments...")
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = ConfigurableFNORC1d(
            modes=16, width=64, num_layers=4,
            cft_segments=segments, cft_modes=8, use_gating=True
        )
        
        best_loss = train_model_quick(model, train_loader, test_loader, device, epochs=200)
        
        results['cft_segments'][f'{segments}_segments'] = {
            'test_error': best_loss,
            'segments': segments,
            'modes': 8,
            'improvement_over_baseline': (results['baseline_fno']['test_error'] - best_loss) / results['baseline_fno']['test_error'] * 100
        }
        
        print(f"{segments} segments: {best_loss:.6f} (改进: {results['cft_segments'][f'{segments}_segments']['improvement_over_baseline']:.2f}%)")
        
        del model
        torch.cuda.empty_cache()
    
    # 3. Chebyshev模式数量消融
    print("\n3. Chebyshev模式数量消融实验")
    print("-" * 40)
    
    modes_to_test = [4, 8, 16]
    
    for modes in modes_to_test:
        print(f"测试 {modes} 个Chebyshev modes...")
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = ConfigurableFNORC1d(
            modes=16, width=64, num_layers=4,
            cft_segments=4, cft_modes=modes, use_gating=True
        )
        
        best_loss = train_model_quick(model, train_loader, test_loader, device, epochs=200)
        
        results['chebyshev_modes'][f'{modes}_modes'] = {
            'test_error': best_loss,
            'segments': 4,
            'modes': modes,
            'improvement_over_baseline': (results['baseline_fno']['test_error'] - best_loss) / results['baseline_fno']['test_error'] * 100
        }
        
        print(f"{modes} modes: {best_loss:.6f} (改进: {results['chebyshev_modes'][f'{modes}_modes']['improvement_over_baseline']:.2f}%)")
        
        del model
        torch.cuda.empty_cache()
    
    # 4. 门控机制消融
    print("\n4. 门控机制消融实验")
    print("-" * 40)
    
    # 无门控
    print("测试无门控机制...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = ConfigurableFNORC1d(
        modes=16, width=64, num_layers=4,
        cft_segments=4, cft_modes=8, use_gating=False
    )
    
    best_loss = train_model_quick(model, train_loader, test_loader, device, epochs=200)
    
    results['gating_ablation']['no_gating'] = {
        'test_error': best_loss,
        'use_gating': False,
        'improvement_over_baseline': (results['baseline_fno']['test_error'] - best_loss) / results['baseline_fno']['test_error'] * 100
    }
    
    print(f"无门控: {best_loss:.6f} (改进: {results['gating_ablation']['no_gating']['improvement_over_baseline']:.2f}%)")
    
    del model
    torch.cuda.empty_cache()
    
    # 有门控
    print("测试有门控机制...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = ConfigurableFNORC1d(
        modes=16, width=64, num_layers=4,
        cft_segments=4, cft_modes=8, use_gating=True
    )
    
    best_loss = train_model_quick(model, train_loader, test_loader, device, epochs=200)
    
    results['gating_ablation']['with_gating'] = {
        'test_error': best_loss,
        'use_gating': True,
        'improvement_over_baseline': (results['baseline_fno']['test_error'] - best_loss) / results['baseline_fno']['test_error'] * 100
    }
    
    print(f"有门控: {best_loss:.6f} (改进: {results['gating_ablation']['with_gating']['improvement_over_baseline']:.2f}%)")
    
    del model
    torch.cuda.empty_cache()
    
    # 保存结果
    results_path = f"{base_path}/results/ablation_studies/ablation_results.json"
    os.makedirs(f"{base_path}/results/ablation_studies", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总结果
    print_ablation_summary(results)
    
    # 生成可视化
    create_ablation_plots(results, base_path)
    
    return results

def print_ablation_summary(results):
    """打印消融实验汇总"""
    print("\n" + "="*60)
    print("消融实验汇总结果")
    print("="*60)
    
    baseline_error = results['baseline_fno']['test_error']
    print(f"基线FNO误差: {baseline_error:.6f}")
    
    print("\n1. CFT分段数量影响:")
    for key, value in results['cft_segments'].items():
        print(f"  {value['segments']} segments: {value['test_error']:.6f} (改进: {value['improvement_over_baseline']:.2f}%)")
    
    print("\n2. Chebyshev模式数量影响:")
    for key, value in results['chebyshev_modes'].items():
        print(f"  {value['modes']} modes: {value['test_error']:.6f} (改进: {value['improvement_over_baseline']:.2f}%)")
    
    print("\n3. 门控机制影响:")
    no_gating = results['gating_ablation']['no_gating']
    with_gating = results['gating_ablation']['with_gating']
    print(f"  无门控: {no_gating['test_error']:.6f} (改进: {no_gating['improvement_over_baseline']:.2f}%)")
    print(f"  有门控: {with_gating['test_error']:.6f} (改进: {with_gating['improvement_over_baseline']:.2f}%)")
    
    gating_improvement = (no_gating['test_error'] - with_gating['test_error']) / no_gating['test_error'] * 100
    print(f"  门控机制额外改进: {gating_improvement:.2f}%")

def create_ablation_plots(results, base_path):
    """创建消融实验可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: CFT分段数量影响
    ax1 = axes[0, 0]
    segments = [results['cft_segments'][key]['segments'] for key in results['cft_segments'].keys()]
    errors = [results['cft_segments'][key]['test_error'] for key in results['cft_segments'].keys()]
    
    ax1.plot(segments, errors, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=results['baseline_fno']['test_error'], color='red', linestyle='--', label='Baseline FNO')
    ax1.set_xlabel('CFT Segments')
    ax1.set_ylabel('Test Error')
    ax1.set_title('Effect of CFT Segments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: Chebyshev模式数量影响
    ax2 = axes[0, 1]
    modes = [results['chebyshev_modes'][key]['modes'] for key in results['chebyshev_modes'].keys()]
    errors = [results['chebyshev_modes'][key]['test_error'] for key in results['chebyshev_modes'].keys()]
    
    ax2.plot(modes, errors, 's-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=results['baseline_fno']['test_error'], color='red', linestyle='--', label='Baseline FNO')
    ax2.set_xlabel('Chebyshev Modes')
    ax2.set_ylabel('Test Error')
    ax2.set_title('Effect of Chebyshev Modes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 门控机制对比
    ax3 = axes[1, 0]
    gating_configs = ['No Gating', 'With Gating']
    gating_errors = [
        results['gating_ablation']['no_gating']['test_error'],
        results['gating_ablation']['with_gating']['test_error']
    ]
    
    bars = ax3.bar(gating_configs, gating_errors, alpha=0.7, color=['orange', 'purple'])
    ax3.axhline(y=results['baseline_fno']['test_error'], color='red', linestyle='--', label='Baseline FNO')
    ax3.set_ylabel('Test Error')
    ax3.set_title('Effect of Gating Mechanism')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, error in zip(bars, gating_errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{error:.6f}', ha='center', va='bottom')
    
    # 子图4: 总体改进对比
    ax4 = axes[1, 1]
    
    # 收集所有配置的改进百分比
    all_improvements = []
    all_labels = []
    
    # CFT segments
    for key, value in results['cft_segments'].items():
        all_improvements.append(value['improvement_over_baseline'])
        all_labels.append(f"{value['segments']} seg")
    
    # Chebyshev modes
    for key, value in results['chebyshev_modes'].items():
        all_improvements.append(value['improvement_over_baseline'])
        all_labels.append(f"{value['modes']} mode")
    
    # Gating
    all_improvements.extend([
        results['gating_ablation']['no_gating']['improvement_over_baseline'],
        results['gating_ablation']['with_gating']['improvement_over_baseline']
    ])
    all_labels.extend(['No Gate', 'With Gate'])
    
    colors = ['blue']*len(results['cft_segments']) + ['green']*len(results['chebyshev_modes']) + ['orange', 'purple']
    
    bars = ax4.bar(range(len(all_improvements)), all_improvements, alpha=0.7, color=colors)
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Improvement over Baseline (%)')
    ax4.set_title('Improvement Summary')
    ax4.set_xticks(range(len(all_labels)))
    ax4.set_xticklabels(all_labels, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/ablation_studies/ablation_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"消融实验图表已保存到: {base_path}/results/ablation_studies/")

# ================================
# 主执行函数
# ================================

if __name__ == "__main__":
    print("FNO-RC 消融实验")
    print("适用于Google Colab环境")
    print("预计运行时间: 2-3小时")
    
    # 运行实验
    results = run_ablation_experiments()
    
    print("\n消融实验完成！结果已保存到Google Drive。")
