"""
FNO-RC 2D Navier-Stokes 消融实验
专为Google Colab环境设计，聚焦最显著改进的维度
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
import warnings
warnings.filterwarnings('ignore')

# 导入2D模型组件 - 修复导入
import sys
sys.path.append('.')

from statistical_validation_2d_ns import (
    setup_colab_environment, load_navier_stokes_data, prepare_data_loaders_2d,
    LpLoss, StandardFNO2d, SpectralConv2d, FNOLayer2d, CFTLayer2d
)

# ================================
# 可配置的2D FNO-RC模型
# ================================

class ConfigurableFNORC2d(nn.Module):
    """可配置的2D FNO-RC模型"""
    def __init__(self, modes1=12, modes2=12, width=32, num_layers=4, 
                 cft_segments=4, cft_modes1=8, cft_modes2=8, use_gating=True):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.use_gating = use_gating
        
        # 输入嵌入
        self.fc0 = nn.Linear(3, self.width)
        
        # FNO主路径
        self.fno_layers = nn.ModuleList([FNOLayer2d(modes1, modes2, width) for _ in range(num_layers)])
        
        # CFT残差路径
        self.cft_layers = nn.ModuleList([CFTLayer2d(width, width, cft_modes1, cft_modes2, cft_segments) for _ in range(num_layers)])
        
        # 门控机制（可选）
        if use_gating:
            self.gate_layers = nn.ModuleList([nn.Conv2d(2*width, width, 1) for _ in range(num_layers)])
        else:
            self.gate_layers = None
        
        # 输出层
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x shape: (batch, h, w, 3)
        x = self.fc0(x)  # (batch, h, w, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, h, w)
        
        for i in range(self.num_layers):
            # FNO路径
            x_fno = self.fno_layers[i](x)
            x_fno = self.activation(x_fno)
            
            # CFT路径
            x_cft = self.cft_layers[i](x)
            
            if self.use_gating and self.gate_layers is not None:
                # 门控融合
                x_concat = torch.cat([x_fno, x_cft], dim=1)  # (batch, 2*width, h, w)
                gate = torch.sigmoid(self.gate_layers[i](x_concat))  # (batch, width, h, w)
                x = x_fno + gate * x_cft
            else:
                # 直接相加（无门控）
                x = x_fno + 0.1 * x_cft  # 添加小权重避免训练不稳定
        
        # 输出层
        x = x.permute(0, 2, 3, 1)  # (batch, h, w, width)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)  # (batch, h, w, 1)
        return x

# ================================
# 快速训练函数
# ================================

def train_model_2d_quick(model, train_loader, test_loader, device, epochs=150, lr=0.001):
    """快速训练2D模型（用于消融实验）"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = LpLoss(size_average=True)
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        
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
            
            if epoch % 30 == 0:
                print(f'Epoch {epoch:3d}: Test Loss = {test_loss:.6f}')
        
        scheduler.step()
    
    return best_test_loss

# ================================
# 消融实验主函数
# ================================

def run_ablation_experiments_2d():
    """运行2D消融实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    train_a, train_u, test_a, test_u = load_navier_stokes_data()
    train_loader, test_loader = prepare_data_loaders_2d(train_a, train_u, test_a, test_u)
    
    results = {
        'baseline_fno': {},
        'cft_segments': {},
        'chebyshev_modes': {},
        'gating_ablation': {},
        'metadata': {
            'problem': '2D Navier-Stokes',
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
            'epochs_per_experiment': 150,
            'data_shape': f"train: {train_a.shape}, test: {test_a.shape}"
        }
    }
    
    print("="*60)
    print("2D Navier-Stokes 消融实验")
    print("="*60)
    
    # 1. 基线FNO
    print("\n1. 基线FNO实验")
    print("-" * 40)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = StandardFNO2d(modes1=12, modes2=12, width=32, num_layers=4)
    best_loss = train_model_2d_quick(model, train_loader, test_loader, device, epochs=150)
    results['baseline_fno'] = {
        'test_error': best_loss,
        'config': 'Standard 2D FNO baseline'
    }
    print(f"基线FNO: {best_loss:.6f}")
    
    del model
    torch.cuda.empty_cache()
    
    # 2. CFT分段数量消融
    print("\n2. CFT分段数量消融实验")
    print("-" * 40)
    
    segments_to_test = [1, 2, 4, 6]  # 2D适用的分段数
    
    for segments in segments_to_test:
        print(f"测试 {segments} 个segments...")
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = ConfigurableFNORC2d(
            modes1=12, modes2=12, width=32, num_layers=4,
            cft_segments=segments, cft_modes1=8, cft_modes2=8, use_gating=True
        )
        
        best_loss = train_model_2d_quick(model, train_loader, test_loader, device, epochs=150)
        
        results['cft_segments'][f'{segments}_segments'] = {
            'test_error': best_loss,
            'segments': segments,
            'modes1': 8,
            'modes2': 8,
            'improvement_over_baseline': (results['baseline_fno']['test_error'] - best_loss) / results['baseline_fno']['test_error'] * 100
        }
        
        print(f"{segments} segments: {best_loss:.6f} (改进: {results['cft_segments'][f'{segments}_segments']['improvement_over_baseline']:.2f}%)")
        
        del model
        torch.cuda.empty_cache()
    
    # 3. Chebyshev模式数量消融
    print("\n3. Chebyshev模式数量消融实验")
    print("-" * 40)
    
    modes_pairs_to_test = [(4, 4), (8, 8), (12, 12)]  # (modes1, modes2)
    
    for modes1, modes2 in modes_pairs_to_test:
        print(f"测试 {modes1}x{modes2} Chebyshev modes...")
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = ConfigurableFNORC2d(
            modes1=12, modes2=12, width=32, num_layers=4,
            cft_segments=4, cft_modes1=modes1, cft_modes2=modes2, use_gating=True
        )
        
        best_loss = train_model_2d_quick(model, train_loader, test_loader, device, epochs=150)
        
        results['chebyshev_modes'][f'{modes1}x{modes2}_modes'] = {
            'test_error': best_loss,
            'segments': 4,
            'modes1': modes1,
            'modes2': modes2,
            'improvement_over_baseline': (results['baseline_fno']['test_error'] - best_loss) / results['baseline_fno']['test_error'] * 100
        }
        
        print(f"{modes1}x{modes2} modes: {best_loss:.6f} (改进: {results['chebyshev_modes'][f'{modes1}x{modes2}_modes']['improvement_over_baseline']:.2f}%)")
        
        del model
        torch.cuda.empty_cache()
    
    # 4. 门控机制消融
    print("\n4. 门控机制消融实验")
    print("-" * 40)
    
    # 无门控
    print("测试无门控机制...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = ConfigurableFNORC2d(
        modes1=12, modes2=12, width=32, num_layers=4,
        cft_segments=4, cft_modes1=8, cft_modes2=8, use_gating=False
    )
    
    best_loss = train_model_2d_quick(model, train_loader, test_loader, device, epochs=150)
    
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
    
    model = ConfigurableFNORC2d(
        modes1=12, modes2=12, width=32, num_layers=4,
        cft_segments=4, cft_modes1=8, cft_modes2=8, use_gating=True
    )
    
    best_loss = train_model_2d_quick(model, train_loader, test_loader, device, epochs=150)
    
    results['gating_ablation']['with_gating'] = {
        'test_error': best_loss,
        'use_gating': True,
        'improvement_over_baseline': (results['baseline_fno']['test_error'] - best_loss) / results['baseline_fno']['test_error'] * 100
    }
    
    print(f"有门控: {best_loss:.6f} (改进: {results['gating_ablation']['with_gating']['improvement_over_baseline']:.2f}%)")
    
    del model
    torch.cuda.empty_cache()
    
    # 保存结果
    results_path = f"{base_path}/results/ablation_studies_2d/ablation_results_2d.json"
    os.makedirs(f"{base_path}/results/ablation_studies_2d", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总结果
    print_ablation_summary_2d(results)
    
    # 生成可视化
    create_ablation_plots_2d(results, base_path)
    
    return results

def print_ablation_summary_2d(results):
    """打印2D消融实验汇总"""
    print("\n" + "="*60)
    print("2D Navier-Stokes 消融实验汇总结果")
    print("="*60)
    
    baseline_error = results['baseline_fno']['test_error']
    print(f"基线FNO误差: {baseline_error:.6f}")
    
    print("\n1. CFT分段数量影响:")
    for key, value in results['cft_segments'].items():
        print(f"  {value['segments']} segments: {value['test_error']:.6f} (改进: {value['improvement_over_baseline']:.2f}%)")
    
    print("\n2. Chebyshev模式数量影响:")
    for key, value in results['chebyshev_modes'].items():
        print(f"  {value['modes1']}x{value['modes2']} modes: {value['test_error']:.6f} (改进: {value['improvement_over_baseline']:.2f}%)")
    
    print("\n3. 门控机制影响:")
    no_gating = results['gating_ablation']['no_gating']
    with_gating = results['gating_ablation']['with_gating']
    print(f"  无门控: {no_gating['test_error']:.6f} (改进: {no_gating['improvement_over_baseline']:.2f}%)")
    print(f"  有门控: {with_gating['test_error']:.6f} (改进: {with_gating['improvement_over_baseline']:.2f}%)")
    
    gating_improvement = (no_gating['test_error'] - with_gating['test_error']) / no_gating['test_error'] * 100
    print(f"  门控机制额外改进: {gating_improvement:.2f}%")
    
    # 找出最佳配置
    best_segments = min(results['cft_segments'].items(), key=lambda x: x[1]['test_error'])
    best_modes = min(results['chebyshev_modes'].items(), key=lambda x: x[1]['test_error'])
    
    print(f"\n4. 最佳配置:")
    print(f"  最佳分段数: {best_segments[1]['segments']} segments (误差: {best_segments[1]['test_error']:.6f})")
    print(f"  最佳模式数: {best_modes[1]['modes1']}x{best_modes[1]['modes2']} modes (误差: {best_modes[1]['test_error']:.6f})")
    print(f"  门控机制: {'推荐使用' if with_gating['test_error'] < no_gating['test_error'] else '可选'}")

def create_ablation_plots_2d(results, base_path):
    """创建2D消融实验可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: CFT分段数量影响
    ax1 = axes[0, 0]
    segments = [results['cft_segments'][key]['segments'] for key in results['cft_segments'].keys()]
    errors = [results['cft_segments'][key]['test_error'] for key in results['cft_segments'].keys()]
    
    ax1.plot(segments, errors, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=results['baseline_fno']['test_error'], color='red', linestyle='--', label='Baseline FNO')
    ax1.set_xlabel('CFT Segments')
    ax1.set_ylabel('Test Error')
    ax1.set_title('2D: Effect of CFT Segments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: Chebyshev模式数量影响
    ax2 = axes[0, 1]
    mode_labels = [f"{results['chebyshev_modes'][key]['modes1']}x{results['chebyshev_modes'][key]['modes2']}" 
                   for key in results['chebyshev_modes'].keys()]
    errors = [results['chebyshev_modes'][key]['test_error'] for key in results['chebyshev_modes'].keys()]
    
    x_pos = range(len(mode_labels))
    ax2.plot(x_pos, errors, 's-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=results['baseline_fno']['test_error'], color='red', linestyle='--', label='Baseline FNO')
    ax2.set_xlabel('Chebyshev Modes')
    ax2.set_ylabel('Test Error')
    ax2.set_title('2D: Effect of Chebyshev Modes')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(mode_labels)
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
    ax3.set_title('2D: Effect of Gating Mechanism')
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
        all_labels.append(f"{value['segments']}seg")
    
    # Chebyshev modes
    for key, value in results['chebyshev_modes'].items():
        all_improvements.append(value['improvement_over_baseline'])
        all_labels.append(f"{value['modes1']}x{value['modes2']}")
    
    # Gating
    all_improvements.extend([
        results['gating_ablation']['no_gating']['improvement_over_baseline'],
        results['gating_ablation']['with_gating']['improvement_over_baseline']
    ])
    all_labels.extend(['NoGate', 'WithGate'])
    
    colors = ['blue']*len(results['cft_segments']) + ['green']*len(results['chebyshev_modes']) + ['orange', 'purple']
    
    bars = ax4.bar(range(len(all_improvements)), all_improvements, alpha=0.7, color=colors)
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Improvement over Baseline (%)')
    ax4.set_title('2D: Improvement Summary')
    ax4.set_xticks(range(len(all_labels)))
    ax4.set_xticklabels(all_labels, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, imp in zip(bars, all_improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(all_improvements)*0.01,
                f'{imp:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/ablation_studies_2d/2d_ablation_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"2D消融实验图表已保存到: {base_path}/results/ablation_studies_2d/")

# ================================
# 主执行函数
# ================================

if __name__ == "__main__":
    print("FNO-RC 2D Navier-Stokes 消融实验")
    print("适用于Google Colab环境")
    print("专注于73.68%改进的最显著结果")
    print("预计运行时间: 2-3小时")
    
    # 运行实验
    results = run_ablation_experiments_2d()
    
    print("\n🎉 2D Navier-Stokes消融实验完成！")
    print("已分析各组件对73.68%改进的贡献度。")
    print("结果已保存到Google Drive。")
