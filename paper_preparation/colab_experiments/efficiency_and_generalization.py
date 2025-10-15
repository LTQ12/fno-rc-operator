"""
FNO-RC 计算效率和泛化性能实验
专为Google Colab环境设计
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
import psutil
import warnings
warnings.filterwarnings('ignore')

# 导入之前定义的模型组件
from statistical_validation_experiments import (
    setup_colab_environment, load_burgers_data, prepare_data_loaders,
    LpLoss, StandardFNO1d
)
from ablation_experiments import ConfigurableFNORC1d

# ================================
# 计算效率测量工具
# ================================

class EfficiencyProfiler:
    """效率分析工具"""
    def __init__(self, device):
        self.device = device
        self.results = {}
    
    def profile_model(self, model, data_loader, model_name, num_batches=10):
        """分析模型的计算效率"""
        model.eval()
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 内存使用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # 推理时间测试
        inference_times = []
        
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                data = data.to(self.device)
                
                # 预热
                if i == 0:
                    _ = model(data)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # 计时
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                output = model(data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                if i > 0:  # 跳过第一次（预热）
                    inference_times.append(end_time - start_time)
        
        # 内存峰值
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage = peak_memory - initial_memory
        else:
            memory_usage = 0
            peak_memory = 0
        
        # FLOPs估算（简化）
        sample_input = next(iter(data_loader))[0][:1].to(self.device)
        flops = self.estimate_flops(model, sample_input)
        
        results = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'inference_time_mean': float(np.mean(inference_times)),
            'inference_time_std': float(np.std(inference_times)),
            'memory_usage_mb': float(memory_usage / 1024 / 1024),
            'peak_memory_mb': float(peak_memory / 1024 / 1024),
            'estimated_flops': int(flops),
            'throughput_samples_per_sec': float(data_loader.batch_size / np.mean(inference_times))
        }
        
        self.results[model_name] = results
        return results
    
    def estimate_flops(self, model, sample_input):
        """简化的FLOPs估算"""
        # 这是一个简化的估算，实际情况会更复杂
        total_params = sum(p.numel() for p in model.parameters())
        input_size = sample_input.numel()
        
        # 粗略估算：每个参数大约需要2次操作（乘法和加法）
        estimated_flops = total_params * 2 * input_size
        
        return estimated_flops
    
    def compare_models(self):
        """比较不同模型的效率"""
        if len(self.results) < 2:
            return None
        
        comparison = {}
        baseline_name = list(self.results.keys())[0]
        baseline = self.results[baseline_name]
        
        for model_name, metrics in self.results.items():
            if model_name == baseline_name:
                continue
            
            comparison[model_name] = {
                'parameter_ratio': metrics['total_parameters'] / baseline['total_parameters'],
                'speed_ratio': baseline['inference_time_mean'] / metrics['inference_time_mean'],
                'memory_ratio': metrics['memory_usage_mb'] / max(baseline['memory_usage_mb'], 1),
                'flops_ratio': metrics['estimated_flops'] / baseline['estimated_flops']
            }
        
        return comparison

# ================================
# 泛化性能测试
# ================================

def create_resolution_datasets(base_resolution=128):
    """创建不同分辨率的数据集"""
    resolutions = [64, 128, 256]
    datasets = {}
    
    for res in resolutions:
        print(f"创建分辨率 {res} 的数据集...")
        
        # 生成数据
        n_samples = 200 if res <= 128 else 100  # 高分辨率用较少样本
        
        train_a = []
        train_u = []
        
        np.random.seed(42)
        
        for i in range(n_samples):
            x = np.linspace(0, 1, res)
            # 生成更复杂的初始条件
            u0 = (np.sin(2 * np.pi * x) + 
                  0.5 * np.sin(4 * np.pi * x) + 
                  0.3 * np.cos(6 * np.pi * x) + 
                  np.random.normal(0, 0.1, res))
            
            # 简化的演化（实际应该用PDE求解器）
            u_final = u0 * 0.8 + 0.1 * np.sin(4 * np.pi * x) + np.random.normal(0, 0.05, res)
            
            train_a.append(u0)
            train_u.append(u_final)
        
        train_a = torch.tensor(np.array(train_a), dtype=torch.float32)
        train_u = torch.tensor(np.array(train_u), dtype=torch.float32)
        
        # 准备输入格式
        x_coord = torch.linspace(0, 1, res).reshape(1, res, 1)
        train_input = torch.cat([train_a.unsqueeze(-1), x_coord.repeat(n_samples, 1, 1)], dim=-1)
        train_target = train_u.unsqueeze(-1)
        
        datasets[res] = {
            'input': train_input,
            'target': train_target,
            'loader': DataLoader(TensorDataset(train_input, train_target), batch_size=10, shuffle=False)
        }
    
    return datasets

def test_resolution_generalization(model, datasets, device):
    """测试分辨率泛化性能"""
    model.eval()
    loss_fn = LpLoss(size_average=True)
    
    results = {}
    
    with torch.no_grad():
        for res, dataset in datasets.items():
            total_loss = 0
            num_batches = 0
            
            try:
                for data, target in dataset['loader']:
                    data, target = data.to(device), target.to(device)
                    
                    # 调整模型输入尺寸（如果需要）
                    if hasattr(model, 'fc0'):
                        output = model(data)
                    else:
                        continue
                    
                    loss = loss_fn(output, target)
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                results[res] = avg_loss
                
            except Exception as e:
                print(f"分辨率 {res} 测试失败: {e}")
                results[res] = float('inf')
    
    return results

# ================================
# 长期预测稳定性测试
# ================================

def test_long_term_stability(model, test_loader, device, max_steps=100):
    """测试长期预测稳定性"""
    model.eval()
    loss_fn = LpLoss(size_average=True)
    
    # 选择一个测试样本
    test_sample = next(iter(test_loader))
    data, target = test_sample[0][:1].to(device), test_sample[1][:1].to(device)
    
    # 模拟长期预测
    stability_results = []
    current_input = data.clone()
    
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            # 预测下一步
            prediction = model(current_input)
            
            # 计算与目标的误差
            error = loss_fn(prediction, target).item()
            stability_results.append({
                'step': step,
                'error': error,
                'prediction_norm': torch.norm(prediction).item(),
                'prediction_mean': torch.mean(prediction).item(),
                'prediction_std': torch.std(prediction).item()
            })
            
            # 使用预测作为下一步输入（自回归预测）
            # 这里简化处理，实际应该更复杂
            if step < max_steps:
                # 更新输入的函数部分，保持坐标不变
                current_input = torch.cat([
                    prediction,  # 新的函数值
                    current_input[:, :, 1:2]  # 保持坐标
                ], dim=-1)
            
            # 检查数值稳定性
            if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                print(f"数值不稳定在第 {step} 步")
                break
            
            # 早停条件
            if error > 10.0:  # 误差过大
                print(f"误差过大，在第 {step} 步停止")
                break
    
    return stability_results

# ================================
# 主实验函数
# ================================

def run_efficiency_experiments():
    """运行计算效率实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    train_a, train_u, test_a, test_u = load_burgers_data()
    train_loader, test_loader = prepare_data_loaders(train_a, train_u, test_a, test_u, batch_size=10)
    
    # 创建效率分析器
    profiler = EfficiencyProfiler(device)
    
    print("="*60)
    print("计算效率分析实验")
    print("="*60)
    
    # 测试模型配置
    models_to_test = [
        {
            'name': 'Standard_FNO',
            'model': StandardFNO1d(modes=16, width=64, num_layers=4),
            'description': 'Baseline FNO model'
        },
        {
            'name': 'FNO_RC_Small',
            'model': ConfigurableFNORC1d(modes=16, width=64, num_layers=4, 
                                       cft_segments=2, cft_modes=4, use_gating=True),
            'description': 'FNO-RC with small CFT configuration'
        },
        {
            'name': 'FNO_RC_Standard',
            'model': ConfigurableFNORC1d(modes=16, width=64, num_layers=4,
                                       cft_segments=4, cft_modes=8, use_gating=True),
            'description': 'FNO-RC with standard CFT configuration'
        },
        {
            'name': 'FNO_RC_Large',
            'model': ConfigurableFNORC1d(modes=16, width=64, num_layers=4,
                                       cft_segments=8, cft_modes=16, use_gating=True),
            'description': 'FNO-RC with large CFT configuration'
        }
    ]
    
    efficiency_results = {}
    
    for model_config in models_to_test:
        print(f"\n测试模型: {model_config['name']}")
        print(f"描述: {model_config['description']}")
        print("-" * 40)
        
        model = model_config['model'].to(device)
        
        # 分析效率
        results = profiler.profile_model(model, test_loader, model_config['name'])
        
        print(f"参数量: {results['total_parameters']:,}")
        print(f"推理时间: {results['inference_time_mean']:.4f} ± {results['inference_time_std']:.4f} 秒")
        print(f"内存使用: {results['memory_usage_mb']:.1f} MB")
        print(f"吞吐量: {results['throughput_samples_per_sec']:.1f} 样本/秒")
        
        efficiency_results[model_config['name']] = results
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
    
    # 比较分析
    comparison = profiler.compare_models()
    
    print("\n" + "="*60)
    print("效率对比分析")
    print("="*60)
    
    if comparison:
        for model_name, ratios in comparison.items():
            print(f"\n{model_name} vs 基线:")
            print(f"  参数量倍数: {ratios['parameter_ratio']:.2f}×")
            print(f"  速度比: {ratios['speed_ratio']:.2f}× ({'更快' if ratios['speed_ratio'] > 1 else '更慢'})")
            print(f"  内存倍数: {ratios['memory_ratio']:.2f}×")
            print(f"  FLOPs倍数: {ratios['flops_ratio']:.2f}×")
    
    # 保存效率结果
    efficiency_path = f"{base_path}/results/efficiency_analysis"
    os.makedirs(efficiency_path, exist_ok=True)
    
    with open(f"{efficiency_path}/efficiency_results.json", 'w') as f:
        json.dump({
            'individual_results': efficiency_results,
            'comparison': comparison,
            'metadata': {
                'device': str(device),
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)
    
    return efficiency_results, comparison

def run_generalization_experiments():
    """运行泛化性能实验"""
    device, base_path = setup_colab_environment()
    
    print("\n" + "="*60)
    print("泛化性能实验")
    print("="*60)
    
    # 创建不同分辨率数据集
    print("\n1. 分辨率泛化测试")
    print("-" * 40)
    
    resolution_datasets = create_resolution_datasets()
    
    # 训练一个标准模型用于测试
    train_a, train_u, test_a, test_u = load_burgers_data()
    train_loader, _ = prepare_data_loaders(train_a, train_u, test_a, test_u)
    
    # 快速训练模型
    torch.manual_seed(42)
    model = ConfigurableFNORC1d(modes=16, width=64, num_layers=4,
                               cft_segments=4, cft_modes=8, use_gating=True)
    
    print("快速训练模型用于泛化测试...")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = LpLoss(size_average=True)
    
    # 简单训练
    model.train()
    for epoch in range(100):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # 测试分辨率泛化
    resolution_results = test_resolution_generalization(model, resolution_datasets, device)
    
    print("\n分辨率泛化结果:")
    for res, error in resolution_results.items():
        print(f"  分辨率 {res}: 误差 = {error:.6f}")
    
    # 长期预测稳定性测试
    print("\n2. 长期预测稳定性测试")
    print("-" * 40)
    
    _, test_loader = prepare_data_loaders(train_a, train_u, test_a, test_u, batch_size=1)
    stability_results = test_long_term_stability(model, test_loader, device, max_steps=50)
    
    print(f"长期预测测试完成，共 {len(stability_results)} 步")
    
    # 分析稳定性趋势
    errors = [result['error'] for result in stability_results]
    error_growth_rate = (errors[-1] - errors[0]) / len(errors) if len(errors) > 1 else 0
    
    print(f"误差增长率: {error_growth_rate:.6f} 每步")
    print(f"最终误差: {errors[-1]:.6f}")
    
    # 保存泛化结果
    generalization_path = f"{base_path}/results/generalization_analysis"
    os.makedirs(generalization_path, exist_ok=True)
    
    generalization_results = {
        'resolution_generalization': resolution_results,
        'long_term_stability': stability_results,
        'stability_metrics': {
            'error_growth_rate': error_growth_rate,
            'final_error': errors[-1] if errors else None,
            'stable_steps': len(stability_results)
        },
        'metadata': {
            'device': str(device),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    with open(f"{generalization_path}/generalization_results.json", 'w') as f:
        json.dump(generalization_results, f, indent=2)
    
    # 清理内存
    del model
    torch.cuda.empty_cache()
    
    return generalization_results

def create_efficiency_plots(efficiency_results, comparison, base_path):
    """创建效率分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    model_names = list(efficiency_results.keys())
    
    # 子图1: 参数量对比
    ax1 = axes[0, 0]
    params = [efficiency_results[name]['total_parameters'] for name in model_names]
    bars1 = ax1.bar(model_names, params, alpha=0.7, color='skyblue')
    ax1.set_ylabel('参数量')
    ax1.set_title('模型参数量对比')
    ax1.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值
    for bar, param in zip(bars1, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}', ha='center', va='bottom', fontsize=8)
    
    # 子图2: 推理时间对比
    ax2 = axes[0, 1]
    inference_times = [efficiency_results[name]['inference_time_mean'] for name in model_names]
    inference_stds = [efficiency_results[name]['inference_time_std'] for name in model_names]
    
    bars2 = ax2.bar(model_names, inference_times, yerr=inference_stds, 
                   alpha=0.7, color='lightcoral', capsize=5)
    ax2.set_ylabel('推理时间 (秒)')
    ax2.set_title('推理时间对比')
    ax2.tick_params(axis='x', rotation=45)
    
    # 子图3: 内存使用对比
    ax3 = axes[1, 0]
    memory_usage = [efficiency_results[name]['memory_usage_mb'] for name in model_names]
    bars3 = ax3.bar(model_names, memory_usage, alpha=0.7, color='lightgreen')
    ax3.set_ylabel('内存使用 (MB)')
    ax3.set_title('内存使用对比')
    ax3.tick_params(axis='x', rotation=45)
    
    # 子图4: 吞吐量对比
    ax4 = axes[1, 1]
    throughput = [efficiency_results[name]['throughput_samples_per_sec'] for name in model_names]
    bars4 = ax4.bar(model_names, throughput, alpha=0.7, color='gold')
    ax4.set_ylabel('吞吐量 (样本/秒)')
    ax4.set_title('吞吐量对比')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/efficiency_analysis/efficiency_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 主执行函数
# ================================

if __name__ == "__main__":
    print("FNO-RC 计算效率和泛化性能实验")
    print("适用于Google Colab环境")
    print("预计运行时间: 2-3小时")
    
    # 运行效率实验
    efficiency_results, comparison = run_efficiency_experiments()
    
    # 运行泛化实验
    generalization_results = run_generalization_experiments()
    
    # 创建可视化
    device, base_path = setup_colab_environment()
    create_efficiency_plots(efficiency_results, comparison, base_path)
    
    print("\n所有实验完成！结果已保存到Google Drive。")
