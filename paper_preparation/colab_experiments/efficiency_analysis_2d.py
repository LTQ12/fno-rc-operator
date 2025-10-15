"""
FNO-RC 2D Navier-Stokes 计算效率分析实验
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
import psutil
import warnings
warnings.filterwarnings('ignore')

# 导入2D模型组件 - 修复导入
import sys
sys.path.append('.')

from statistical_validation_2d_ns import (
    setup_colab_environment, load_navier_stokes_data, prepare_data_loaders_2d,
    LpLoss, StandardFNO2d, FNORCF2d
)

# ================================
# 计算效率测量工具
# ================================

class EfficiencyProfiler2D:
    """2D效率分析工具"""
    def __init__(self, device):
        self.device = device
        self.results = {}
    
    def profile_model_2d(self, model, data_loader, model_name, num_batches=20):
        """分析2D模型的计算效率"""
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
        
        # FLOPs估算（2D特定）
        sample_input = next(iter(data_loader))[0][:1].to(self.device)
        flops = self.estimate_flops_2d(model, sample_input)
        
        results = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'inference_time_mean': float(np.mean(inference_times)),
            'inference_time_std': float(np.std(inference_times)),
            'memory_usage_mb': float(memory_usage / 1024 / 1024),
            'peak_memory_mb': float(peak_memory / 1024 / 1024),
            'estimated_flops': int(flops),
            'throughput_samples_per_sec': float(data_loader.batch_size / np.mean(inference_times)),
            'flops_per_sample': int(flops / data_loader.batch_size)
        }
        
        self.results[model_name] = results
        return results
    
    def estimate_flops_2d(self, model, sample_input):
        """2D模型的FLOPs估算"""
        total_params = sum(p.numel() for p in model.parameters())
        
        # 获取输入尺寸
        batch_size, h, w, channels = sample_input.shape
        
        # 2D卷积和FFT操作的粗略估算
        # FFT: O(N log N) where N = h * w
        fft_ops = batch_size * h * w * np.log2(h * w) * 2  # 2D FFT
        
        # 线性层操作
        linear_ops = total_params * batch_size * 2  # 乘法和加法
        
        # 总FLOPs
        total_flops = fft_ops + linear_ops
        
        return total_flops
    
    def measure_training_time(self, model, train_loader, device, epochs=10):
        """测量训练时间"""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = LpLoss(size_average=True)
        
        training_times = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 10:  # 只训练前10个batch
                    break
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            training_times.append(end_time - start_time)
            
            if epoch % 3 == 0:
                print(f'Training epoch {epoch}: {end_time - start_time:.2f}s')
        
        return {
            'mean_training_time_per_epoch': float(np.mean(training_times)),
            'std_training_time_per_epoch': float(np.std(training_times)),
            'total_training_time': float(np.sum(training_times))
        }
    
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
                'flops_ratio': metrics['estimated_flops'] / baseline['estimated_flops'],
                'throughput_ratio': metrics['throughput_samples_per_sec'] / baseline['throughput_samples_per_sec']
            }
        
        return comparison

# ================================
# 主实验函数
# ================================

def run_efficiency_experiments_2d():
    """运行2D计算效率实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    train_a, train_u, test_a, test_u = load_navier_stokes_data()
    train_loader, test_loader = prepare_data_loaders_2d(train_a, train_u, test_a, test_u, batch_size=8)
    
    # 创建效率分析器
    profiler = EfficiencyProfiler2D(device)
    
    print("="*60)
    print("2D Navier-Stokes 计算效率分析实验")
    print("="*60)
    
    # 测试模型配置
    models_to_test = [
        {
            'name': 'Standard_FNO_2D',
            'model': StandardFNO2d(modes1=12, modes2=12, width=32, num_layers=4),
            'description': 'Baseline 2D FNO model'
        },
        {
            'name': 'FNO_RC_Small_2D',
            'model': FNORCF2d(modes1=12, modes2=12, width=32, num_layers=4, 
                             cft_segments=2, cft_modes1=4, cft_modes2=4),
            'description': 'FNO-RC with small CFT configuration'
        },
        {
            'name': 'FNO_RC_Standard_2D',
            'model': FNORCF2d(modes1=12, modes2=12, width=32, num_layers=4,
                             cft_segments=4, cft_modes1=8, cft_modes2=8),
            'description': 'FNO-RC with standard CFT configuration'
        },
        {
            'name': 'FNO_RC_Large_2D',
            'model': FNORCF2d(modes1=12, modes2=12, width=32, num_layers=4,
                             cft_segments=6, cft_modes1=12, cft_modes2=12),
            'description': 'FNO-RC with large CFT configuration'
        }
    ]
    
    efficiency_results = {}
    training_results = {}
    
    for model_config in models_to_test:
        print(f"\n测试模型: {model_config['name']}")
        print(f"描述: {model_config['description']}")
        print("-" * 40)
        
        model = model_config['model'].to(device)
        
        # 推理效率分析
        print("分析推理效率...")
        results = profiler.profile_model_2d(model, test_loader, model_config['name'])
        
        print(f"参数量: {results['total_parameters']:,}")
        print(f"推理时间: {results['inference_time_mean']:.4f} ± {results['inference_time_std']:.4f} 秒")
        print(f"内存使用: {results['memory_usage_mb']:.1f} MB")
        print(f"吞吐量: {results['throughput_samples_per_sec']:.1f} 样本/秒")
        print(f"FLOPs: {results['estimated_flops']:,}")
        
        efficiency_results[model_config['name']] = results
        
        # 训练时间分析
        print("分析训练效率...")
        training_time_results = profiler.measure_training_time(model, train_loader, device, epochs=5)
        training_results[model_config['name']] = training_time_results
        
        print(f"训练时间/epoch: {training_time_results['mean_training_time_per_epoch']:.2f} ± {training_time_results['std_training_time_per_epoch']:.2f} 秒")
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
    
    # 比较分析
    comparison = profiler.compare_models()
    
    print("\n" + "="*60)
    print("2D效率对比分析")
    print("="*60)
    
    if comparison:
        for model_name, ratios in comparison.items():
            print(f"\n{model_name} vs 基线:")
            print(f"  参数量倍数: {ratios['parameter_ratio']:.2f}×")
            print(f"  推理速度比: {ratios['speed_ratio']:.2f}× ({'更快' if ratios['speed_ratio'] > 1 else '更慢'})")
            print(f"  内存倍数: {ratios['memory_ratio']:.2f}×")
            print(f"  FLOPs倍数: {ratios['flops_ratio']:.2f}×")
            print(f"  吞吐量比: {ratios['throughput_ratio']:.2f}×")
    
    # 保存效率结果
    efficiency_path = f"{base_path}/results/efficiency_analysis_2d"
    os.makedirs(efficiency_path, exist_ok=True)
    
    final_results = {
        'inference_efficiency': efficiency_results,
        'training_efficiency': training_results,
        'comparison': comparison,
        'metadata': {
            'problem': '2D Navier-Stokes',
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
            'data_shape': f"train: {train_a.shape}, test: {test_a.shape}"
        }
    }
    
    with open(f"{efficiency_path}/2d_efficiency_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 生成可视化
    create_efficiency_plots_2d(efficiency_results, training_results, comparison, base_path)
    
    return final_results

def create_efficiency_plots_2d(efficiency_results, training_results, comparison, base_path):
    """创建2D效率分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_names = list(efficiency_results.keys())
    
    # 子图1: 参数量对比
    ax1 = axes[0, 0]
    params = [efficiency_results[name]['total_parameters'] for name in model_names]
    bars1 = ax1.bar(range(len(model_names)), params, alpha=0.7, color='skyblue')
    ax1.set_ylabel('参数量')
    ax1.set_title('2D: 模型参数量对比')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels([name.replace('_2D', '') for name in model_names], rotation=45)
    
    # 在柱状图上添加数值
    for i, (bar, param) in enumerate(zip(bars1, params)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param/1000:.0f}K', ha='center', va='bottom', fontsize=8)
    
    # 子图2: 推理时间对比
    ax2 = axes[0, 1]
    inference_times = [efficiency_results[name]['inference_time_mean'] for name in model_names]
    inference_stds = [efficiency_results[name]['inference_time_std'] for name in model_names]
    
    bars2 = ax2.bar(range(len(model_names)), inference_times, yerr=inference_stds, 
                   alpha=0.7, color='lightcoral', capsize=5)
    ax2.set_ylabel('推理时间 (秒)')
    ax2.set_title('2D: 推理时间对比')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([name.replace('_2D', '') for name in model_names], rotation=45)
    
    # 子图3: 内存使用对比
    ax3 = axes[0, 2]
    memory_usage = [efficiency_results[name]['memory_usage_mb'] for name in model_names]
    bars3 = ax3.bar(range(len(model_names)), memory_usage, alpha=0.7, color='lightgreen')
    ax3.set_ylabel('内存使用 (MB)')
    ax3.set_title('2D: 内存使用对比')
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels([name.replace('_2D', '') for name in model_names], rotation=45)
    
    # 子图4: 吞吐量对比
    ax4 = axes[1, 0]
    throughput = [efficiency_results[name]['throughput_samples_per_sec'] for name in model_names]
    bars4 = ax4.bar(range(len(model_names)), throughput, alpha=0.7, color='gold')
    ax4.set_ylabel('吞吐量 (样本/秒)')
    ax4.set_title('2D: 吞吐量对比')
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels([name.replace('_2D', '') for name in model_names], rotation=45)
    
    # 子图5: FLOPs对比
    ax5 = axes[1, 1]
    flops = [efficiency_results[name]['estimated_flops'] / 1e9 for name in model_names]  # 转换为GFLOPs
    bars5 = ax5.bar(range(len(model_names)), flops, alpha=0.7, color='mediumpurple')
    ax5.set_ylabel('FLOPs (GFLOPs)')
    ax5.set_title('2D: 计算复杂度对比')
    ax5.set_xticks(range(len(model_names)))
    ax5.set_xticklabels([name.replace('_2D', '') for name in model_names], rotation=45)
    
    # 子图6: 训练时间对比
    ax6 = axes[1, 2]
    train_times = [training_results[name]['mean_training_time_per_epoch'] for name in model_names]
    train_stds = [training_results[name]['std_training_time_per_epoch'] for name in model_names]
    
    bars6 = ax6.bar(range(len(model_names)), train_times, yerr=train_stds,
                   alpha=0.7, color='lightsteelblue', capsize=5)
    ax6.set_ylabel('训练时间/Epoch (秒)')
    ax6.set_title('2D: 训练时间对比')
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels([name.replace('_2D', '') for name in model_names], rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/efficiency_analysis_2d/2d_efficiency_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建效率比较雷达图
    create_efficiency_radar_chart(efficiency_results, comparison, base_path)

def create_efficiency_radar_chart(efficiency_results, comparison, base_path):
    """创建效率比较雷达图"""
    if not comparison:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 定义指标
    metrics = ['参数效率', '推理速度', '内存效率', '计算效率', '吞吐量']
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合圆圈
    
    # 为每个模型绘制雷达图
    colors = ['blue', 'green', 'orange']
    model_names = list(comparison.keys())
    
    for i, (model_name, ratios) in enumerate(comparison.items()):
        # 计算效率得分（越高越好）
        values = [
            1 / ratios['parameter_ratio'],  # 参数效率：参数越少越好
            ratios['speed_ratio'],          # 推理速度：越快越好
            1 / ratios['memory_ratio'],     # 内存效率：内存越少越好
            1 / ratios['flops_ratio'],      # 计算效率：FLOPs越少越好
            ratios['throughput_ratio']      # 吞吐量：越高越好
        ]
        values += values[:1]  # 闭合圆圈
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name.replace('_2D', ''), color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    # 添加标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 2)
    ax.set_title('2D模型效率对比雷达图\n(数值越大表示效率越高)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.grid(True)
    
    plt.savefig(f"{base_path}/results/efficiency_analysis_2d/2d_efficiency_radar.png",
                dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 主执行函数
# ================================

if __name__ == "__main__":
    print("FNO-RC 2D Navier-Stokes 计算效率分析实验")
    print("适用于Google Colab环境")
    print("专注于73.68%改进的最显著结果")
    print("预计运行时间: 1-2小时")
    
    # 运行实验
    results = run_efficiency_experiments_2d()
    
    print("\n🎉 2D Navier-Stokes效率分析实验完成！")
    print("已详细分析73.68%改进的计算开销。")
    print("结果已保存到Google Drive。")
