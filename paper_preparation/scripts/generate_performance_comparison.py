#!/usr/bin/env python3
"""
生成论文用的性能对比图表
包括条形图、改进幅度图和误差量级对比图
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置matplotlib中文字体和高质量输出
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# 实验数据
experimental_results = {
    'dimensions': ['1D Burgers', '2D Navier-Stokes', '3D Navier-Stokes'],
    'baseline_error': [0.221149, 0.021767, 0.884708],
    'fno_rc_error': [0.214498, 0.005730, 0.497562],
    'improvement': [3.01, 73.68, 43.76],
    'task_type': ['Time Sequence', 'Spatiotemporal', 'High Reynolds'],
    'difficulty': ['Simple', 'Complex', 'Extreme']
}

def create_performance_bar_chart():
    """创建性能对比条形图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: 误差对比
    x_pos = np.arange(len(experimental_results['dimensions']))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, experimental_results['baseline_error'], 
                    width, label='Baseline FNO', color='#ff7f7f', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, experimental_results['fno_rc_error'], 
                    width, label='FNO-RC (Ours)', color='#2E8B57', alpha=0.8)
    
    ax1.set_xlabel('Problem Dimension')
    ax1.set_ylabel('L2 Test Error')
    ax1.set_title('Performance Comparison: Baseline FNO vs FNO-RC')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(experimental_results['dimensions'])
    ax1.legend()
    ax1.set_yscale('log')  # 使用对数刻度因为误差量级差异大
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, error in zip(bars1, experimental_results['baseline_error']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.4f}', ha='center', va='bottom', fontsize=10)
    
    for bar, error in zip(bars2, experimental_results['fno_rc_error']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 子图2: 相对改进幅度
    colors = ['#FFB347', '#32CD32', '#4169E1']  # 不同颜色表示不同复杂度
    bars3 = ax2.bar(experimental_results['dimensions'], experimental_results['improvement'], 
                    color=colors, alpha=0.8)
    
    ax2.set_xlabel('Problem Dimension')
    ax2.set_ylabel('Relative Improvement (%)')
    ax2.set_title('Relative Performance Improvement')
    ax2.grid(True, alpha=0.3)
    
    # 添加改进数值和复杂度标签
    for bar, improvement, difficulty in zip(bars3, experimental_results['improvement'], 
                                          experimental_results['difficulty']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{improvement:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'({difficulty})', ha='center', va='center', fontsize=10, 
                style='italic', color='white')
    
    # 添加改进显著性指示线
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% improvement')
    ax2.axhline(y=50, color='purple', linestyle='--', alpha=0.7, label='50% improvement')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def create_error_magnitude_comparison():
    """创建误差量级对比图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 数据准备
    dimensions = experimental_results['dimensions']
    baseline_errors = experimental_results['baseline_error']
    fno_rc_errors = experimental_results['fno_rc_error']
    
    # 创建散点图
    x_pos = np.arange(len(dimensions))
    
    # 基线FNO误差
    scatter1 = ax.scatter(x_pos, baseline_errors, s=200, c='red', alpha=0.7, 
                         marker='o', label='Baseline FNO', edgecolors='black', linewidth=2)
    
    # FNO-RC误差  
    scatter2 = ax.scatter(x_pos, fno_rc_errors, s=200, c='green', alpha=0.7,
                         marker='^', label='FNO-RC (Ours)', edgecolors='black', linewidth=2)
    
    # 连接线显示改进
    for i, (base, rc) in enumerate(zip(baseline_errors, fno_rc_errors)):
        ax.plot([i, i], [rc, base], 'k-', alpha=0.5, linewidth=2)
        # 添加改进箭头
        ax.annotate('', xy=(i, rc), xytext=(i, base),
                   arrowprops=dict(arrowstyle='<-', color='blue', lw=2))
        
        # 添加改进百分比标签
        mid_point = (base + rc) / 2
        improvement = experimental_results['improvement'][i]
        ax.text(i + 0.1, mid_point, f'-{improvement:.1f}%', 
               fontsize=12, fontweight='bold', color='blue',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    ax.set_yscale('log')
    ax.set_xlabel('Problem Dimension')
    ax.set_ylabel('L2 Test Error (Log Scale)')
    ax.set_title('Error Magnitude Comparison Across Dimensions')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dimensions)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (base, rc) in enumerate(zip(baseline_errors, fno_rc_errors)):
        ax.text(i, base * 1.1, f'{base:.4f}', ha='center', va='bottom', 
               fontsize=10, color='red', fontweight='bold')
        ax.text(i, rc * 0.8, f'{rc:.4f}', ha='center', va='top', 
               fontsize=10, color='green', fontweight='bold')
    
    return fig

def create_task_complexity_analysis():
    """创建任务复杂度分析图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 数据准备
    complexity_score = [1, 3, 5]  # 1D简单, 2D复杂, 3D极端
    improvements = experimental_results['improvement']
    dimensions = experimental_results['dimensions']
    
    # 散点图显示复杂度vs改进幅度的关系
    colors = ['orange', 'green', 'blue']
    sizes = [100, 300, 200]  # 根据问题特征调整大小
    
    for i, (comp, imp, dim, color, size) in enumerate(zip(complexity_score, improvements, 
                                                         dimensions, colors, sizes)):
        ax.scatter(comp, imp, s=size, c=color, alpha=0.7, 
                  edgecolors='black', linewidth=2, label=dim)
        
        # 添加标签
        ax.text(comp + 0.1, imp + 2, f'{dim}\n({imp:.1f}%)', 
               fontsize=10, ha='left', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 拟合趋势线（除了2D异常点）
    # 2D的73.68%改进是异常高的，显示方法在中等复杂度问题上的突出表现
    trend_x = [1, 5]
    trend_y = [3.01, 43.76]
    ax.plot(trend_x, trend_y, 'r--', alpha=0.7, linewidth=2, 
           label='Expected Trend (1D→3D)')
    
    # 高亮2D异常点
    ax.scatter(3, 73.68, s=500, facecolors='none', edgecolors='red', 
              linewidth=3, marker='o', label='Outstanding Performance')
    
    ax.set_xlabel('Task Complexity Level')
    ax.set_ylabel('Relative Improvement (%)')
    ax.set_title('CFT-RC Performance vs Task Complexity')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 添加复杂度标签
    ax.text(1, -5, 'Simple\n(Time Sequence)', ha='center', va='top', fontsize=10)
    ax.text(3, -5, 'Complex\n(Spatiotemporal)', ha='center', va='top', fontsize=10)
    ax.text(5, -5, 'Extreme\n(High Reynolds)', ha='center', va='top', fontsize=10)
    
    return fig

def main():
    """生成所有性能对比图"""
    print("正在生成论文用性能对比图表...")
    
    # 创建保存目录
    import os
    os.makedirs('../figures', exist_ok=True)
    
    # 生成图表1: 基本性能对比
    print("1. 生成基本性能对比图...")
    fig1 = create_performance_bar_chart()
    fig1.savefig('../figures/performance_comparison.png', bbox_inches='tight', dpi=300)
    fig1.savefig('../figures/performance_comparison.pdf', bbox_inches='tight')
    
    # 生成图表2: 误差量级对比
    print("2. 生成误差量级对比图...")
    fig2 = create_error_magnitude_comparison()
    fig2.savefig('../figures/error_magnitude_comparison.png', bbox_inches='tight', dpi=300)
    fig2.savefig('../figures/error_magnitude_comparison.pdf', bbox_inches='tight')
    
    # 生成图表3: 任务复杂度分析
    print("3. 生成任务复杂度分析图...")
    fig3 = create_task_complexity_analysis()
    fig3.savefig('../figures/task_complexity_analysis.png', bbox_inches='tight', dpi=300)
    fig3.savefig('../figures/task_complexity_analysis.pdf', bbox_inches='tight')
    
    print("所有性能对比图表生成完成！")
    print("保存位置: paper_preparation/figures/")
    
    plt.show()

if __name__ == "__main__":
    main()
