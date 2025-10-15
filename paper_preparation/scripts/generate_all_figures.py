#!/usr/bin/env python3
"""
一键生成论文所需的所有图表
"""

import os
import sys
import subprocess
import argparse

def run_script(script_name, description):
    """运行指定的脚本"""
    print(f"\n{'='*60}")
    print(f"正在生成: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print(f"✅ {description} 生成成功!")
            if result.stdout:
                print("输出:", result.stdout)
        else:
            print(f"❌ {description} 生成失败!")
            print("错误:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行 {script_name} 时出错: {e}")
        return False
    
    return True

def create_figure_summary():
    """创建图表汇总文档"""
    summary_content = """# 论文图表汇总

## 已生成的图表文件

### 1. 性能对比图表
- `performance_comparison.png/pdf` - 基本性能对比条形图
- `error_magnitude_comparison.png/pdf` - 误差量级对比散点图  
- `task_complexity_analysis.png/pdf` - 任务复杂度vs改进幅度分析

### 2. 训练过程图表
- `training_curves_comparison.png/pdf` - 训练收敛曲线对比
- `convergence_speed_analysis.png/pdf` - 收敛速度分析
- `loss_landscape_comparison.png/pdf` - 损失景观概念图

### 3. 误差分布图表
- `2d_error_distribution.png/pdf` - 2D空间误差分布热力图
- `3d_error_slices.png/pdf` - 3D误差正交切片图
- `error_evolution.png/pdf` - 误差随时间演化分析

## 图表在论文中的使用建议

### 主要结果图 (必须包含)
1. **性能对比图** (`performance_comparison`) - 放在Results章节开头
2. **2D误差分布图** (`2d_error_distribution`) - 展示主要改进结果
3. **训练收敛曲线** (`training_curves_comparison`) - 验证训练稳定性

### 补充分析图 (根据篇幅决定)
4. **任务复杂度分析** (`task_complexity_analysis`) - 分析CFT适用性
5. **3D误差切片图** (`3d_error_slices`) - 展示高维问题处理能力
6. **误差演化分析** (`error_evolution`) - 长期预测能力分析

### 概念说明图 (可选)
7. **损失景观对比** (`loss_landscape_comparison`) - 理论解释
8. **误差量级对比** (`error_magnitude_comparison`) - 方法普适性

## 图表质量检查清单

### 技术质量
- [ ] 所有图表都是300 DPI高分辨率
- [ ] 同时提供PNG和PDF格式
- [ ] 字体大小适合论文印刷
- [ ] 颜色搭配适合黑白打印

### 内容质量  
- [ ] 数据标签清晰准确
- [ ] 图例和坐标轴标题完整
- [ ] 数值精度适当
- [ ] 改进百分比突出显示

### 论文规范
- [ ] 图表标题符合期刊要求
- [ ] 配色方案专业统一
- [ ] 布局美观平衡
- [ ] 与正文描述一致

## 下一步工作

1. **检查图表内容** - 确认所有数据准确无误
2. **调整图表样式** - 根据目标期刊要求修改格式
3. **编写图表说明** - 为每个图表准备详细的caption
4. **集成到论文** - 将图表插入到相应章节

## 图表文件位置
```
paper_preparation/figures/
├── performance_comparison.png
├── performance_comparison.pdf
├── training_curves_comparison.png
├── training_curves_comparison.pdf
├── 2d_error_distribution.png
├── 2d_error_distribution.pdf
└── ... (其他图表)
```
"""
    
    summary_file = os.path.join(os.path.dirname(__file__), '..', 'figures', 'README.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"📋 图表汇总文档已生成: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='生成论文所需的所有图表')
    parser.add_argument('--skip-errors', action='store_true', 
                       help='遇到错误时继续执行下一个脚本')
    args = parser.parse_args()
    
    # 确保figures目录存在
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("🎨 开始生成论文图表...")
    print(f"📁 图表保存目录: {os.path.abspath(figures_dir)}")
    
    # 图表生成脚本列表
    scripts = [
        ('generate_performance_comparison.py', '性能对比图表'),
        ('generate_training_curves.py', '训练曲线图表'),
        ('generate_error_visualization.py', '误差分布可视化'),
    ]
    
    # 统计结果
    success_count = 0
    total_count = len(scripts)
    
    # 逐个运行脚本
    for script_name, description in scripts:
        success = run_script(script_name, description)
        if success:
            success_count += 1
        elif not args.skip_errors:
            print(f"\n❌ 由于错误停止执行。使用 --skip-errors 参数可以跳过错误继续执行。")
            break
    
    # 生成图表汇总
    create_figure_summary()
    
    # 最终统计
    print(f"\n{'='*60}")
    print("📊 图表生成完成汇总")
    print(f"{'='*60}")
    print(f"✅ 成功生成: {success_count}/{total_count} 组图表")
    
    if success_count == total_count:
        print("🎉 所有图表生成成功！")
    else:
        print(f"⚠️  有 {total_count - success_count} 组图表生成失败")
    
    print(f"\n📁 所有图表保存在: {os.path.abspath(figures_dir)}")
    print("📋 查看 figures/README.md 了解图表使用建议")
    
    # 检查生成的文件
    print(f"\n📈 生成的图表文件:")
    if os.path.exists(figures_dir):
        files = sorted([f for f in os.listdir(figures_dir) if f.endswith(('.png', '.pdf'))])
        for f in files:
            print(f"  - {f}")
    
    print(f"\n🚀 现在可以开始撰写论文了！")

if __name__ == "__main__":
    main()
