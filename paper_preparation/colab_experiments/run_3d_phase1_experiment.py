#!/usr/bin/env python3
"""
🚀 启动3D第一阶段对比实验
一键运行脚本，包含数据检查和模型对比
"""

print("🔬 3D Navier-Stokes 第一阶段对比实验")
print("=" * 80)
print("📋 实验内容:")
print("   1. 数据质量检查")
print("   2. FNO-3D Baseline vs B-DeepONet-3D vs FNO-RC-3D")
print("   3. 结果分析和可视化")
print("=" * 80)

# 检查环境
import sys
import os
import torch
import subprocess

# 检查CUDA
if torch.cuda.is_available():
    print(f"✅ GPU可用: {torch.cuda.get_device_name()}")
    print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  仅CPU可用，建议使用GPU加速")

# 检查数据文件
data_path = "/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat"
if os.path.exists(data_path):
    print(f"✅ 数据文件存在: {data_path}")
else:
    print(f"❌ 数据文件不存在: {data_path}")
    print("请确保数据文件已上传到Google Drive")

print("\n🔍 第一步: 数据质量检查")
print("-" * 40)

try:
    exec(open('/content/fourier_neural_operator-master/paper_preparation/colab_experiments/check_3d_data_quality.py').read())
    print("✅ 数据质量检查完成")
except Exception as e:
    print(f"❌ 数据检查失败: {e}")
    print("继续进行实验...")

print("\n🏁 第二步: 开始3方法对比实验")
print("-" * 40)
print("预计运行时间: 2-3小时")
print("请耐心等待...")

try:
    exec(open('/content/fourier_neural_operator-master/paper_preparation/colab_experiments/phase1_3d_comparison.py').read())
    print("🎉 实验完成!")
except Exception as e:
    print(f"❌ 实验执行失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("🎯 实验总结:")
print("   - 结果已保存到 /content/drive/MyDrive/FNO_RC_Experiments/phase1_3d_comparison/")
print("   - 包含训练曲线图和详细JSON结果")
print("   - 可以查看各模型的性能对比")
print("=" * 80)
