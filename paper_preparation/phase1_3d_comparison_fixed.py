#!/usr/bin/env python3
"""
第一阶段3D对比实验: FNO-3D vs B-DeepONet-3D vs FNO-RC-3D
修复版本 - 解决tensor格式化问题
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 添加项目根目录到路径
import sys
sys.path.append('/content/fourier_neural_operator-master')
from utilities3 import LpLoss, count_params, MatReader, UnitGaussianNormalizer
from Adam import Adam
from chebyshev import vectorized_batched_cft

torch.manual_seed(42)
np.random.seed(42)

print("🔬 3D Navier-Stokes 第一阶段对比实验")
print("=" * 80)
print("📋 实验内容:")
print("   1. 数据质量检查")
print("   2. FNO-3D Baseline vs B-DeepONet-3D vs FNO-RC-3D")
print("   3. 结果分析和可视化")
print("=" * 80)

# 检查环境
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

# 继续执行原来的实验脚本...
exec(open('/content/phase1_3d_comparison_standalone.py').read())
