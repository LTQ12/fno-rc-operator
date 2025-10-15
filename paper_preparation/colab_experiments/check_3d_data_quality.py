#!/usr/bin/env python3
"""
检查3D Navier-Stokes数据质量和预处理方式
数据文件: ns_V1e-4_N10000_T30.mat
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os

# 添加项目根目录到路径
sys.path.append('/content/fourier_neural_operator-master')
from utilities3 import MatReader, UnitGaussianNormalizer

def check_3d_data_quality():
    """检查3D数据的质量、形状、统计特性等"""
    
    print("🔍 3D Navier-Stokes数据质量检查")
    print("=" * 60)
    
    # 数据路径（Colab环境）
    data_path = "/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat"
    
    try:
        # 方法1：使用MatReader
        print("📂 使用MatReader加载数据...")
        reader = MatReader(data_path)
        u_field = reader.read_field('u')
        print(f"✅ 数据形状: {u_field.shape}")
        print(f"✅ 数据类型: {u_field.dtype}")
        
    except Exception as e:
        print(f"❌ MatReader加载失败: {e}")
        
        # 方法2：直接使用scipy.io
        print("\n📂 使用scipy.io加载数据...")
        try:
            data = loadmat(data_path)
            print(f"✅ 文件中的键: {list(data.keys())}")
            
            # 寻找数据字段
            for key in data.keys():
                if not key.startswith('__'):
                    field_data = data[key]
                    print(f"✅ 字段 '{key}': 形状={field_data.shape}, 类型={field_data.dtype}")
                    if 'u' in key.lower():
                        u_field = field_data
                        break
        except Exception as e2:
            print(f"❌ scipy.io加载也失败: {e2}")
            return
    
    print("\n📊 数据统计分析")
    print("-" * 40)
    
    # 基本统计
    print(f"数据维度: {len(u_field.shape)}D")
    print(f"数据形状: {u_field.shape}")
    print(f"数据大小: {u_field.size:,} 个元素")
    print(f"内存占用: {u_field.nbytes / (1024**3):.2f} GB")
    
    # 数值统计
    print(f"\n数值范围:")
    print(f"  最小值: {u_field.min():.6f}")
    print(f"  最大值: {u_field.max():.6f}")
    print(f"  均值: {u_field.mean():.6f}")
    print(f"  标准差: {u_field.std():.6f}")
    
    # 检查NaN和Inf
    nan_count = np.isnan(u_field).sum()
    inf_count = np.isinf(u_field).sum()
    print(f"\n数据质量:")
    print(f"  NaN数量: {nan_count}")
    print(f"  Inf数量: {inf_count}")
    print(f"  质量状态: {'✅ 良好' if nan_count == 0 and inf_count == 0 else '❌ 有问题'}")
    
    # 形状解析
    print(f"\n📐 维度解析:")
    if len(u_field.shape) == 4:
        N, H, W, T = u_field.shape
        print(f"  样本数 (N): {N}")
        print(f"  空间高度 (H): {H}")
        print(f"  空间宽度 (W): {W}")
        print(f"  时间步数 (T): {T}")
        
        # 检查时间演化
        print(f"\n⏰ 时间演化分析:")
        time_means = [u_field[:, :, :, t].mean() for t in range(min(10, T))]
        time_stds = [u_field[:, :, :, t].std() for t in range(min(10, T))]
        
        print("前10个时间步的统计:")
        for t in range(min(10, T)):
            print(f"  t={t}: 均值={time_means[t]:.4f}, 标准差={time_stds[t]:.4f}")
    
    # 训练/测试划分检查
    print(f"\n🔀 数据划分检查:")
    ntrain, ntest = 1000, 200
    T_in, T_out = 10, 20
    
    if u_field.shape[0] >= ntrain + ntest:
        train_a = u_field[:ntrain, ..., :T_in]
        train_u = u_field[:ntrain, ..., T_in:T_in + T_out]
        test_a = u_field[-ntest:, ..., :T_in]
        test_u = u_field[-ntest:, ..., T_in:T_in + T_out]
        
        print(f"  训练输入: {train_a.shape}")
        print(f"  训练输出: {train_u.shape}")
        print(f"  测试输入: {test_a.shape}")
        print(f"  测试输出: {test_u.shape}")
        
        # 检查数据分布一致性
        print(f"\n📈 分布一致性检查:")
        print(f"  训练输入均值: {train_a.mean():.6f}")
        print(f"  测试输入均值: {test_a.mean():.6f}")
        print(f"  训练输出均值: {train_u.mean():.6f}")
        print(f"  测试输出均值: {test_u.mean():.6f}")
        
        # 标准化检查
        print(f"\n🔧 标准化效果检查:")
        a_normalizer = UnitGaussianNormalizer(train_a)
        train_a_norm = a_normalizer.encode(train_a)
        test_a_norm = a_normalizer.encode(test_a)
        
        print(f"  标准化后训练输入: 均值={train_a_norm.mean():.6f}, 标准差={train_a_norm.std():.6f}")
        print(f"  标准化后测试输入: 均值={test_a_norm.mean():.6f}, 标准差={test_a_norm.std():.6f}")
        
        y_normalizer = UnitGaussianNormalizer(train_u)
        train_u_norm = y_normalizer.encode(train_u)
        
        print(f"  标准化后训练输出: 均值={train_u_norm.mean():.6f}, 标准差={train_u_norm.std():.6f}")
        
    else:
        print(f"  ❌ 数据不足: 总样本{u_field.shape[0]} < 需要的{ntrain + ntest}")
    
    # 可视化一个样本
    print(f"\n🎨 数据可视化检查:")
    try:
        if len(u_field.shape) == 4:
            # 选择第一个样本的第一个和最后一个时间步
            sample_0 = u_field[0, :, :, 0]
            sample_T = u_field[0, :, :, -1]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(sample_0, cmap='viridis')
            plt.title('t=0时刻')
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.imshow(sample_T, cmap='viridis')
            plt.title(f't={u_field.shape[-1]-1}时刻')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig('/content/drive/MyDrive/FNO_RC_Experiments/3d_data_visualization.png', dpi=150, bbox_inches='tight')
            print("  ✅ 可视化图已保存")
            
    except Exception as e:
        print(f"  ❌ 可视化失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 3D数据质量检查完成！")
    
    return {
        'shape': u_field.shape,
        'dtype': u_field.dtype,
        'min': u_field.min(),
        'max': u_field.max(),
        'mean': u_field.mean(),
        'std': u_field.std(),
        'nan_count': nan_count,
        'inf_count': inf_count,
        'quality': 'good' if nan_count == 0 and inf_count == 0 else 'bad'
    }

if __name__ == "__main__":
    check_3d_data_quality()
