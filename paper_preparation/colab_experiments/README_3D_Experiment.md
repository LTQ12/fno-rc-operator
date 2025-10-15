# 🔬 3D Navier-Stokes 第一阶段对比实验

## 📋 实验概述

本实验对比三种神经算子方法在3D Navier-Stokes问题上的性能：
- **FNO-3D Baseline**: 标准3D傅里叶神经算子
- **B-DeepONet-3D**: 贝叶斯深度算子网络  
- **FNO-RC-3D**: 基于CFT残差修正的傅里叶神经算子

## 🎯 实验目标

验证FNO-RC在高维复杂流场中的优势，证明"**复杂度越高，CFT优势越大**"的理论假设。

## 📊 数据集

- **数据**: 3D Navier-Stokes方程 (`ns_V1e-4_N10000_T30.mat`)
- **粘性系数**: ν = 1e-4 (低粘性，湍流特征明显)
- **样本数**: 10000个样本，训练1000个，测试200个
- **时间设置**: 输入前10步，预测后20步
- **空间维度**: 2D空间 + 1D时间 → 3D问题

## 🔧 实验设置

### 模型配置
- **Fourier模式数**: 8
- **网络宽度**: 20  
- **CFT参数**: L_segments=8, M_cheb=8 (已优化)
- **训练轮数**: 200 epochs
- **批大小**: 10
- **学习率**: 0.001

### 评估指标
- **相对L2误差** (主要指标)
- **训练时间** (计算效率)
- **参数量** (模型复杂度)
- **内存使用** (资源需求)

## 🚀 运行实验

### 方法1: 一键运行
```python
# 在Colab中运行
exec(open('/content/fourier_neural_operator-master/paper_preparation/colab_experiments/run_3d_phase1_experiment.py').read())
```

### 方法2: 分步运行

#### 步骤1: 数据质量检查
```python
exec(open('/content/fourier_neural_operator-master/paper_preparation/colab_experiments/check_3d_data_quality.py').read())
```

#### 步骤2: 主要对比实验
```python
exec(open('/content/fourier_neural_operator-master/paper_preparation/colab_experiments/phase1_3d_comparison.py').read())
```

## 📈 预期结果

基于理论分析，预期结果：

| 模型 | 预期测试误差 | 相对改进 | 特点 |
|------|-------------|----------|------|
| **FNO-3D** | 基线 | - | 标准频域方法 |
| **B-DeepONet** | 可能较差 | - | 算子学习方法 |
| **FNO-RC** | **最优** | **≥50%** | CFT残差修正 |

## 📁 输出文件

实验完成后，结果保存在：
```
/content/drive/MyDrive/FNO_RC_Experiments/phase1_3d_comparison/
├── phase1_3d_comparison_results.json  # 详细数值结果
├── phase1_3d_training_curves.png      # 训练曲线图
└── 3d_data_visualization.png          # 数据可视化
```

## 📊 结果分析

### JSON结果文件包含:
- 每个模型的参数量、超参数
- 完整的训练历史（每个epoch的损失）
- 最终性能指标和训练时间
- FNO-RC相对于基线的改进百分比

### 训练曲线图包含:
- 训练损失曲线对比
- 测试损失曲线对比  
- 最终性能柱状图对比

## ⚠️ 注意事项

1. **运行时间**: 预计2-3小时（取决于GPU性能）
2. **内存需求**: 建议使用A100或V100 GPU
3. **数据路径**: 确保数据文件已上传到指定路径
4. **随机种子**: 已固定随机种子确保可重现性

## 🔍 故障排除

### 常见问题:

1. **内存不足**:
   ```python
   # 减小批大小
   batch_size = 5  # 默认10
   ```

2. **数据加载失败**:
   ```python
   # 检查数据路径
   import os
   print(os.path.exists('/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'))
   ```

3. **CUDA错误**:
   ```python
   # 检查GPU状态
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name())
   ```

## 📞 技术支持

如遇问题，请检查：
1. 数据文件路径是否正确
2. GPU内存是否充足
3. 所有依赖包是否已安装

---
**🎯 目标**: 通过这个实验证明FNO-RC在3D高复杂度问题上的显著优势，为论文投稿提供强有力的实验支撑！
