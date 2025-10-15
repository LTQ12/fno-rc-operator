# FNO-RC Colab实验完整指南

## 🚀 快速开始

### 1. 环境准备
```python
# 在Colab新建notebook中运行以下代码

# 检查GPU
!nvidia-smi

# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 安装依赖
!pip install torch torchvision torchaudio
!pip install matplotlib scipy h5py psutil

# 上传实验脚本到Colab
# 将所有.py文件上传到Colab Files面板
```

### 2. 一键运行所有实验
```python
# 运行完整实验套件
exec(open('run_all_experiments.py').read())
```

### 3. 单独运行实验
```python
# 统计显著性验证
exec(open('statistical_validation_experiments.py').read())

# 消融实验
exec(open('ablation_experiments.py').read())

# 效率和泛化实验
exec(open('efficiency_and_generalization.py').read())
```

## 📊 实验详情

### 实验1: 统计显著性验证
- **目的**: 验证FNO-RC相对于基线FNO的改进具有统计显著性
- **内容**: 每个模型运行5次，计算均值和标准差
- **时间**: 约3-4小时
- **输出**: 
  - 统计结果JSON文件
  - 对比图表
  - p值和显著性检验

### 实验2: 消融实验
- **目的**: 分析各组件对性能的贡献
- **内容**: 
  - CFT分段数量影响 (1, 2, 4, 8 segments)
  - Chebyshev模式数量影响 (4, 8, 16 modes)
  - 门控机制有效性验证
- **时间**: 约2-3小时
- **输出**: 
  - 消融结果JSON文件
  - 组件贡献分析图表

### 实验3: 计算效率分析
- **目的**: 详细分析计算开销和效率
- **内容**:
  - 参数量统计
  - 推理时间测量
  - 内存使用分析
  - FLOPs估算
- **时间**: 约1-2小时
- **输出**: 
  - 效率对比JSON文件
  - 性能开销分析图表

### 实验4: 泛化性能测试
- **目的**: 验证模型的泛化能力
- **内容**:
  - 不同分辨率泛化测试
  - 长期预测稳定性分析
- **时间**: 约1-2小时
- **输出**: 
  - 泛化性能JSON文件
  - 稳定性分析图表

## 📁 输出文件结构

实验完成后，在Google Drive中会生成以下文件结构：

```
/content/drive/MyDrive/FNO_RC_Experiments/
├── results/
│   ├── statistical_validation/
│   │   ├── 1d_burgers_statistical_results.json
│   │   └── statistical_comparison.png
│   ├── ablation_studies/
│   │   ├── ablation_results.json
│   │   └── ablation_analysis.png
│   ├── efficiency_analysis/
│   │   ├── efficiency_results.json
│   │   └── efficiency_comparison.png
│   └── generalization_analysis/
│       ├── generalization_results.json
│       └── stability_analysis.png
├── models/
│   ├── fno_baseline_run_1.pt
│   ├── fno_rc_run_1.pt
│   └── ... (其他模型文件)
├── logs/
│   └── experiment_log_YYYYMMDD_HHMMSS.json
├── data/
│   └── burgers_data.h5
└── FINAL_EXPERIMENT_REPORT.json
```

## ⚠️ 重要注意事项

### Colab使用限制
1. **GPU时间限制**: 连续使用12小时后会断开
2. **内存限制**: 约12-16GB RAM
3. **存储限制**: 临时存储约100GB

### 实验策略
1. **分批运行**: 建议分4次运行，每次2-3小时
2. **数据保存**: 所有结果自动保存到Google Drive
3. **会话恢复**: 支持中断后恢复运行

### 故障处理
1. **内存不足**: 重启运行时环境
2. **GPU断开**: 重新连接GPU并从中断点继续
3. **文件丢失**: 检查Google Drive连接

## 🔧 自定义实验参数

### 修改实验配置
```python
# 在脚本中修改以下参数
n_runs = 5          # 统计验证运行次数
epochs = 300        # 每次运行的训练轮数
batch_size = 20     # 批次大小
learning_rate = 0.001  # 学习率
```

### 添加新的消融实验
```python
# 在ablation_experiments.py中添加新的测试配置
new_config = {
    'name': 'Custom_Config',
    'model': ConfigurableFNORC1d(
        modes=32,           # 自定义参数
        width=128,
        cft_segments=6,
        cft_modes=12
    )
}
```

## 📊 结果解读

### 统计显著性结果
- **p < 0.05**: 改进具有统计显著性
- **改进百分比**: 相对误差减少的百分比
- **标准差**: 多次运行结果的稳定性

### 消融实验结果
- **组件贡献**: 每个组件对总体性能的贡献度
- **最佳配置**: 性能最优的参数组合
- **参数敏感性**: 参数变化对性能的影响

### 效率分析结果
- **参数开销**: 相对于基线的参数量增加
- **计算开销**: 训练和推理时间的增加
- **内存开销**: 峰值内存使用的增加

## 🎯 论文数据使用

实验完成后，您将获得以下论文所需数据：

1. **Table 1**: 主要结果对比 (均值 ± 标准差)
2. **Table 2**: 消融实验结果
3. **Table 3**: 计算效率对比
4. **Figure 1**: 性能对比图
5. **Figure 2**: 消融分析图
6. **Figure 3**: 效率分析图
7. **Figure 4**: 泛化性能图

所有数据都包含统计显著性验证，满足顶级期刊要求。

## 🆘 技术支持

如果遇到问题，请检查：

1. **日志文件**: 查看详细错误信息
2. **GPU状态**: 确保GPU可用
3. **Drive连接**: 确保Google Drive正常挂载
4. **依赖安装**: 确保所有包都已正确安装

实验完成后，您将拥有投稿Neural Networks期刊所需的全部数据！
