# FNO-RC Colab实验指南

## 🚀 Colab实验环境设置

### 环境准备
```python
# 1. 检查GPU
!nvidia-smi

# 2. 安装依赖
!pip install torch torchvision torchaudio
!pip install matplotlib scipy h5py
!pip install tensorboard

# 3. 挂载Google Drive (用于数据持久化)
from google.colab import drive
drive.mount('/content/drive')
```

## 📊 实验计划

### Phase 1: 统计显著性验证
- **脚本**: `statistical_validation_experiments.py`
- **时间**: 每个维度约2-3小时
- **输出**: 5次运行的统计结果

### Phase 2: 消融实验
- **脚本**: `ablation_experiments.py`  
- **时间**: 约4-6小时
- **输出**: 各组件贡献度分析

### Phase 3: 效率和泛化实验
- **脚本**: `efficiency_and_generalization.py`
- **时间**: 约3-4小时
- **输出**: 计算效率和泛化性能数据

## ⚠️ Colab注意事项

1. **会话管理**: 每12小时会断开，需要分批运行
2. **数据保存**: 所有结果自动保存到Google Drive
3. **GPU限制**: 合理安排GPU使用时间
4. **内存管理**: 及时清理不用的变量

## 📁 文件结构
```
/content/drive/MyDrive/FNO_RC_Experiments/
├── results/
│   ├── statistical_validation/
│   ├── ablation_studies/
│   └── efficiency_analysis/
├── models/
└── logs/
```
