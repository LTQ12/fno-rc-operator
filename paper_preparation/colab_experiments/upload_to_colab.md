# 📤 将实验文件上传到Colab

## 🚨 当前问题
文件路径不存在，需要将本地创建的实验文件上传到Colab环境。

## 📁 需要上传的文件

### 1. 数据质量检查脚本
**文件**: `check_3d_data_quality.py`
**本地路径**: `/Users/liutaiqian/Downloads/fourier_neural_operator-master/paper_preparation/colab_experiments/check_3d_data_quality.py`
**Colab目标路径**: `/content/fourier_neural_operator-master/paper_preparation/colab_experiments/`

### 2. 3D基线FNO实现
**文件**: `fourier_3d_baseline_reliable.py`
**本地路径**: `/Users/liutaiqian/Downloads/fourier_neural_operator-master/paper_preparation/colab_experiments/fourier_3d_baseline_reliable.py`
**Colab目标路径**: `/content/fourier_neural_operator-master/paper_preparation/colab_experiments/`

### 3. B-DeepONet 3D实现
**文件**: `b_deeponet_3d.py`
**本地路径**: `/Users/liutaiqian/Downloads/fourier_neural_operator-master/paper_preparation/colab_experiments/b_deeponet_3d.py`
**Colab目标路径**: `/content/fourier_neural_operator-master/paper_preparation/colab_experiments/`

### 4. 第一阶段对比实验
**文件**: `phase1_3d_comparison.py`
**本地路径**: `/Users/liutaiqian/Downloads/fourier_neural_operator-master/paper_preparation/colab_experiments/phase1_3d_comparison.py`
**Colab目标路径**: `/content/fourier_neural_operator-master/paper_preparation/colab_experiments/`

### 5. 一键运行脚本
**文件**: `run_3d_phase1_experiment.py`
**本地路径**: `/Users/liutaiqian/Downloads/fourier_neural_operator-master/paper_preparation/colab_experiments/run_3d_phase1_experiment.py`
**Colab目标路径**: `/content/fourier_neural_operator-master/paper_preparation/colab_experiments/`

## 🔧 上传方法

### 方法1: 通过Colab文件管理器上传
1. 在Colab左侧点击文件夹图标
2. 导航到 `/content/fourier_neural_operator-master/paper_preparation/`
3. 创建 `colab_experiments` 文件夹
4. 将上述5个文件拖拽上传到该文件夹

### 方法2: 使用代码上传
在Colab中运行以下代码创建目录：

```python
import os
os.makedirs('/content/fourier_neural_operator-master/paper_preparation/colab_experiments/', exist_ok=True)
print("✅ 目录已创建")
```

然后手动上传文件到该目录。

### 方法3: 通过Google Drive同步
1. 将文件上传到Google Drive的某个文件夹
2. 在Colab中挂载Drive并复制文件：

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
import os

# 创建目标目录
os.makedirs('/content/fourier_neural_operator-master/paper_preparation/colab_experiments/', exist_ok=True)

# 从Drive复制文件（假设您上传到了Drive的experiment_files文件夹）
files_to_copy = [
    'check_3d_data_quality.py',
    'fourier_3d_baseline_reliable.py', 
    'b_deeponet_3d.py',
    'phase1_3d_comparison.py',
    'run_3d_phase1_experiment.py'
]

for file in files_to_copy:
    src = f'/content/drive/MyDrive/experiment_files/{file}'
    dst = f'/content/fourier_neural_operator-master/paper_preparation/colab_experiments/{file}'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✅ 已复制: {file}")
    else:
        print(f"❌ 文件不存在: {src}")
```

## ✅ 验证上传成功

上传完成后，在Colab中运行以下代码验证：

```python
import os

files_to_check = [
    '/content/fourier_neural_operator-master/paper_preparation/colab_experiments/check_3d_data_quality.py',
    '/content/fourier_neural_operator-master/paper_preparation/colab_experiments/fourier_3d_baseline_reliable.py',
    '/content/fourier_neural_operator-master/paper_preparation/colab_experiments/b_deeponet_3d.py', 
    '/content/fourier_neural_operator-master/paper_preparation/colab_experiments/phase1_3d_comparison.py',
    '/content/fourier_neural_operator-master/paper_preparation/colab_experiments/run_3d_phase1_experiment.py'
]

print("📁 文件检查结果:")
for file in files_to_check:
    status = "✅ 存在" if os.path.exists(file) else "❌ 缺失"
    print(f"   {os.path.basename(file)}: {status}")
```

## 🚀 上传完成后重新运行

文件上传成功后，重新运行启动脚本：

```python
exec(open('/content/fourier_neural_operator-master/paper_preparation/colab_experiments/run_3d_phase1_experiment.py').read())
```

---
**注意**: 请确保所有5个文件都成功上传到Colab环境中，否则实验无法正常运行。
