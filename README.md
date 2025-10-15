# FNO-RC: Fourier Neural Operator with Conformal Fourier Transform Residual Correction

本仓库包含 FNO-RC（FNO + CFT 残差校正）的代码、实验脚本、论文与图表。对应论文 LaTeX 与 PDF 在 `paper_preparation/` 下。

- 论文主稿：`paper_preparation/fno_rc_nmi_v5_revised.pdf`
- 结构图和实验图：`paper_preparation/figures/`

## 安装

```bash
git clone git@github.com:LTQ12/fno-rc-operator.git
cd fno-rc-operator
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r paper_preparation/colab_experiments/requirements.txt  # 如无此文件，可直接 pip install torch numpy scipy matplotlib h5py tqdm
```

PyTorch 版本建议 >= 1.12（支持新的复数 FFT 内核）。

## 数据下载

数据较大，未随仓库提交（.gitignore 已排除 *.mat/*.pt）。你可以：
- 使用 `data_download/download_ns.py` 或你自己的路径放置数据；
- 或者在训练脚本中通过 `--data_path` 指定外部路径。

示例（Navier–Stokes 3D）：
```
/content/data/ns_V1e-4_N10000_T30.mat  # Colab 路径示例
```

## 训练与评测（3D 主实验）

- 标准 FNO（多步序列）：
```bash
python train_fno_ns_3d.py \
  --data_path /path/to/ns_V1e-4_N10000_T30.mat \
  --T_in 10 --T_out 20 --modes 8 --width 20 \
  --epochs 50 --batch_size 20
```

- FNO-RC（带 CFT 残差）：
```bash
python train_cft_residual_ns_3d.py \
  --data_path /path/to/ns_V1e-4_N10000_T30.mat \
  --T_in 10 --T_out 20 --modes 6 --width 20 \
  --num_correction_layers 1 --cft_L 4 --cft_M 6 \
  --epochs 60 --batch_size 10 \
  --rc_time_smooth_weight 3e-3 --hf_reg_weight 5e-4
```

- 跨分辨率评测：
```bash
python eval_cross_resolution.py \
  --fno_path /path/to/fno_3d_standard.pt \
  --fno_rc_path /path/to/fno_rc_3d_seq_multires.pt \
  --T_in 10 --T_out 20 --resample_mode spectral
```

- 长时域 rollout（自回归 100 步）：
```bash
python eval_long_rollout.py \
  --fno_path /path/to/fno_3d_standard.pt \
  --fno_rc_path /path/to/fno_rc_3d_seq_multires.pt \
  --T_in 10 --rollout_T 100
```

- 频谱分析（能量/幅度/相位）：
```bash
python analyze_spectrum.py \
  --fno_path /path/to/fno_3d_standard.pt \
  --fno_rc_path /path/to/fno_rc_3d_seq_multires.pt \
  --T_in 10 --T_out 20 --save_plot
```

以上脚本默认启用滑窗、绝对时间通道、统一归一化与统一误差度量，确保公平对比。

## 可复现性
- 固定随机种子：2025（PyTorch/NumPy/Python）；
- 结果报告通常为多窗口/多序列的 mean±std；
- 训练与评测命令可在 `paper_preparation/colab_experiments/` 查找示例与注释脚本；
- 模型权重可通过训练脚本保存到 `models/` 目录（已被 .gitignore 排除），请自行上传或发布 Release。

## 项目结构
- `fourier_3d_clean.py` / `fourier_3d_cft_residual.py`：3D FNO 与 FNO‑RC 核心模型；
- `train_*_3d.py`：各基线与 FNO‑RC 的训练脚本；
- `eval_cross_resolution.py` / `eval_long_rollout.py` / `analyze_spectrum.py`：评测与诊断；
- `paper_preparation/`：论文与图表、编译脚本、参考文献。

## 许可
MIT

## 引用
若本项目对你有帮助，请引用：

- FNO 原文（供背景对比）：
```
@misc{li2020fourier,
  title={Fourier Neural Operator for Parametric Partial Differential Equations},
  author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
  year={2020}, eprint={2010.08895}, archivePrefix={arXiv}, primaryClass={cs.LG}
}
```

- FNO‑RC（本仓库对应论文，见 `paper_preparation`）：
```
@article{fno_rc_2025,
  title={Fourier Neural Operator with Conformal Fourier Transform Residual Correction for Partial Differential Equations},
  author={Liu, Taiqian and Liu, Lijun},
  journal={Manuscript},
  year={2025}
}
```
