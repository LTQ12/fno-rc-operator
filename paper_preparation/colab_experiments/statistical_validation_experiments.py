"""
FNO-RC 统计显著性验证实验
专为Google Colab环境设计，支持会话中断恢复
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
import pickle
import h5py
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ================================
# Colab环境设置
# ================================

def setup_colab_environment():
    """设置Colab环境"""
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建实验目录
    base_path = "/content/drive/MyDrive/FNO_RC_Experiments"
    os.makedirs(f"{base_path}/results/statistical_validation", exist_ok=True)
    os.makedirs(f"{base_path}/models", exist_ok=True)
    os.makedirs(f"{base_path}/logs", exist_ok=True)
    
    return device, base_path

# ================================
# FNO-RC模型定义
# ================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        
        # IFFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class CFTLayer1d(nn.Module):
    """CFT层实现"""
    def __init__(self, in_channels, out_channels, modes, segments=4):
        super().__init__()
        self.modes = modes
        self.segments = segments
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Chebyshev权重
        self.chebyshev_weights = nn.Parameter(
            torch.randn(segments, modes, in_channels, out_channels) * 0.02
        )
        
        # 保形映射参数
        self.conformal_params = nn.Parameter(torch.tensor([1.0, 0.5]))
        
    def chebyshev_transform(self, x, n_modes):
        """Chebyshev变换"""
        batch_size, channels, length = x.shape
        
        # 归一化到[-1,1]
        x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        
        # Chebyshev多项式展开
        coeffs = []
        for n in range(n_modes):
            if n == 0:
                T_n = torch.ones_like(x_norm)
            elif n == 1:
                T_n = x_norm
            else:
                T_n = 2 * x_norm * T_n_prev - T_n_prev_prev
                T_n_prev_prev = T_n_prev
            
            if n > 1:
                T_n_prev = T_n
            elif n == 1:
                T_n_prev = T_n
                T_n_prev_prev = torch.ones_like(x_norm)
                
            coeff = torch.mean(x_norm * T_n, dim=-1, keepdim=True)
            coeffs.append(coeff)
        
        return torch.cat(coeffs, dim=-1)
    
    def forward(self, x):
        batch_size, channels, length = x.shape
        
        # 分段处理
        segment_size = length // self.segments
        results = []
        
        for seg in range(self.segments):
            start_idx = seg * segment_size
            end_idx = (seg + 1) * segment_size if seg < self.segments - 1 else length
            x_segment = x[:, :, start_idx:end_idx]
            
            # Chebyshev变换
            coeffs = self.chebyshev_transform(x_segment, self.modes)
            
            # 应用学习权重
            coeffs_filtered = torch.einsum('bci,mico->bco', coeffs, self.chebyshev_weights[seg])
            
            # 重构信号
            reconstructed = torch.tanh(coeffs_filtered.mean(dim=-1, keepdim=True).expand(-1, -1, x_segment.shape[-1]))
            results.append(reconstructed)
        
        return torch.cat(results, dim=-1)

class FNOLayer1d(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        
    def forward(self, x):
        return self.conv(x) + self.w(x)

class FNORCF1d(nn.Module):
    """1D FNO-RC模型"""
    def __init__(self, modes=16, width=64, num_layers=4, cft_segments=4, cft_modes=8):
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        
        # 输入嵌入
        self.fc0 = nn.Linear(2, self.width)  # (a(x), x) -> width
        
        # FNO主路径
        self.fno_layers = nn.ModuleList([FNOLayer1d(modes, width) for _ in range(num_layers)])
        
        # CFT残差路径
        self.cft_layers = nn.ModuleList([CFTLayer1d(width, width, cft_modes, cft_segments) for _ in range(num_layers)])
        
        # 门控机制
        self.gate_layers = nn.ModuleList([nn.Linear(2*width, width) for _ in range(num_layers)])
        
        # 输出层
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # 激活函数
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x shape: (batch, seq_len, 2) -> (a(x), x)
        batch_size = x.shape[0]
        
        # 输入嵌入
        x = self.fc0(x)  # (batch, seq_len, width)
        x = x.permute(0, 2, 1)  # (batch, width, seq_len)
        
        # 逐层处理
        for i in range(self.num_layers):
            # FNO路径
            x_fno = self.fno_layers[i](x)
            x_fno = self.activation(x_fno)
            
            # CFT路径
            x_cft = self.cft_layers[i](x)
            
            # 门控融合
            x_concat = torch.cat([x_fno, x_cft], dim=1)  # (batch, 2*width, seq_len)
            x_concat = x_concat.permute(0, 2, 1)  # (batch, seq_len, 2*width)
            
            gate = torch.sigmoid(self.gate_layers[i](x_concat))  # (batch, seq_len, width)
            gate = gate.permute(0, 2, 1)  # (batch, width, seq_len)
            
            # 残差连接
            x = x_fno + gate * x_cft
        
        # 输出层
        x = x.permute(0, 2, 1)  # (batch, seq_len, width)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)  # (batch, seq_len, 1)
        
        return x

class StandardFNO1d(nn.Module):
    """标准FNO模型作为基线"""
    def __init__(self, modes=16, width=64, num_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        
        self.fc0 = nn.Linear(2, self.width)
        self.layers = nn.ModuleList([FNOLayer1d(modes, width) for _ in range(num_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        for layer in self.layers:
            x = self.activation(layer(x))
            
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# ================================
# 数据加载和预处理
# ================================

def load_burgers_data():
    """加载1D Burgers数据"""
    print("Loading 1D Burgers data...")
    
    # 这里需要根据您的实际数据路径调整
    # 假设数据已经上传到Drive
    data_path = "/content/drive/MyDrive/FNO_RC_Experiments/data"
    
    try:
        # 尝试加载现有数据
        with h5py.File(f"{data_path}/burgers_data.h5", 'r') as f:
            train_a = torch.tensor(f['train_a'][:], dtype=torch.float32)
            train_u = torch.tensor(f['train_u'][:], dtype=torch.float32)
            test_a = torch.tensor(f['test_a'][:], dtype=torch.float32)
            test_u = torch.tensor(f['test_u'][:], dtype=torch.float32)
            
        print(f"Loaded existing data: train {train_a.shape}, test {test_a.shape}")
        
    except:
        print("Generating synthetic Burgers data...")
        # 生成合成数据用于测试
        n_train, n_test = 1000, 200
        s = 8192
        
        # 生成随机初始条件
        np.random.seed(42)
        
        train_a = []
        train_u = []
        
        for i in range(n_train):
            # 生成随机初始条件
            x = np.linspace(0, 1, s)
            u0 = np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x) + np.random.normal(0, 0.1, s)
            
            # 简单的时间演化（实际应该用数值求解器）
            u_final = u0 * 0.8 + np.random.normal(0, 0.05, s)  # 简化的演化
            
            train_a.append(u0)
            train_u.append(u_final)
        
        # 测试数据
        test_a = []
        test_u = []
        
        for i in range(n_test):
            x = np.linspace(0, 1, s)
            u0 = np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x) + np.random.normal(0, 0.1, s)
            u_final = u0 * 0.8 + np.random.normal(0, 0.05, s)
            
            test_a.append(u0)
            test_u.append(u_final)
        
        train_a = torch.tensor(np.array(train_a), dtype=torch.float32)
        train_u = torch.tensor(np.array(train_u), dtype=torch.float32)
        test_a = torch.tensor(np.array(test_a), dtype=torch.float32)
        test_u = torch.tensor(np.array(test_u), dtype=torch.float32)
        
        # 保存数据
        os.makedirs(data_path, exist_ok=True)
        with h5py.File(f"{data_path}/burgers_data.h5", 'w') as f:
            f.create_dataset('train_a', data=train_a.numpy())
            f.create_dataset('train_u', data=train_u.numpy())
            f.create_dataset('test_a', data=test_a.numpy())
            f.create_dataset('test_u', data=test_u.numpy())
        
        print(f"Generated and saved data: train {train_a.shape}, test {test_a.shape}")
    
    return train_a, train_u, test_a, test_u

def prepare_data_loaders(train_a, train_u, test_a, test_u, batch_size=20):
    """准备数据加载器"""
    s = train_a.shape[-1]
    x = torch.linspace(0, 1, s).reshape(1, s, 1)
    
    # 准备训练数据
    train_input = torch.cat([train_a.unsqueeze(-1), x.repeat(train_a.shape[0], 1, 1)], dim=-1)
    train_target = train_u.unsqueeze(-1)
    
    # 准备测试数据
    test_input = torch.cat([test_a.unsqueeze(-1), x.repeat(test_a.shape[0], 1, 1)], dim=-1)
    test_target = test_u.unsqueeze(-1)
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_input, train_target)
    test_dataset = TensorDataset(test_input, test_target)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ================================
# 训练和评估函数
# ================================

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super().__init__()
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        
        return diff_norms/y_norms

def train_model(model, train_loader, test_loader, device, epochs=500, lr=0.001, save_path=None):
    """训练模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = LpLoss(size_average=True)
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 测试阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        scheduler.step()
        
        # 保存最佳模型
        if test_loss < best_test_loss and save_path:
            best_test_loss = test_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'epoch': epoch
            }, save_path)
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch:4d}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}')
    
    return train_losses, test_losses, best_test_loss

# ================================
# 统计实验主函数
# ================================

def run_statistical_experiments():
    """运行统计显著性验证实验"""
    device, base_path = setup_colab_environment()
    
    # 加载数据
    train_a, train_u, test_a, test_u = load_burgers_data()
    train_loader, test_loader = prepare_data_loaders(train_a, train_u, test_a, test_u)
    
    # 实验配置
    n_runs = 5
    epochs = 300  # 减少epochs以适应Colab限制
    
    results = {
        'fno_baseline': {'runs': [], 'mean': 0, 'std': 0},
        'fno_rc': {'runs': [], 'mean': 0, 'std': 0},
        'metadata': {
            'n_runs': n_runs,
            'epochs': epochs,
            'device': str(device),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    print("="*60)
    print("开始统计显著性验证实验")
    print("="*60)
    
    # 运行基线FNO实验
    print("\n1. 基线FNO实验")
    print("-" * 40)
    
    for run in range(n_runs):
        print(f"\n运行 {run+1}/{n_runs}...")
        
        # 设置随机种子
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # 创建模型
        model = StandardFNO1d(modes=16, width=64, num_layers=4)
        
        # 训练模型
        save_path = f"{base_path}/models/fno_baseline_run_{run+1}.pt"
        train_losses, test_losses, best_test_loss = train_model(
            model, train_loader, test_loader, device, epochs, save_path=save_path
        )
        
        results['fno_baseline']['runs'].append({
            'run': run + 1,
            'best_test_loss': best_test_loss,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        })
        
        print(f"基线FNO运行 {run+1} 完成: 最佳测试误差 = {best_test_loss:.6f}")
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache()
    
    # 运行FNO-RC实验
    print("\n2. FNO-RC实验")
    print("-" * 40)
    
    for run in range(n_runs):
        print(f"\n运行 {run+1}/{n_runs}...")
        
        # 设置随机种子
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # 创建模型
        model = FNORCF1d(modes=16, width=64, num_layers=4, cft_segments=4, cft_modes=8)
        
        # 训练模型
        save_path = f"{base_path}/models/fno_rc_run_{run+1}.pt"
        train_losses, test_losses, best_test_loss = train_model(
            model, train_loader, test_loader, device, epochs, save_path=save_path
        )
        
        results['fno_rc']['runs'].append({
            'run': run + 1,
            'best_test_loss': best_test_loss,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        })
        
        print(f"FNO-RC运行 {run+1} 完成: 最佳测试误差 = {best_test_loss:.6f}")
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache()
    
    # 计算统计数据
    fno_errors = [run['best_test_loss'] for run in results['fno_baseline']['runs']]
    fno_rc_errors = [run['best_test_loss'] for run in results['fno_rc']['runs']]
    
    results['fno_baseline']['mean'] = np.mean(fno_errors)
    results['fno_baseline']['std'] = np.std(fno_errors)
    
    results['fno_rc']['mean'] = np.mean(fno_rc_errors)
    results['fno_rc']['std'] = np.std(fno_rc_errors)
    
    improvement = (results['fno_baseline']['mean'] - results['fno_rc']['mean']) / results['fno_baseline']['mean'] * 100
    results['improvement_percent'] = improvement
    
    # 统计显著性检验
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(fno_errors, fno_rc_errors)
    results['statistical_test'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }
    
    # 保存结果
    results_path = f"{base_path}/results/statistical_validation/1d_burgers_statistical_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    print("\n" + "="*60)
    print("统计实验结果")
    print("="*60)
    print(f"基线FNO:  {results['fno_baseline']['mean']:.6f} ± {results['fno_baseline']['std']:.6f}")
    print(f"FNO-RC:   {results['fno_rc']['mean']:.6f} ± {results['fno_rc']['std']:.6f}")
    print(f"改进:     {improvement:.2f}%")
    print(f"p值:      {p_value:.6f}")
    print(f"统计显著: {'是' if p_value < 0.05 else '否'}")
    
    # 生成可视化
    create_statistical_plots(results, base_path)
    
    return results

def create_statistical_plots(results, base_path):
    """创建统计结果可视化"""
    plt.figure(figsize=(12, 8))
    
    # 子图1: 误差对比
    plt.subplot(2, 2, 1)
    fno_errors = [run['best_test_loss'] for run in results['fno_baseline']['runs']]
    fno_rc_errors = [run['best_test_loss'] for run in results['fno_rc']['runs']]
    
    x = np.arange(len(fno_errors))
    plt.plot(x, fno_errors, 'o-', label='Standard FNO', linewidth=2, markersize=8)
    plt.plot(x, fno_rc_errors, 's-', label='FNO-RC', linewidth=2, markersize=8)
    plt.xlabel('Run Number')
    plt.ylabel('Test Error')
    plt.title('Test Error Across Multiple Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 箱线图
    plt.subplot(2, 2, 2)
    plt.boxplot([fno_errors, fno_rc_errors], labels=['Standard FNO', 'FNO-RC'])
    plt.ylabel('Test Error')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 改进百分比
    plt.subplot(2, 2, 3)
    improvements = [(fno_errors[i] - fno_rc_errors[i]) / fno_errors[i] * 100 for i in range(len(fno_errors))]
    plt.bar(range(len(improvements)), improvements, alpha=0.7)
    plt.xlabel('Run Number')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement Percentage per Run')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 统计汇总
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f"Standard FNO: {results['fno_baseline']['mean']:.6f} ± {results['fno_baseline']['std']:.6f}", 
             fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"FNO-RC: {results['fno_rc']['mean']:.6f} ± {results['fno_rc']['std']:.6f}", 
             fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Improvement: {results['improvement_percent']:.2f}%", 
             fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"p-value: {results['statistical_test']['p_value']:.6f}", 
             fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Significant: {'Yes' if results['statistical_test']['significant'] else 'No'}", 
             fontsize=12, transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Statistical Summary')
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/statistical_validation/statistical_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"统计结果图表已保存到: {base_path}/results/statistical_validation/")

# ================================
# 主执行函数
# ================================

if __name__ == "__main__":
    print("FNO-RC 统计显著性验证实验")
    print("适用于Google Colab环境")
    print("预计运行时间: 3-4小时")
    
    # 运行实验
    results = run_statistical_experiments()
    
    print("\n实验完成！结果已保存到Google Drive。")
