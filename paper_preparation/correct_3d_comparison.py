#!/usr/bin/env python3
"""
正确的3D对比实验 - 严格基于现有架构，统一输入输出格式
对比: FNO3d vs FNO_RC_3D vs 2024最新模型变体
核心修复：统一数据格式，确保公平对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import h5py
import json
import os
from datetime import datetime
from timeit import default_timer

# 导入现有的模型和工具
from fourier_3d_clean import FNO3d
from fourier_3d_cft_residual import FNO_RC_3D
from utilities3 import LpLoss, count_params, UnitGaussianNormalizer, MatReader
from Adam import Adam

################################################################
# 数据加载 - 与现有训练脚本完全一致
################################################################
def load_3d_data(data_path, ntrain=1000, ntest=200, T_in=10, T_out=20):
    """快速加载3D数据 - 使用h5py读取MATLAB v7.3文件"""
    print(f"📁 正在加载数据: {data_path}")
    
    try:
        print("⏳ 正在读取.mat文件 (MATLAB v7.3格式)...")
        # 使用h5py读取MATLAB v7.3格式文件
        import h5py
        with h5py.File(data_path, 'r') as f:
            u_field = f['u'][:]  # [T, H, W, N] -> [50, 64, 64, 10000]
        
        print(f"✅ 原始数据形状: {u_field.shape}")
        
        # 转换为tensor并调整格式: [T, H, W, N] -> [N, H, W, T]
        print("🔄 正在转换数据格式...")
        u_field = torch.from_numpy(u_field).float()
        u_field = u_field.permute(3, 1, 2, 0)  # [10000, 64, 64, 50]
        print(f"✅ 转换后数据形状: {u_field.shape}")
        
        # 动态确定实际可用的训练和测试样本数
        total_samples = u_field.shape[0]
        ntrain_actual = min(total_samples - 100, ntrain)  # 确保至少留100个给测试
        ntest_actual = min(total_samples - ntrain_actual, ntest)
        
        print(f"📊 总样本数: {total_samples}, 训练: {ntrain_actual}, 测试: {ntest_actual}")
        
        # 检查是否有足够的时间步
        if u_field.shape[-1] < T_in + T_out:
            print(f"❌ 时间步不足: 需要 {T_in + T_out}, 实际 {u_field.shape[-1]}")
            return None
        
        # 检查是否有足够的样本
        if ntest_actual <= 0:
            print(f"⚠️ 测试样本不足，调整参数...")
            ntest_actual = max(50, total_samples // 10)  # 至少50个测试样本
            ntrain_actual = total_samples - ntest_actual
        
        print("🔄 正在分割训练数据...")
        train_a = u_field[:ntrain_actual, ..., :T_in]
        train_u = u_field[:ntrain_actual, ..., T_in:T_in + T_out]
        
        print("🔄 正在分割测试数据...")
        test_a = u_field[-ntest_actual:, ..., :T_in]
        test_u = u_field[-ntest_actual:, ..., T_in:T_in + T_out]

        print(f"✅ 数据形状: train_a: {train_a.shape}, train_u: {train_u.shape}")
        print(f"✅ 最终样本数: ntrain={ntrain_actual}, ntest={ntest_actual}")
        
        # 清理原始数据以释放内存
        del u_field
        import gc
        gc.collect()
        
        return train_a, train_u, test_a, test_u, ntrain_actual, ntest_actual
        
    except KeyboardInterrupt:
        print("❌ 数据加载被用户中断")
        raise
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

################################################################
# 数据预处理 - 与现有脚本完全一致
################################################################
def preprocess_3d_data(train_a, train_u, test_a, test_u, T_in, T_out, device):
    """统一的数据预处理 - 为了公平对比，所有模型使用相同的目标"""
    
    S1, S2 = train_a.shape[1], train_a.shape[2]
    ntrain_actual, ntest_actual = train_a.shape[0], test_a.shape[0]
    
    # 标准化输入
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    # 统一目标：预测关键时间点 - 中间点和最后一个时间步
    # 抽取时间点：中间点(T_out//2)和最后一个时间步(T_out-1)
    mid_idx = T_out // 2
    last_idx = T_out - 1
    
    train_u_mid = train_u[..., mid_idx]  # [N, H, W] - 中间时间点
    train_u_last = train_u[..., last_idx]  # [N, H, W] - 最后时间点
    test_u_mid = test_u[..., mid_idx]  # [N, H, W]
    test_u_last = test_u[..., last_idx]  # [N, H, W]
    
    # 合并关键时间点作为目标
    train_u_target = torch.stack([train_u_mid, train_u_last], dim=-1)  # [N, H, W, 2]
    test_u_target = torch.stack([test_u_mid, test_u_last], dim=-1)  # [N, H, W, 2]
    
    # 标准化目标
    y_normalizer = UnitGaussianNormalizer(train_u_target)
    train_u_normalized = y_normalizer.encode(train_u_target)
    y_normalizer.to(device)
    
    # === FNO3d数据格式 ===
    # 输入: [N, H, W, T_out, T_in] (在时间网格上重复输入)
    train_a_fno = train_a.reshape(ntrain_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    test_a_fno = test_a.reshape(ntest_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    
    # === FNO_RC_3D数据格式 ===  
    # 输入: [N, H, W, T_out, T_in] (与原始训练脚本完全一致！)
    train_a_rc = train_a.reshape(ntrain_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    test_a_rc = test_a.reshape(ntest_actual, S1, S2, 1, T_in).repeat([1,1,1,T_out,1])
    
    print(f"预处理完成:")
    print(f"  FNO输入: {train_a_fno.shape}, 目标: {train_u_normalized.shape}")
    print(f"  FNO_RC输入: {train_a_rc.shape}, 目标: {train_u_normalized.shape}")
    
    return (train_a_fno, train_u_normalized, test_a_fno, test_u_target, 
            train_a_rc, test_a_rc, y_normalizer, S1, S2)

################################################################
# B-DeepONet 3D模型 - 简化但有效的实现
################################################################
class BDeepONet3D_Simple(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, T_in=10, T_out=1):
        super(BDeepONet3D_Simple, self).__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.width = width
        
        # Branch网络 - 处理输入函数
        self.branch_net = nn.Sequential(
            nn.Linear(T_in + 3, width),  # T_in个时间步 + 3个坐标
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width)
        )
        
        # Trunk网络 - 处理查询点
        self.trunk_net = nn.Sequential(
            nn.Linear(3, width),  # 3个坐标
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width)
        )
        
        # 输出层 - 预测完整时间序列
        self.output_layer = nn.Linear(width, T_out)

    def forward(self, x):
        # x: [B, H, W, T_out, T_in] (与FNO保持一致的输入格式)
        B, H, W, T_out_dim, T_in_dim = x.shape  # T_out_dim = T_out = 20
        
        # 生成网格坐标
        h_coords = torch.linspace(0, 1, H, device=x.device)
        w_coords = torch.linspace(0, 1, W, device=x.device)
        t_coords = torch.linspace(0, 1, T_out_dim, device=x.device)
        
        hh, ww, tt = torch.meshgrid(h_coords, w_coords, t_coords, indexing='ij')
        coords = torch.stack([hh, ww, tt], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        
        # Branch网络输入：函数值 + 坐标
        x_with_coords = torch.cat([x, coords], dim=-1)  # [B, H, W, T_out, T_in + 3]
        branch_out = self.branch_net(x_with_coords)  # [B, H, W, T_out, width]
        
        # Trunk网络输入：坐标
        trunk_out = self.trunk_net(coords)  # [B, H, W, T_out, width]
        
        # DeepONet组合
        combined = branch_out * trunk_out  # [B, H, W, T_out, width]
        output = self.output_layer(combined)  # [B, H, W, T_out, T_out]
        
        # 对角线提取 - 每个时间位置预测对应的输出
        B, H, W, T, _ = output.shape
        output_diag = torch.zeros(B, H, W, T, device=output.device)
        for t in range(T):
            output_diag[:, :, :, t] = output[:, :, :, t, t]
        
        return output_diag  # [B, H, W, T_out]

################################################################
# 训练函数 - 与现有脚本一致
################################################################
def train_model(model, model_name, train_loader, test_loader, y_normalizer, device, ntrain_actual, ntest_actual, epochs=100, mid_idx=5, last_idx=9):
    """统一的训练函数"""
    print(f"\n🔧 Training {model_name}...")
    print(f"Parameters: {count_params(model):,}")
    
    model.to(device)
    # 使用统一的学习率设置 - 与之前实验一致
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_func = LpLoss(size_average=False)
    
    train_losses = []
    test_losses = []
    
    # 早停机制
    best_test_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if model_name == 'FNO3d':
                # FNO3d: 添加网格坐标，输出关键时间点
                grid = create_grid_for_fno3d(x.shape, device)
                x_with_grid = torch.cat((x, grid), dim=-1)  # [B, H, W, T_out, 13]
                out_full = model(x_with_grid)  # [B, H, W, T_out, 1]
                out_full = out_full.squeeze(-1)  # [B, H, W, T_out]
                # 抽取关键时间点
                out = torch.stack([out_full[..., mid_idx], out_full[..., last_idx]], dim=-1)  # [B, H, W, 2]
            elif model_name == 'FNO_RC_3D':
                # FNO_RC_3D: 输出关键时间点
                out_full = model(x)  # [B, H, W, T_out, 1]
                out_full = out_full.squeeze(-1)  # [B, H, W, T_out]
                # 抽取关键时间点
                out = torch.stack([out_full[..., mid_idx], out_full[..., last_idx]], dim=-1)  # [B, H, W, 2]
                
                # 调试信息：监控CFT路径的输出范围
                if ep % 10 == 0 and batch_idx == 0:
                    print(f"  FNO_RC_3D 输出范围: min={out.min().item():.6f}, max={out.max().item():.6f}, std={out.std().item():.6f}")
            else:
                # 其他模型: 确保输出格式一致
                out = model(x)  # [B, H, W]
            
            # 正确的训练逻辑：反向传播使用标准化数据，记录使用真实尺度数据
            # 1. 反向传播损失 - 使用标准化数据保证训练稳定性
            loss_normalized = loss_func(out, y)
            loss_normalized.backward()
            optimizer.step()
            
            # 2. 记录损失 - 使用真实尺度数据，与测试保持一致
            with torch.no_grad():
                out_decoded = y_normalizer.decode(out)
                y_decoded = y_normalizer.decode(y)
                loss_real_scale = loss_func(out_decoded, y_decoded)
                train_l2 += loss_real_scale.item()
        
        scheduler.step()
        
        # 测试
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                if model_name == 'FNO3d':
                    # FNO3d: 添加网格坐标，输出关键时间点
                    grid = create_grid_for_fno3d(x.shape, device)
                    x_with_grid = torch.cat((x, grid), dim=-1)  # [B, H, W, T_out, 13]
                    out_full = model(x_with_grid)  # [B, H, W, T_out, 1]
                    out_full = out_full.squeeze(-1)  # [B, H, W, T_out]
                    # 抽取关键时间点
                    out = torch.stack([out_full[..., mid_idx], out_full[..., last_idx]], dim=-1)  # [B, H, W, 2]
                elif model_name == 'FNO_RC_3D':
                    # FNO_RC_3D: 输出关键时间点
                    out_full = model(x)  # [B, H, W, T_out, 1]
                    out_full = out_full.squeeze(-1)  # [B, H, W, T_out]
                    # 抽取关键时间点
                    out = torch.stack([out_full[..., mid_idx], out_full[..., last_idx]], dim=-1)  # [B, H, W, 2]
                else:
                    # 其他模型: 确保输出格式一致
                    out = model(x)  # [B, H, W]
                
                # 统一的测试损失计算 - 使用真实尺度数据
                out_decoded = y_normalizer.decode(out)
                y_decoded = y_normalizer.decode(y)
                test_l2 += loss_func(out_decoded, y_decoded).item()
        
        train_l2 /= ntrain_actual
        test_l2 /= ntest_actual
        
        train_losses.append(train_l2)
        test_losses.append(test_l2)
        
        # 早停检查
        if test_l2 < best_test_loss:
            best_test_loss = test_l2
            patience_counter = 0
        else:
            patience_counter += 1
            
        if ep % 5 == 0:  # 🔧 调试版本：每5个epoch输出一次
            print(f'Epoch {ep+1}/{epochs}: Train {train_l2:.6f}, Test {test_l2:.6f}')
        
        # 早停
        if patience_counter >= patience:
            print(f"🛑 早停: 测试损失在{patience}个epoch内没有改善")
            break
    
    return model, train_losses, test_losses

################################################################
# 网格生成函数 - 为FNO3d使用
################################################################
def create_grid_for_fno3d(shape, device):
    """为FNO3d创建网格坐标 - 按照现有脚本的方式"""
    B, H, W, T_dim, _ = shape  # T_dim = T_out = 20
    
    h_coords = torch.linspace(0, 1, H, device=device)
    w_coords = torch.linspace(0, 1, W, device=device)
    t_coords = torch.linspace(0, 1, T_dim, device=device)
    
    hh, ww, tt = torch.meshgrid(h_coords, w_coords, t_coords, indexing='ij')
    grid = torch.stack([hh, ww, tt], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
    
    return grid

################################################################
# 主函数
################################################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Colab环境设置
    print("🔧 设置Colab环境...")
    import sys
    sys.path.append('/content')  # 修改为正确的路径
    
    # 参数设置 - 快速测试版本
    data_path = '/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat'
    ntrain, ntest = 500, 100  # 增加样本数以获得更稳定的结果
    T_in, T_out = 10, 10  # 🔧 调整：输入10步，预测窗口10步（数据只有50个时间步）
    modes = 8
    width = 20
    batch_size = 10
    epochs = 50  # 增加epochs以充分训练CFT路径
    
    # 关键时间点索引
    mid_idx = T_out // 2  # 中间点
    last_idx = T_out - 1  # 最后时间点
    
    print("🚀 开始3D对比实验 - 充分训练版本")
    print(f"📋 实验参数: epochs={epochs}, batch_size={batch_size}")
    print(f"📊 数据参数: ntrain={ntrain}, ntest={ntest}, T_in={T_in}, T_out={T_out}")
    
    # 数据加载
    print("\n📁 步骤1: 数据加载")
    data = load_3d_data(data_path, ntrain, ntest, T_in, T_out)
    if data is None:
        return
    
    train_a, train_u, test_a, test_u, ntrain_actual, ntest_actual = data
    
    # 数据预处理
    print("\n🔄 步骤2: 数据预处理")
    processed_data = preprocess_3d_data(train_a, train_u, test_a, test_u, T_in, T_out, device)
    (train_a_fno, train_u, test_a_fno, test_u, 
     train_a_rc, test_a_rc, y_normalizer, S1, S2) = processed_data
    
    # 数据加载器 - 统一使用相同的输入格式
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a_fno, train_u), 
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a_fno, test_u), 
        batch_size=batch_size, shuffle=False
    )
    
    # 模型定义 - 统一输出单个时间步进行公平对比
    models = {
        'FNO3d': FNO3d(modes, modes, modes, width, in_dim=13, out_dim=1),  # 输出单个时间步
        'FNO_RC_3D': FNO_RC_3D(modes, modes, T_out//2 + 1, width, in_channels=T_in, out_channels=1),  # 输出单个时间步
        
        # 🔧 TODO: 确认上述两个模型正常运行后，添加2024年最新对比模型：
        # 'U-FNO': U_FNO_3D(...),           # 多相流增强版FNO
        # 'Geo-FNO': Geo_FNO_3D(...),       # 几何增强版FNO  
        # 'Nested-FNO': Nested_FNO_3D(...), # 嵌套式FNO
        # 'DeepONet-2024': DeepONet_2024_3D(...), # 2024最新DeepONet变体
    }
    
    print(f"📊 模型配置:")
    for name, model in models.items():
        print(f"  {name}: {count_params(model):,} 参数")
    
    # FNO_RC_3D现在使用正确的4维输入格式，不需要修复get_grid方法
    
    # 训练所有模型
    print("\n🏋️ 步骤4: 开始训练模型")
    results = {}
    for model_name, model in models.items():
        # 现在两个模型都使用相同的5维输入格式
        trained_model, train_losses, test_losses = train_model(
            model, model_name, train_loader, test_loader, y_normalizer, device, ntrain_actual, ntest_actual, epochs, mid_idx, last_idx
        )
        
        final_test_loss = test_losses[-1]
        results[model_name] = {
            'final_test_loss': final_test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'parameters': count_params(model)
        }
        print(f"✅ {model_name}: {final_test_loss:.6f}")
    
    # 结果对比
    print(f"\n🏆 3D实验结果 (关键时间点预测 - {epochs} epochs):")
    print("-" * 60)
    
    # 按性能排序显示
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_loss'])
    
    for i, (name, result) in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"{rank} {name}: {result['final_test_loss']:.6f} ({result['parameters']:,} 参数)")
    
    # 计算改进百分比
    if len(sorted_results) >= 2:
        best_name, best_result = sorted_results[0]
        baseline_name, baseline_result = sorted_results[1]
        improvement = (baseline_result['final_test_loss'] - best_result['final_test_loss']) / baseline_result['final_test_loss'] * 100
        print(f"\n📈 {best_name} 相对于 {baseline_name} 改进: {improvement:.2f}%")
    
    # 如果有基线结果可以对比
    # baseline_error = results.get('FNO3d', {}).get('final_test_loss', None)
    # if baseline_error:
    #     for name, result in results.items():
    #         if name != 'FNO3d':
    #             improvement = (baseline_error - result['final_test_loss']) / baseline_error * 100
    #             print(f"{name}: {result['final_test_loss']:.6f} (改进: {improvement:+.1f}%)")
    
    # 保存结果到Google Drive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = '/content/drive/MyDrive/FNO_RC_Experiments/'
    os.makedirs(results_dir, exist_ok=True)
    results_file = f'{results_dir}/final_3d_comparison_{timestamp}.json'
    
    serializable_results = {}
    for name, result in results.items():
        serializable_results[name] = {
            'final_test_loss': float(result['final_test_loss']),
            'parameters': int(result['parameters']),
            'train_losses': [float(x) for x in result['train_losses']],
            'test_losses': [float(x) for x in result['test_losses']]
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 结果已保存到: {results_file}")
    print("🎉 3D对比实验完成！")

if __name__ == "__main__":
    main()
