#!/usr/bin/env python3
"""
第一阶段3D对比实验: FNO-3D vs B-DeepONet-3D vs FNO-RC-3D
数据: 3D Navier-Stokes (ns_V1e-4_N10000_T30.mat)
目标: 验证FNO-RC在高维复杂问题上的优势
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 导入项目模块
import sys
sys.path.append('/content/fourier_neural_operator-master')

# 导入模型
from fourier_3d_cft_residual import FNO_RC_3D
from fourier_3d_baseline_reliable import FNO3d_Baseline
from b_deeponet_3d import BDeepONet3D_Simplified

# 导入工具
from utilities3 import LpLoss, count_params, MatReader, UnitGaussianNormalizer
from Adam import Adam

torch.manual_seed(42)  # 使用不同的随机种子确保公平性
np.random.seed(42)

class ExperimentTracker:
    """实验跟踪器"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.results = {}
        self.start_time = datetime.now()
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
    def log_model_info(self, model_name, model, args):
        """记录模型信息"""
        self.results[model_name] = {
            'model_info': {
                'parameters': count_params(model),
                'modes': args.modes,
                'width': args.width,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs,
                'batch_size': args.batch_size
            },
            'training_history': {
                'train_loss': [],
                'test_loss': [],
                'epoch_times': []
            },
            'final_results': {}
        }
        
    def log_epoch(self, model_name, epoch, train_loss, test_loss, epoch_time):
        """记录每个epoch的结果"""
        self.results[model_name]['training_history']['train_loss'].append(train_loss)
        self.results[model_name]['training_history']['test_loss'].append(test_loss)
        self.results[model_name]['training_history']['epoch_times'].append(epoch_time)
        
    def log_final_results(self, model_name, final_train_loss, final_test_loss, total_time):
        """记录最终结果"""
        self.results[model_name]['final_results'] = {
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'total_training_time': total_time,
            'avg_epoch_time': np.mean(self.results[model_name]['training_history']['epoch_times'])
        }
        
    def save_results(self):
        """保存结果到文件"""
        # 计算相对改进
        if 'FNO_Baseline' in self.results and 'FNO_RC' in self.results:
            baseline_error = self.results['FNO_Baseline']['final_results']['final_test_loss']
            fno_rc_error = self.results['FNO_RC']['final_results']['final_test_loss']
            improvement = (baseline_error - fno_rc_error) / baseline_error * 100
            self.results['comparison'] = {
                'fno_rc_vs_baseline_improvement': f"{improvement:.2f}%"
            }
        
        # 保存JSON结果
        results_file = os.path.join(self.save_dir, 'phase1_3d_comparison_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\n📊 实验结果已保存到: {results_file}")
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))
        
        # 训练损失
        plt.subplot(1, 3, 1)
        for model_name in self.results.keys():
            if model_name != 'comparison':
                train_losses = self.results[model_name]['training_history']['train_loss']
                epochs = range(1, len(train_losses) + 1)
                plt.plot(epochs, train_losses, label=model_name, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training L2 Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 测试损失
        plt.subplot(1, 3, 2)
        for model_name in self.results.keys():
            if model_name != 'comparison':
                test_losses = self.results[model_name]['training_history']['test_loss']
                epochs = range(1, len(test_losses) + 1)
                plt.plot(epochs, test_losses, label=model_name, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Test L2 Loss')
        plt.title('Test Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 最终对比
        plt.subplot(1, 3, 3)
        model_names = []
        test_errors = []
        for model_name in self.results.keys():
            if model_name != 'comparison':
                model_names.append(model_name)
                test_errors.append(self.results[model_name]['final_results']['final_test_loss'])
        
        bars = plt.bar(model_names, test_errors, 
                      color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        plt.xlabel('Model')
        plt.ylabel('Final Test L2 Loss')
        plt.title('Final Performance Comparison')
        plt.yscale('log')
        
        # 添加数值标签
        for bar, error in zip(bars, test_errors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{error:.6f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(self.save_dir, 'phase1_3d_training_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 训练曲线已保存到: {plot_file}")
        plt.close()

def train_model(model, model_name, train_loader, test_loader, args, device, tracker):
    """训练单个模型"""
    print(f"\n🚀 开始训练 {model_name}")
    print("=" * 60)
    
    # 记录模型信息
    tracker.log_model_info(model_name, model, args)
    
    # 优化器和调度器
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    loss_func = LpLoss(size_average=False)
    
    print(f"📊 模型参数量: {count_params(model):,}")
    print(f"🔧 超参数: modes={args.modes}, width={args.width}, lr={args.learning_rate}")
    
    # 训练循环
    model_start_time = default_timer()
    
    for ep in range(args.epochs):
        model.train()
        epoch_start = default_timer()
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(x).squeeze(-1)  # 移除最后的channel维度
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        
        # 测试
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze(-1)
                
                # 解码进行公平比较
                out_decoded = y_normalizer.decode(out)
                y_decoded = y_normalizer.decode(y)
                test_l2 += loss_func(out_decoded, y_decoded).item()
        
        # 归一化损失
        train_l2 /= args.ntrain
        test_l2 /= args.ntest
        
        epoch_time = default_timer() - epoch_start
        
        # 记录结果
        tracker.log_epoch(model_name, ep + 1, train_l2, test_l2, epoch_time)
        
        # 打印进度
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1:3d}/{args.epochs} | Time: {epoch_time:.2f}s | "
                  f"Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}")
    
    total_time = default_timer() - model_start_time
    
    # 记录最终结果
    final_train_loss = tracker.results[model_name]['training_history']['train_loss'][-1]
    final_test_loss = tracker.results[model_name]['training_history']['test_loss'][-1]
    tracker.log_final_results(model_name, final_train_loss, final_test_loss, total_time)
    
    print(f"✅ {model_name} 训练完成!")
    print(f"   最终测试误差: {final_test_loss:.6f}")
    print(f"   总训练时间: {total_time:.1f}秒")
    
    return model

def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='Phase 1: 3D Comparison Experiment')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                       default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--save_dir', type=str, 
                       default='/content/drive/MyDrive/FNO_RC_Experiments/phase1_3d_comparison')
    
    # 实验参数
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--ntest', type=int, default=200)
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_out', type=int, default=20)
    
    # 模型参数
    parser.add_argument('--modes', type=int, default=8)
    parser.add_argument('--width', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)  # 减少epochs加快实验
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_step', type=int, default=100)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 初始化实验跟踪器
    tracker = ExperimentTracker(args.save_dir)
    
    print("\n" + "=" * 80)
    print("🧪 第一阶段3D对比实验 - FNO vs B-DeepONet vs FNO-RC")
    print("=" * 80)
    print(f"📊 实验设置:")
    print(f"   数据集: 3D Navier-Stokes (ν=1e-4)")
    print(f"   训练样本: {args.ntrain}, 测试样本: {args.ntest}")
    print(f"   输入时间步: {args.T_in}, 预测时间步: {args.T_out}")
    print(f"   训练轮数: {args.epochs}")
    print("=" * 80)
    
    # 数据加载
    print("\n📂 加载3D Navier-Stokes数据...")
    try:
        reader = MatReader(args.data_path)
        u_field = reader.read_field('u')
        print(f"✅ 数据形状: {u_field.shape}")
        
        # 数据划分
        train_a = u_field[:args.ntrain, ..., :args.T_in]
        train_u = u_field[:args.ntrain, ..., args.T_in:args.T_in + args.T_out]
        test_a = u_field[-args.ntest:, ..., :args.T_in]
        test_u = u_field[-args.ntest:, ..., args.T_in:args.T_in + args.T_out]
        
        print(f"   训练输入: {train_a.shape}")
        print(f"   训练输出: {train_u.shape}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 数据预处理
    print("\n🔧 数据预处理...")
    global y_normalizer  # 用于测试时解码
    
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)
    
    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)
    y_normalizer.to(device)
    
    # 重塑数据以匹配模型输入格式
    S1, S2 = train_a.shape[1], train_a.shape[2]
    train_a = train_a.reshape(args.ntrain, S1, S2, 1, args.T_in).repeat([1,1,1,args.T_out,1])
    test_a = test_a.reshape(args.ntest, S1, S2, 1, args.T_in).repeat([1,1,1,args.T_out,1])
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), 
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), 
        batch_size=args.batch_size, shuffle=False
    )
    
    print("✅ 数据预处理完成")
    
    # 模型定义
    models = {
        'FNO_Baseline': FNO3d_Baseline(
            args.modes, args.modes, args.modes, args.width, 
            in_channels=args.T_in, out_channels=1
        ).to(device),
        
        'B_DeepONet': BDeepONet3D_Simplified(
            args.modes, args.modes, args.modes, args.width,
            in_channels=args.T_in, out_channels=1
        ).to(device),
        
        'FNO_RC': FNO_RC_3D(
            args.modes, args.modes, args.modes, args.width,
            in_channels=args.T_in, out_channels=1
        ).to(device)
    }
    
    # 训练所有模型
    trained_models = {}
    for model_name, model in models.items():
        trained_models[model_name] = train_model(
            model, model_name, train_loader, test_loader, args, device, tracker
        )
    
    # 保存结果和绘制图表
    tracker.save_results()
    tracker.plot_training_curves()
    
    # 输出最终对比结果
    print("\n" + "=" * 80)
    print("🏆 第一阶段实验结果总结")
    print("=" * 80)
    
    results_summary = []
    for model_name in ['FNO_Baseline', 'B_DeepONet', 'FNO_RC']:
        if model_name in tracker.results:
            result = tracker.results[model_name]['final_results']
            results_summary.append({
                'model': model_name,
                'test_error': result['final_test_loss'],
                'params': tracker.results[model_name]['model_info']['parameters'],
                'time': result['total_training_time']
            })
    
    # 排序并显示
    results_summary.sort(key=lambda x: x['test_error'])
    
    print(f"{'模型':<15} {'测试误差':<12} {'参数量':<10} {'训练时间(s)':<12} {'相对改进'}")
    print("-" * 70)
    
    baseline_error = None
    for i, result in enumerate(results_summary):
        if result['model'] == 'FNO_Baseline':
            baseline_error = result['test_error']
            improvement = "基线"
        else:
            if baseline_error:
                improvement = f"{(baseline_error - result['test_error']) / baseline_error * 100:+.1f}%"
            else:
                improvement = "N/A"
        
        print(f"{result['model']:<15} {result['test_error']:<12.6f} {result['params']:<10,} "
              f"{result['time']:<12.1f} {improvement}")
    
    print("=" * 80)
    print("🎉 第一阶段实验完成！")

if __name__ == "__main__":
    main()
