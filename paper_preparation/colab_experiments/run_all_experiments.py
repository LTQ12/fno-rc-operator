"""
FNO-RC 完整实验套件
一键运行所有补充实验，适合Colab环境
"""

import os
import sys
import json
import time
from datetime import datetime
import traceback

def setup_experiment_environment():
    """设置实验环境"""
    print("="*80)
    print("FNO-RC 完整实验套件")
    print("适用于Google Colab环境")
    print("="*80)
    
    # 检查GPU
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 挂载Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive 已挂载")
    except:
        print("⚠️ 非Colab环境或Drive挂载失败")
    
    # 创建实验目录
    base_path = "/content/drive/MyDrive/FNO_RC_Experiments"
    os.makedirs(f"{base_path}/logs", exist_ok=True)
    
    return device, base_path

def run_experiment_with_logging(experiment_func, experiment_name, log_path):
    """运行实验并记录日志"""
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行实验
        results = experiment_func()
        
        # 记录成功
        end_time = time.time()
        duration = end_time - start_time
        
        log_entry = {
            'experiment': experiment_name,
            'status': 'SUCCESS',
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'duration_seconds': duration,
            'duration_formatted': f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s",
            'results_summary': str(results)[:500] if results else "No results returned"
        }
        
        print(f"\n✅ 实验 {experiment_name} 成功完成")
        print(f"⏱️ 耗时: {log_entry['duration_formatted']}")
        
        return log_entry, results
        
    except Exception as e:
        # 记录失败
        end_time = time.time()
        duration = end_time - start_time
        
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        
        log_entry = {
            'experiment': experiment_name,
            'status': 'FAILED',
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'duration_seconds': duration,
            'duration_formatted': f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s",
            'error_message': error_msg,
            'traceback': traceback_msg
        }
        
        print(f"\n❌ 实验 {experiment_name} 失败")
        print(f"⏱️ 耗时: {log_entry['duration_formatted']}")
        print(f"错误: {error_msg}")
        
        return log_entry, None

def save_experiment_log(log_entries, base_path):
    """保存实验日志"""
    log_file = f"{base_path}/logs/experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(log_file, 'w') as f:
        json.dump({
            'experiment_suite': 'FNO-RC Complete Experiments',
            'total_experiments': len(log_entries),
            'successful_experiments': sum(1 for entry in log_entries if entry['status'] == 'SUCCESS'),
            'failed_experiments': sum(1 for entry in log_entries if entry['status'] == 'FAILED'),
            'total_duration': sum(entry['duration_seconds'] for entry in log_entries),
            'experiments': log_entries
        }, f, indent=2)
    
    print(f"\n📝 实验日志已保存: {log_file}")

def main():
    """主执行函数"""
    device, base_path = setup_experiment_environment()
    
    # 实验计划
    experiments = [
        {
            'name': 'Statistical Validation',
            'description': '统计显著性验证实验 (5次运行)',
            'estimated_time': '3-4小时',
            'function': None,  # 将在运行时导入
            'module': 'statistical_validation_experiments',
            'func_name': 'run_statistical_experiments'
        },
        {
            'name': 'Ablation Studies',
            'description': 'CFT参数和门控机制消融实验',
            'estimated_time': '2-3小时',
            'function': None,
            'module': 'ablation_experiments',
            'func_name': 'run_ablation_experiments'
        },
        {
            'name': 'Efficiency Analysis',
            'description': '计算效率和泛化性能分析',
            'estimated_time': '2-3小时',
            'function': None,
            'module': 'efficiency_and_generalization',
            'func_name': 'run_efficiency_experiments'
        },
        {
            'name': 'Generalization Tests',
            'description': '分辨率泛化和长期预测稳定性',
            'estimated_time': '1-2小时',
            'function': None,
            'module': 'efficiency_and_generalization',
            'func_name': 'run_generalization_experiments'
        }
    ]
    
    print(f"\n📋 实验计划 (共{len(experiments)}个实验):")
    total_estimated_time = 0
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   描述: {exp['description']}")
        print(f"   预计时间: {exp['estimated_time']}")
        
        # 粗略计算总时间（取中值）
        time_str = exp['estimated_time']
        if '-' in time_str:
            times = time_str.split('-')
            avg_time = (float(times[0]) + float(times[1].split('小')[0])) / 2
        else:
            avg_time = float(time_str.split('小')[0])
        total_estimated_time += avg_time
    
    print(f"\n⏰ 预计总时间: {total_estimated_time:.1f}小时")
    
    # 用户确认
    response = input(f"\n是否开始运行所有实验？(y/N): ")
    if response.lower() != 'y':
        print("实验取消。")
        return
    
    # 记录开始时间
    suite_start_time = time.time()
    log_entries = []
    results_summary = {}
    
    # 逐个运行实验
    for exp in experiments:
        try:
            # 动态导入实验模块
            if exp['module'] == 'statistical_validation_experiments':
                from statistical_validation_experiments import run_statistical_experiments as func
            elif exp['module'] == 'ablation_experiments':
                from ablation_experiments import run_ablation_experiments as func
            elif exp['module'] == 'efficiency_and_generalization':
                if exp['func_name'] == 'run_efficiency_experiments':
                    from efficiency_and_generalization import run_efficiency_experiments as func
                else:
                    from efficiency_and_generalization import run_generalization_experiments as func
            
            # 运行实验
            log_entry, results = run_experiment_with_logging(
                func, exp['name'], f"{base_path}/logs"
            )
            
            log_entries.append(log_entry)
            if results:
                results_summary[exp['name']] = results
                
        except ImportError as e:
            print(f"❌ 无法导入实验模块 {exp['module']}: {e}")
            log_entry = {
                'experiment': exp['name'],
                'status': 'FAILED',
                'error_message': f"Import error: {e}",
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0
            }
            log_entries.append(log_entry)
        
        except Exception as e:
            print(f"❌ 运行实验 {exp['name']} 时出错: {e}")
            log_entry = {
                'experiment': exp['name'],
                'status': 'FAILED',
                'error_message': str(e),
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0
            }
            log_entries.append(log_entry)
        
        # 清理GPU内存
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n💾 中间结果已保存，继续下一个实验...")
    
    # 计算总体统计
    suite_end_time = time.time()
    total_duration = suite_end_time - suite_start_time
    
    successful_experiments = sum(1 for entry in log_entries if entry['status'] == 'SUCCESS')
    failed_experiments = len(log_entries) - successful_experiments
    
    # 保存完整日志
    save_experiment_log(log_entries, base_path)
    
    # 生成最终报告
    generate_final_report(log_entries, results_summary, total_duration, base_path)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("实验套件完成总结")
    print(f"{'='*80}")
    print(f"✅ 成功完成: {successful_experiments}/{len(experiments)} 个实验")
    print(f"❌ 失败: {failed_experiments}/{len(experiments)} 个实验")
    print(f"⏱️ 总耗时: {total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s")
    print(f"📁 结果保存位置: {base_path}")
    
    if successful_experiments == len(experiments):
        print(f"\n🎉 所有实验成功完成！论文数据已准备就绪。")
    else:
        print(f"\n⚠️ 部分实验失败，请检查日志文件。")

def generate_final_report(log_entries, results_summary, total_duration, base_path):
    """生成最终实验报告"""
    report = {
        'experiment_suite_summary': {
            'total_experiments': len(log_entries),
            'successful_experiments': sum(1 for entry in log_entries if entry['status'] == 'SUCCESS'),
            'failed_experiments': sum(1 for entry in log_entries if entry['status'] == 'FAILED'),
            'total_duration_seconds': total_duration,
            'total_duration_formatted': f"{total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s",
            'completion_time': datetime.now().isoformat()
        },
        'experiment_logs': log_entries,
        'results_summary': results_summary,
        'paper_readiness_checklist': {
            'statistical_validation': any(entry['experiment'] == 'Statistical Validation' and entry['status'] == 'SUCCESS' for entry in log_entries),
            'ablation_studies': any(entry['experiment'] == 'Ablation Studies' and entry['status'] == 'SUCCESS' for entry in log_entries),
            'efficiency_analysis': any(entry['experiment'] == 'Efficiency Analysis' and entry['status'] == 'SUCCESS' for entry in log_entries),
            'generalization_tests': any(entry['experiment'] == 'Generalization Tests' and entry['status'] == 'SUCCESS' for entry in log_entries)
        }
    }
    
    # 保存报告
    report_file = f"{base_path}/FINAL_EXPERIMENT_REPORT.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 最终实验报告已保存: {report_file}")

if __name__ == "__main__":
    main()
