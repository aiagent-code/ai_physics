#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的贝叶斯优化脚本
直接获取数据并调用NeuralNetwork进行回归训练模型

功能:
- 直接加载训练和测试数据
- 调用NeuralNetwork进行贝叶斯优化
- 快速、直接、简洁的优化流程
"""

import pandas as pd
import numpy as np
import os
import argparse
import time
from utils.NeuralNetwork import NeuralNetwork
from utils.EnhancedPlotManager import EnhancedPlotManager

# 使用统一的数据加载方法
from data_initializer import load_data

def run_bayesian_optimization(item_name="贝叶斯优化", n_trials=10, epochs=200, batch_size=32, 
                             loss_function='huber_loss_tf', previous_results_file=None, n_repeats=2, enable_plotting=True):
    """
    运行贝叶斯优化
    
    Args:
        item_name: 项目名称
        n_trials: 优化试验次数
        epochs: 每次试验的训练轮数
        batch_size: 批次大小
        loss_function: 损失函数类型
        previous_results_file: 历史优化结果CSV文件路径
        n_repeats: 每个参数组合重复训练次数，默认为2
        enable_plotting: 是否绘制训练历史图像，默认为True
    
    Returns:
        dict: 优化结果
    """
    print(f"\n开始贝叶斯优化: {item_name}")
    print(f"参数设置:")
    print(f"  试验次数: {n_trials}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  损失函数: {loss_function}")
    if previous_results_file:
        print(f"  历史结果文件: {previous_results_file}")
    print("="*50)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_data()
    
    # 创建绘图管理器
    plot_manager = EnhancedPlotManager()
    
    # 创建神经网络实例
    nn = NeuralNetwork(plot_manager=plot_manager)
    
    # 运行贝叶斯优化
    try:
        results = nn.bayesian_optimization(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_trials=n_trials,
            epochs=epochs,
            batch_size=batch_size,
            item=item_name,
            loss_function=loss_function,
            previous_results_file=previous_results_file,
            n_repeats=n_repeats,
            enable_plotting=enable_plotting
        )
        
        print("\n" + "="*50)
        print("贝叶斯优化完成!")
        print(f"最佳参数: {results['best_params']}")
        print(f"最佳损失值: {results['best_value']:.6f}")
        
        # 从优化历史中获取最佳试验的详细信息
        if 'optimization_history' in results and results['optimization_history']:
            best_trial = min(results['optimization_history'], key=lambda x: x.get('huber_loss', float('inf')))
            if 'test_r2' in best_trial:
                print(f"最佳测试R²: {best_trial['test_r2']:.6f}")
            if 'huber_loss' in best_trial:
                print(f"最佳Huber Loss: {best_trial['huber_loss']:.6f}")
        
        return results
        
    except Exception as e:
        print(f"贝叶斯优化过程中发生错误: {e}")
        raise

def validate_parameters(n_trials, epochs, batch_size):
    """
    验证输入参数的有效性
    
    Args:
        n_trials: 试验次数
        epochs: 训练轮数
        batch_size: 批次大小
    
    Raises:
        ValueError: 参数无效时抛出异常
    """
    if n_trials <= 0:
        raise ValueError(f"试验次数必须大于0，当前值: {n_trials}")
    
    if epochs <= 0:
        raise ValueError(f"训练轮数必须大于0，当前值: {epochs}")
    
    if batch_size <= 0:
        raise ValueError(f"批次大小必须大于0，当前值: {batch_size}")
    
    if n_trials > 100:
        print(f"警告: 试验次数较大({n_trials})，可能需要很长时间")
    
    if epochs > 1000:
        print(f"警告: 训练轮数较大({epochs})，可能需要很长时间")

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='独立贝叶斯优化脚本')
    
    parser.add_argument('--item_name', type=str, default='回归模型优化',
                       help='项目名称 (默认: 回归模型优化)')
    parser.add_argument('--n_trials', type=int, default=10,
                       help='优化试验次数 (默认: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='每次试验的训练轮数 (默认: 200)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--loss_function', type=str, default='huber_loss_tf',
                       choices=['mean_squared_error', 'huber_loss_tf', 'weighted_huber_loss_tf'],
                       help='损失函数类型 (默认: huber_loss_tf)')
    parser.add_argument('--data_dir', type=str, default='raman_row',
                       help='数据目录路径 (默认: raman_row)')
    parser.add_argument('--previous_results', type=str, default='bayesian_optimization_回归模型优化.csv',
                       help='历史优化结果CSV文件路径 (默认: bayesian_optimization_回归模型优化.csv)')
    parser.add_argument('--n_repeats', type=int, default=2,
                       help='每个参数组合重复训练次数 (默认: 2)')
    parser.add_argument('--enable_plotting', action='store_true', default=False,
                       help='是否绘制训练历史图像 (默认: False，不绘制)')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    print("独立贝叶斯优化脚本")
    print("="*50)
    
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 验证参数
        validate_parameters(args.n_trials, args.epochs, args.batch_size)
        
        # 检查历史结果文件是否存在
        previous_results_file = None
        if args.previous_results and os.path.exists(args.previous_results):
            previous_results_file = args.previous_results
            print(f"找到历史结果文件: {previous_results_file}")
        elif args.previous_results:
            print(f"历史结果文件不存在: {args.previous_results}，将从头开始优化")
        
        # 配置参数
        config = {
            'item_name': args.item_name,
            'n_trials': args.n_trials,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'loss_function': args.loss_function,
            'previous_results_file': previous_results_file,
            'n_repeats': args.n_repeats,
            'enable_plotting': args.enable_plotting
        }
        
        print(f"使用参数: {config}")
        print(f"数据目录: {args.data_dir}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行贝叶斯优化
        results = run_bayesian_optimization(**config)
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 保存结果
        print("\n正在保存优化结果...")
        
        # 保存最佳参数到文件
        results_file = f"bayesian_results_{args.item_name}_{int(time.time())}.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"贝叶斯优化结果\n")
            f.write(f"="*50 + "\n")
            f.write(f"项目名称: {args.item_name}\n")
            f.write(f"试验次数: {args.n_trials}\n")
            f.write(f"训练轮数: {args.epochs}\n")
            f.write(f"批次大小: {args.batch_size}\n")
            f.write(f"损失函数: {args.loss_function}\n")
            f.write(f"总耗时: {total_time:.2f}秒\n")
            f.write(f"\n最佳参数:\n")
            for key, value in results['best_params'].items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n最佳性能:\n")
            f.write(f"  最佳损失值: {results['best_value']:.6f}\n")
            
            # 从优化历史中获取详细信息
            if 'optimization_history' in results and results['optimization_history']:
                best_trial = min(results['optimization_history'], key=lambda x: x.get('huber_loss', float('inf')))
                if 'test_r2' in best_trial:
                    f.write(f"  测试R²: {best_trial['test_r2']:.6f}\n")
                if 'huber_loss' in best_trial:
                    f.write(f"  Huber Loss: {best_trial['huber_loss']:.6f}\n")
        
        print(f"结果已保存到: {results_file}")
        print(f"总耗时: {total_time:.2f}秒")
        print("优化完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断优化过程")
    except Exception as e:
        print(f"\n优化过程发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()