#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从贝叶斯优化结果中提取最佳参数并进行训练

功能:
1. 从CSV文件中读取贝叶斯优化结果
2. 找到最佳参数（基于最低Huber损失）
3. 使用最佳参数调用NeuralNetwork.train_model进行训练
4. 保存训练结果和模型
"""

import pandas as pd
import numpy as np
import argparse
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.NeuralNetwork import NeuralNetwork
from utils.EnhancedPlotManager import EnhancedPlotManager

# 使用统一的数据加载方法
from data_initializer import load_data

def load_best_params(csv_file):
    """
    从CSV文件中加载最佳参数
    
    Args:
        csv_file: CSV文件路径
    
    Returns:
        dict: 最佳参数字典
    """
    print(f"正在从 {csv_file} 读取贝叶斯优化结果...")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"找到 {len(df)} 个优化试验结果")
        
        # 找到最佳试验（最低Huber损失）
        best_idx = df['huber_loss'].idxmin()
        best_row = df.loc[best_idx]
        
        print(f"\n最佳试验结果 (Trial {best_row['trial']}):")
        print(f"  训练R²: {best_row['train_r2']:.6f}")
        print(f"  测试R²: {best_row['test_r2']:.6f}")
        print(f"  Huber损失: {best_row['huber_loss']:.6f}")
        
        # 提取参数
        best_params = {
            'DenseN': int(best_row['DenseN']),
            'DropoutR': float(best_row['DropoutR']),
            'C1_K': int(best_row['C1_K']),
            'C1_S': int(best_row['C1_S']),
            'C2_K': int(best_row['C2_K']),
            'C2_S': int(best_row['C2_S'])
        }
        
        print(f"\n最佳参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        return best_params, best_row
        
    except Exception as e:
        raise RuntimeError(f"读取CSV文件失败: {str(e)}")

def train_with_best_params(X_train, y_train, X_test, y_test, best_params, 
                          epochs=200, batch_size=32, item_name="最佳参数训练",
                          loss_function='huber_loss_tf'):
    """
    使用最佳参数进行训练
    
    Args:
        X_train, y_train, X_test, y_test: 训练和测试数据
        best_params: 最佳参数字典
        epochs: 训练轮数
        batch_size: 批次大小
        item_name: 项目名称
        loss_function: 损失函数类型
    
    Returns:
        dict: 训练结果
    """
    print(f"\n开始使用最佳参数进行训练...")
    print(f"训练配置:")
    print(f"  轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  损失函数: {loss_function}")
    print(f"  输入数据形状: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  测试数据形状: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # 创建绘图管理器
    plot_manager = EnhancedPlotManager()
    
    # 创建神经网络实例
    nn = NeuralNetwork(plot_manager=plot_manager)
    
    # 开始训练
    start_time = time.time()
    
    results = nn.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_params=best_params,
        epochs=epochs,
        batch_size=batch_size,
        item=item_name,
        save_path=f"models/best_params_model_{int(time.time())}.h5",
        loss_function=loss_function
    )
    
    training_time = time.time() - start_time
    
    print(f"\n训练完成! 总耗时: {training_time:.2f}秒")
    print(f"最终结果:")
    print(f"  训练集R²: {results['train_metrics']['r2']:.6f}")
    print(f"  测试集R²: {results['test_metrics']['r2']:.6f}")
    print(f"  测试集Huber损失: {results['test_metrics']['huber_loss']:.6f}")
    
    # 输出神经网络模型架构
    print(f"\n=== 神经网络模型架构 ===")
    print(f"模型输入维度: {X_train.shape[1]}")
    print(f"模型参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 显示模型详细架构
    if nn.model is not None:
        print(f"\n模型详细架构:")
        nn.model.summary()
        
        # 计算模型参数数量
        total_params = nn.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in nn.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n模型参数统计:")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  不可训练参数: {non_trainable_params:,}")
    
    # 绘制改进的训练集测试集拟合图像
    print(f"\n=== 绘制训练集测试集拟合图像 ===")
    _plot_detailed_fitting_results(nn, X_train, y_train, X_test, y_test, results, item_name)
    
    return results

def save_training_summary(best_params, best_row, results, output_file):
    """
    保存训练总结到文件
    
    Args:
        best_params: 最佳参数
        best_row: 最佳试验行数据
        results: 训练结果
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("最佳参数训练总结\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("贝叶斯优化最佳结果:\n")
        f.write(f"  试验编号: {best_row['trial']}\n")
        f.write(f"  优化训练R²: {best_row['train_r2']:.6f}\n")
        f.write(f"  优化测试R²: {best_row['test_r2']:.6f}\n")
        f.write(f"  优化Huber损失: {best_row['huber_loss']:.6f}\n\n")
        
        f.write("使用的最佳参数:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("重新训练结果:\n")
        f.write(f"  训练集R²: {results['train_metrics']['r2']:.6f}\n")
        f.write(f"  测试集R²: {results['test_metrics']['r2']:.6f}\n")
        f.write(f"  测试集Huber损失: {results['test_metrics']['huber_loss']:.6f}\n")
        f.write(f"  测试集RMSE: {results['test_metrics']['rmse']:.6f}\n")
        f.write(f"  测试集MAE: {results['test_metrics']['mae']:.6f}\n")
        
        if 'model_path' in results:
            f.write(f"\n模型保存路径: {results['model_path']}\n")
    
    print(f"训练总结已保存到: {output_file}")

def _plot_detailed_fitting_results(nn, X_train, y_train, X_test, y_test, results, item_name):
    """
    绘制详细的训练集测试集拟合图像
    
    Args:
        nn: 神经网络实例
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        results: 训练结果
        item_name: 项目名称
    """
    try:
        # 获取预测结果
        train_pred = results.get('train_predictions', nn.model.predict(X_train).flatten())
        test_pred = results.get('test_predictions', nn.model.predict(X_test).flatten())
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{item_name} - 详细拟合结果分析', fontsize=16, fontweight='bold')
        
        # 1. 训练集预测散点图
        ax1 = axes[0, 0]
        ax1.scatter(y_train, train_pred, alpha=0.6, color='blue', s=30)
        min_val = min(y_train.min(), train_pred.min())
        max_val = max(y_train.max(), train_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想拟合线')
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title(f'训练集拟合 (R² = {results["train_metrics"]["r2"]:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 测试集预测散点图
        ax2 = axes[0, 1]
        ax2.scatter(y_test, test_pred, alpha=0.6, color='green', s=30)
        min_val = min(y_test.min(), test_pred.min())
        max_val = max(y_test.max(), test_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想拟合线')
        ax2.set_xlabel('真实值')
        ax2.set_ylabel('预测值')
        ax2.set_title(f'测试集拟合 (R² = {results["test_metrics"]["r2"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差分析 - 训练集
        ax3 = axes[1, 0]
        train_residuals = y_train - train_pred
        ax3.scatter(train_pred, train_residuals, alpha=0.6, color='blue', s=30)
        ax3.axhline(y=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('预测值')
        ax3.set_ylabel('残差 (真实值 - 预测值)')
        ax3.set_title(f'训练集残差分析 (RMSE = {results["train_metrics"]["rmse"]:.4f})')
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差分析 - 测试集
        ax4 = axes[1, 1]
        test_residuals = y_test - test_pred
        ax4.scatter(test_pred, test_residuals, alpha=0.6, color='green', s=30)
        ax4.axhline(y=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('预测值')
        ax4.set_ylabel('残差 (真实值 - 预测值)')
        ax4.set_title(f'测试集残差分析 (RMSE = {results["test_metrics"]["rmse"]:.4f})')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        plot_filename = os.path.join(plots_dir, f'{item_name}_详细拟合结果.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"详细拟合结果图已保存到: {plot_filename}")
        
        # 打印详细统计信息
        print(f"\n=== 详细统计信息 ===")
        print(f"训练集:")
        print(f"  样本数量: {len(y_train)}")
        print(f"  真实值范围: [{y_train.min():.4f}, {y_train.max():.4f}]")
        print(f"  预测值范围: [{train_pred.min():.4f}, {train_pred.max():.4f}]")
        print(f"  残差标准差: {np.std(train_residuals):.4f}")
        
        print(f"测试集:")
        print(f"  样本数量: {len(y_test)}")
        print(f"  真实值范围: [{y_test.min():.4f}, {y_test.max():.4f}]")
        print(f"  预测值范围: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
        print(f"  残差标准差: {np.std(test_residuals):.4f}")
        
    except Exception as e:
        print(f"绘制详细拟合结果时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='从贝叶斯优化结果中提取最佳参数并进行训练')
    parser.add_argument('--csv_file', type=str, default='bayesian_optimization_回归模型优化.csv',
                       help='贝叶斯优化结果CSV文件路径')
    parser.add_argument('--data_dir', type=str, default='data_row',
                       help='数据目录路径')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--item_name', type=str, default='最佳参数训练',
                       help='项目名称')
    parser.add_argument('--loss_function', type=str, default='huber_loss_tf',
                       choices=['huber_loss_tf', 'mean_squared_error', 'weighted_huber_loss_tf'],
                       help='损失函数类型')
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("从贝叶斯优化结果中提取最佳参数并进行训练")
        print("=" * 60)
        
        # 1. 加载数据
        X_train, y_train, X_test, y_test = load_data(args.data_dir)
        
        # 2. 加载最佳参数
        best_params, best_row = load_best_params(args.csv_file)
        
        # 3. 使用最佳参数进行训练
        results = train_with_best_params(
            X_train, y_train, X_test, y_test, best_params,
            epochs=args.epochs,
            batch_size=args.batch_size,
            item_name=args.item_name,
            loss_function=args.loss_function
        )
        
        # 4. 保存训练总结
        summary_file = f"best_params_training_summary_{int(time.time())}.txt"
        save_training_summary(best_params, best_row, results, summary_file)
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())