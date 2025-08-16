#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ErrorDeviationAnalyzer类介绍

一.方法
1. 通过评估结果画出预测图像(包括误差棒,不同颜色,画图调用EnhancedPlotManager,样本调用SampleManager)
2. '偏差图+方差图'函数(同上)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from .EnhancedPlotManager import EnhancedPlotManager
from .SampleManager import SampleManager


class ErrorDeviationAnalyzer:
    """误差偏差分析器：用于分析和可视化模型预测的误差和偏差"""
    
    def __init__(self, plot_manager: Optional[EnhancedPlotManager] = None, 
                 sample_manager: Optional[SampleManager] = None):
        """
        初始化误差偏差分析器
        
        Args:
            plot_manager: 绘图管理器实例
            sample_manager: 样本管理器实例
        """
        self.plot_manager = plot_manager or EnhancedPlotManager()
        self.sample_manager = sample_manager or SampleManager()
        
    def plot_prediction_with_error_bars(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       sample_ids: np.ndarray, config: Dict,
                                       model_name: str = "Model") -> None:
        """
        通过评估结果画出预测图像(包括误差棒,不同颜色)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            sample_ids: 样本ID数组
            config: 配置字典，包含物质名称等信息
            model_name: 模型名称
        """
        # 使用新的简化接口绘制预测结果（带误差棒）
        self.plot_manager.plot_prediction_with_error_bars(
            y_true=y_true,
            y_pred=y_pred,
            sample_ids=sample_ids,
            title=f'{model_name} - 预测结果对比图（带误差棒）',
            xlabel='样本',
            ylabel='浓度值'
        )
        
    def plot_bias_variance_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   sample_ids: np.ndarray, config: Dict,
                                   model_name: str = "Model") -> Dict:
        """
        偏差图+方差图函数
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            sample_ids: 样本ID数组
            config: 配置字典
            model_name: 模型名称
            
        Returns:
            包含偏差和方差统计信息的字典
        """
        # 计算样本级别的统计信息
        sample_stats = self._calculate_sample_bias_variance(y_true, y_pred, sample_ids)
        
        # 打印样本偏差和方差数据表格
        self._print_bias_variance_table(sample_stats, model_name)
        
        # 使用新的简化接口绘制偏差分析
        bias_save_path = f"./plots/{model_name}_样本偏差分析.png"
        self.plot_manager.plot_sample_metric(
            x=sample_stats['bias'],
            title=f'{model_name} - 样本偏差分析',
            xlabel='样本ID',
            ylabel='偏差值',
            save_path=bias_save_path
        )
        
        # 使用新的简化接口绘制方差分析
        variance_save_path = f"./plots/{model_name}_样本方差分析.png"
        self.plot_manager.plot_sample_metric(
            x=sample_stats['variance'],
            title=f'{model_name} - 样本方差分析',
            xlabel='样本ID',
            ylabel='方差值',
            save_path=variance_save_path
        )
        
        return sample_stats

    def _print_bias_variance_table(self, sample_stats: Dict, model_name: str) -> None:
        """
        打印样本偏差和方差数据表格
        
        Args:
            sample_stats: 样本统计信息字典
            model_name: 模型名称
        """
        print(f"\n=== {model_name} 样本偏差和方差分析表 ===")
        print("-" * 80)
        print(f"{'样本ID':<10} {'偏差':<12} {'方差':<12} {'MSE':<12} {'真实均值':<12} {'预测均值':<12}")
        print("-" * 80)
        
        for i in range(len(sample_stats['sample_ids'])):
            sample_id = sample_stats['sample_ids'][i]
            bias = sample_stats['bias'][i]
            variance = sample_stats['variance'][i]
            mse = sample_stats['mse'][i]
            true_mean = sample_stats['sample_means_true'][i]
            pred_mean = sample_stats['sample_means_pred'][i]
            
            print(f"{sample_id:<10} {bias:<12.4f} {variance:<12.4f} {mse:<12.4f} {true_mean:<12.4f} {pred_mean:<12.4f}")
        
        print("-" * 80)
        # 计算总体统计
        avg_bias = np.mean(sample_stats['bias'])
        avg_variance = np.mean(sample_stats['variance'])
        avg_mse = np.mean(sample_stats['mse'])
        
        print(f"{'平均值':<10} {avg_bias:<12.4f} {avg_variance:<12.4f} {avg_mse:<12.4f}")
        print(f"样本总数: {len(sample_stats['sample_ids'])}")
        print("-" * 80)

    def analyze_multi_dataset_errors(self, datasets: Dict[str, Dict], 
                                   title_prefix: str = "Model") -> Dict:
        """
        分析多数据集的样本误差和方差
        
        Args:
            datasets: 数据集字典，格式为 {'dataset_name': {'y_true': array, 'y_pred': array}}
            title_prefix: 图表标题前缀
            
        Returns:
            分析结果字典
        """
        print(f"\n=== {title_prefix} 样本误差偏差分析 ===")
        
        # 首先绘制样本级别的预测图（类似evaluate但以样本为单位）
        self._plot_sample_level_predictions(datasets, title_prefix)
        
        results = {}
        all_sample_errors = []
        all_sample_ids = []
        all_dataset_labels = []
        
        # 分析每个数据集
        for dataset_name, data in datasets.items():
            y_true = data['y_true'].flatten()
            y_pred = data['y_pred'].flatten()
            
            # 获取样本ID（如果没有提供则生成）
            sample_ids = data.get('sample_ids', np.arange(len(y_true)))
            
            # 计算样本误差
            errors = np.abs(y_true - y_pred)
            squared_errors = (y_true - y_pred) ** 2
            
            # 计算统计指标
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            rmse = np.sqrt(np.mean(squared_errors))
            
            dataset_result = {
                'mean_absolute_error': mean_error,
                'std_error': std_error,
                'rmse': rmse,
                'sample_errors': errors,
                'sample_ids': sample_ids,
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            results[dataset_name] = dataset_result
            
            # 收集所有样本数据用于综合分析
            all_sample_errors.extend(errors)
            all_sample_ids.extend([f"{dataset_name}_{i}" for i in sample_ids])
            all_dataset_labels.extend([dataset_name] * len(errors))
            
            print(f"{dataset_name} - 平均绝对误差: {mean_error:.4f}, 误差标准差: {std_error:.4f}, RMSE: {rmse:.4f}")
        
        # 绘制样本方差分析图
        self._plot_multi_dataset_variance_analysis(all_sample_errors, all_sample_ids, 
                                                  all_dataset_labels, title_prefix)
        
        return results
    
    def _plot_sample_level_predictions(self, datasets: Dict[str, Dict], title_prefix: str) -> None:
        """
        绘制样本级别的预测图（类似evaluate但以样本为单位）
        
        Args:
            datasets: 数据集字典，包含y_true, y_pred, sample_ids
            title_prefix: 图表标题前缀
        """
        for dataset_name, data in datasets.items():
            y_true = data['y_true'].flatten()
            y_pred = data['y_pred'].flatten()
            sample_ids = data.get('sample_ids', np.arange(len(y_true)))
            
            # 使用plot_manager绘制带误差棒的预测图
            self.plot_manager.plot_prediction_with_error_bars(
                y_true=y_true,
                y_pred=y_pred,
                sample_ids=sample_ids,
                title=f'{title_prefix} - {dataset_name}集样本级别预测结果'
            )
        
    def _plot_multi_dataset_sample_errors(self, y_true_list: List[np.ndarray], 
                                         y_pred_list: List[np.ndarray],
                                         sample_ids_list: List[np.ndarray],
                                         dataset_labels: List[str],
                                         item: str) -> None:
        """
        绘制多数据集样本误差图
        """
        # 计算每个样本的绝对误差
        sample_errors_list = []
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            errors = np.abs(y_true.flatten() - y_pred.flatten())
            sample_errors_list.append(errors)
        
        # 使用EnhancedPlotManager绘制样本-指标图
        save_path = f"./plots/{item}_多数据集样本误差分析图.png"
        self.plot_manager.plot_sample_variance_analysis(
            sample_ids_list=sample_ids_list,
            values_list=sample_errors_list,
            dataset_labels=dataset_labels,
            title=f"{item} - 多数据集样本绝对误差分析",
            ylabel="绝对误差",
            save_path=save_path
        )
        
        print(f"已生成{item}的多数据集样本误差分析图")
    
    def _plot_multi_dataset_variance_analysis(self, sample_errors: List[float],
                                             sample_ids: List[str],
                                             dataset_labels: List[str],
                                             item: str) -> None:
        """
        绘制多数据集样本方差分析图
        """
        # 按数据集分组计算方差
        unique_datasets = list(set(dataset_labels))
        variance_by_dataset = {}
        
        for dataset in unique_datasets:
            dataset_errors = [error for error, label in zip(sample_errors, dataset_labels) 
                            if label == dataset]
            variance_by_dataset[dataset] = np.var(dataset_errors)
        
        # 打印方差分析结果
        print("\n=== 方差分析结果 ===")
        for dataset, variance in variance_by_dataset.items():
            print(f"{dataset}: 误差方差 = {variance:.6f}")
        
        print(f"已完成{item}的多数据集方差分析")
        
    def _calculate_sample_bias_variance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       sample_ids: np.ndarray) -> Dict:
        """
        计算样本级别的偏差和方差
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            sample_ids: 样本ID数组
            
        Returns:
            包含每个样本统计信息的字典
        """
        unique_samples = np.unique(sample_ids)
        sample_stats = {
            'sample_ids': [],
            'bias': [],
            'variance': [],
            'mse': [],
            'sample_means_true': [],
            'sample_means_pred': []
        }
        
        for sample_id in unique_samples:
            mask = sample_ids == sample_id
            y_true_sample = y_true[mask]
            y_pred_sample = y_pred[mask]
            
            # 计算偏差（预测均值与真实均值的差异）
            bias = np.mean(y_pred_sample) - np.mean(y_true_sample)
            
            # 计算方差（预测值的方差）
            variance = np.var(y_pred_sample)
            
            # 计算MSE
            mse = np.mean((y_true_sample - y_pred_sample) ** 2)
            
            sample_stats['sample_ids'].append(sample_id)
            sample_stats['bias'].append(bias)
            sample_stats['variance'].append(variance)
            sample_stats['mse'].append(mse)
            sample_stats['sample_means_true'].append(np.mean(y_true_sample))
            sample_stats['sample_means_pred'].append(np.mean(y_pred_sample))
            
        return sample_stats
        
    def _plot_bias_analysis(self, sample_stats: Dict, ax, config: Dict) -> None:
        """
        绘制偏差分析图
        
        Args:
            sample_stats: 样本统计信息
            ax: matplotlib轴对象
            config: 配置字典
        """
        sample_ids = sample_stats['sample_ids']
        bias_values = sample_stats['bias']
        
        # 使用样本颜色
        colors = [self.plot_manager._get_sample_color(sid) for sid in sample_ids]
        
        bars = ax.bar(range(len(sample_ids)), bias_values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('样本ID')
        ax.set_ylabel(f'偏差 ({config.get("substance_name", "Unknown")})')
        ax.set_title('样本偏差分析')
        ax.set_xticks(range(len(sample_ids)))
        ax.set_xticklabels([f'S{sid}' for sid in sample_ids], rotation=45)
        
        # 添加数值标签
        for i, (bar, bias) in enumerate(zip(bars, bias_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + np.sign(height) * 0.01,
                   f'{bias:.3f}', ha='center', va='bottom' if height > 0 else 'top')
                   
    def _plot_variance_analysis(self, sample_stats: Dict, ax, config: Dict) -> None:
        """
        绘制方差分析图
        
        Args:
            sample_stats: 样本统计信息
            ax: matplotlib轴对象
            config: 配置字典
        """
        sample_ids = sample_stats['sample_ids']
        variance_values = sample_stats['variance']
        
        # 使用样本颜色
        colors = [self.plot_manager._get_sample_color(sid) for sid in sample_ids]
        
        bars = ax.bar(range(len(sample_ids)), variance_values, color=colors, alpha=0.7)
        
        ax.set_xlabel('样本ID')
        ax.set_ylabel(f'方差 ({config.get("substance_name", "Unknown")}²)')
        ax.set_title('样本方差分析')
        ax.set_xticks(range(len(sample_ids)))
        ax.set_xticklabels([f'S{sid}' for sid in sample_ids], rotation=45)
        
        # 添加数值标签
        for i, (bar, var) in enumerate(zip(bars, variance_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{var:.3f}', ha='center', va='bottom')
                   
    def analyze_prediction_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                                sample_ids: np.ndarray, config: Dict,
                                model_name: str = "Model") -> Dict:
        """
        综合分析预测误差，包括预测图像和偏差方差分析
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            sample_ids: 样本ID数组
            config: 配置字典
            model_name: 模型名称
            
        Returns:
            完整的误差分析结果
        """
        print(f"\n=== {model_name} 误差偏差分析 ===")
        
        # 1. 绘制预测图像（带误差棒）
        print("1. 绘制预测对比图...")
        self.plot_prediction_with_error_bars(y_true, y_pred, sample_ids, config, model_name)
        
        # 2. 偏差和方差分析
        print("2. 进行偏差方差分析...")
        sample_stats = self.plot_bias_variance_analysis(y_true, y_pred, sample_ids, config, model_name)
        
        # 3. 计算总体统计信息
        overall_stats = {
            'total_bias': np.mean(sample_stats['bias']),
            'total_variance': np.mean(sample_stats['variance']),
            'total_mse': np.mean(sample_stats['mse']),
            'rmse': np.sqrt(np.mean(sample_stats['mse'])),
            'sample_count': len(sample_stats['sample_ids'])
        }
        
        print(f"\n总体统计：")
        print(f"  平均偏差: {overall_stats['total_bias']:.4f}")
        print(f"  平均方差: {overall_stats['total_variance']:.4f}")
        print(f"  平均MSE: {overall_stats['total_mse']:.4f}")
        print(f"  RMSE: {overall_stats['rmse']:.4f}")
        print(f"  样本数量: {overall_stats['sample_count']}")
        
        return {
            'sample_stats': sample_stats,
            'overall_stats': overall_stats
        }