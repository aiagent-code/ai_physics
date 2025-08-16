#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EnhancedPlotManager类介绍

一. 基本函数 (7个核心绘图功能)
1. plot_multi_line: 输入x,多个y,颜色掩码(可选),画多线折线图,用于光谱画图(基于样本点和数据点的都支持)
2. plot_single_line: 输入x,一个y,颜色掩码(可选),画单线折线图,用于画训练过程损失
3. plot_prediction_scatter: 输入向量y1和y2,还有颜色编码y3(可选),以及脚注等,画真实-预测散点图
4. plot_prediction_with_error_bars: 有误差棒的预测图像,按样本聚合显示(一个样本一个点)
5. plot_sample_metric: 输入一个x向量,画样本-指标图,用于偏差和方差图,浓度图等
6. plot_table: 画表格函数,支持DataFrame、numpy数组和列表
7. plot_activations: 激活层显示函数,用于神经网络可视化

二. 简单专用绘图方法(基于基本函数组合)
- plot_pls_prediction_results: PLS预测结果绘图(调用plot_prediction_scatter)

三. 兼容性方法
- plot_spectra_by_sample: 按样本绘制光谱图(调用plot_multi_line)
- plot_prediction_comparison: 预测对比图(调用相应核心方法)

注意：复杂的多子图专用绘图方法已移至SpecializedPlotManager类
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


class EnhancedPlotManager:
    """增强的绘图管理器：7个核心绘图功能"""
    
    def __init__(self, color_palette: str = 'tab10', random_seed: int = 42):
        """
        初始化增强绘图管理器
        
        Args:
            color_palette: 颜色调色板名称
            random_seed: 随机种子，确保颜色一致性
        """
        self.color_palette = color_palette
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建样本颜色映射
        self.sample_colors = {}
        self.color_counter = 0
        
    def _get_sample_color(self, sample_id: int) -> str:
        """
        为样本ID分配颜色，确保同一样本始终使用相同颜色
        
        Args:
            sample_id: 样本ID
            
        Returns:
            颜色字符串
        """
        if sample_id not in self.sample_colors:
            # 使用预定义的颜色调色板
            colors = plt.cm.get_cmap(self.color_palette).colors
            self.sample_colors[sample_id] = colors[self.color_counter % len(colors)]
            self.color_counter += 1
        
        return self.sample_colors[sample_id]
    
    def plot_multi_line(self, x: np.ndarray, Y: np.ndarray, color_mask: Optional[np.ndarray] = None,
                       title: str = "多线图", xlabel: str = "X轴", ylabel: str = "Y轴",
                       figsize: Tuple[int, int] = (12, 8), **kwargs) -> None:
        """
        1. 输入x,多个y,颜色掩码(可选,没有则颜色均不同),在一个窗口中画折线图,用于光谱画图
        
        Args:
            x: x轴数据
            Y: y轴数据，形状为(n_lines, n_points)或(n_points, n_lines)
            color_mask: 颜色掩码，可选
            title: 图标题
            xlabel: x轴标签
            ylabel: y轴标签
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        
        # 确保Y是二维数组
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        elif Y.shape[0] == len(x) and Y.shape[1] != len(x):
            Y = Y.T  # 转置
            
        n_lines = Y.shape[0]
        
        # 生成颜色
        if color_mask is not None:
            colors = [plt.cm.get_cmap(self.color_palette)(color_mask[i]) for i in range(n_lines)]
        else:
            colors = [plt.cm.get_cmap(self.color_palette)(i/max(1, n_lines-1)) for i in range(n_lines)]
        
        # 绘制每条线
        for i in range(n_lines):
            plt.plot(x, Y[i], color=colors[i], alpha=0.7, **kwargs)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_single_line(self, x: np.ndarray, y: np.ndarray, color_mask: Optional[np.ndarray] = None,
                        title: str = "单线图", xlabel: str = "X轴", ylabel: str = "Y轴",
                        figsize: Tuple[int, int] = (10, 6), **kwargs) -> None:
        """
        2. 输入x,一个y,颜色掩码(可选,没有则颜色均不同),在一个窗口中画折线图,用于画训练过程损失
        
        Args:
            x: x轴数据
            y: y轴数据
            color_mask: 颜色掩码，可选
            title: 图标题
            xlabel: x轴标签
            ylabel: y轴标签
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        
        # 确定颜色
        if color_mask is not None:
            color = plt.cm.get_cmap(self.color_palette)(color_mask)
        else:
            color = plt.cm.get_cmap(self.color_palette)(0.5)
            
        plt.plot(x, y, color=color, **kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               color_encoding: Optional[np.ndarray] = None,
                               title: str = "真实值vs预测值", footnote: str = "",
                               figsize: Tuple[int, int] = (8, 8), show_metrics: bool = True, **kwargs) -> None:
        """
        3. 输入向量y1和y2,还有颜色编码y3(可选),以及脚注等,画真实-预测图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            color_encoding: 颜色编码，可选
            title: 图标题
            footnote: 脚注
            figsize: 图像大小
            show_metrics: 是否显示拟合指标(R²、RMSE、Huber Loss)
        """
        from sklearn.metrics import r2_score, mean_squared_error
        
        plt.figure(figsize=figsize)
        
        # 确定颜色
        if color_encoding is not None:
            scatter = plt.scatter(y_true, y_pred, c=color_encoding, 
                                cmap=self.color_palette, alpha=0.6, **kwargs)
            plt.colorbar(scatter)
        else:
            plt.scatter(y_true, y_pred, alpha=0.6, **kwargs)
            
        # 绘制对角线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.title(title)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.grid(True, alpha=0.3)
        
        # 计算并显示拟合指标
        if show_metrics:
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # 计算Huber Loss (delta=3.0)
            residuals = np.abs(y_true - y_pred)
            delta = 3.0
            huber_loss = np.where(residuals <= delta,
                                0.5 * residuals**2,
                                delta * (residuals - 0.5 * delta)).mean()
            
            # 在图上显示指标
            metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nHuber Loss = {huber_loss:.4f}'
            plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 添加脚注
        if footnote:
            plt.figtext(0.5, 0.02, footnote, ha='center', fontsize=10)
            
        plt.tight_layout()
        plt.show()
        
    def plot_prediction_with_error_bars(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       sample_ids: np.ndarray, title: str = "预测结果(含误差棒)",
                                       figsize: Tuple[int, int] = (12, 8), dataset_labels=None, 
                                       save_path: str = None, **kwargs) -> None:
        """
        4. 有误差棒的预测图像,支持多数据集并为同一样本分配一致颜色
        
        Args:
            y_true: 真实值（可以是单个数组或多个数组的列表）
            y_pred: 预测值（可以是单个数组或多个数组的列表）
            sample_ids: 样本ID（可以是单个数组或多个数组的列表）
            title: 图标题
            figsize: 图像大小
            dataset_labels: 数据集标签列表，如['训练集', '测试集', '验证集']
            save_path: 保存路径，如果提供则保存图像到指定路径
        """
        plt.figure(figsize=figsize)
        
        # 处理单个数据集的情况
        if not isinstance(y_true, list):
            y_true = [y_true]
            y_pred = [y_pred]
            sample_ids = [sample_ids]
            dataset_labels = dataset_labels or ['数据集']
        
        # 获取所有唯一样本ID并分配颜色
        all_sample_ids = np.concatenate(sample_ids)
        unique_samples = np.unique(all_sample_ids)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_samples)))
        sample_color_map = {sample_id: colors[i] for i, sample_id in enumerate(unique_samples)}
        
        # 为每个数据集绘制误差棒
        markers = ['o', 's', '^', 'D', 'v']  # 不同数据集使用不同标记
        
        for dataset_idx, (y_t, y_p, s_ids) in enumerate(zip(y_true, y_pred, sample_ids)):
            dataset_label = dataset_labels[dataset_idx] if dataset_labels else f'数据集{dataset_idx+1}'
            marker = markers[dataset_idx % len(markers)]
            
            # 按样本分组计算统计量
            unique_dataset_samples = np.unique(s_ids)
            sample_true_means = []
            sample_pred_means = []
            sample_pred_stds = []
            sample_colors = []
            
            for sample_id in unique_dataset_samples:
                mask = s_ids == sample_id
                sample_true_means.append(y_t[mask].mean())
                sample_pred_means.append(y_p[mask].mean())
                sample_pred_stds.append(y_p[mask].std())
                sample_colors.append(sample_color_map[sample_id])
                
            sample_true_means = np.array(sample_true_means)
            sample_pred_means = np.array(sample_pred_means)
            sample_pred_stds = np.array(sample_pred_stds)
            
            # 为每个样本绘制误差棒
            for i, (true_mean, pred_mean, pred_std, color) in enumerate(zip(
                sample_true_means, sample_pred_means, sample_pred_stds, sample_colors)):
                
                if i == 0:  # 只在第一个点添加图例标签
                    plt.errorbar(true_mean, pred_mean, yerr=pred_std,
                               fmt=marker, color=color, alpha=0.7, capsize=5,
                               label=dataset_label, markersize=8)
                else:
                    plt.errorbar(true_mean, pred_mean, yerr=pred_std,
                               fmt=marker, color=color, alpha=0.7, capsize=5,
                               markersize=8)
        
        # 绘制对角线
        all_true = np.concatenate(y_true)
        all_pred = np.concatenate(y_pred)
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想预测线')
        
        plt.title(title)
        plt.xlabel('真实值(样本均值)')
        plt.ylabel('预测值(样本均值±标准差)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图像（如果提供了保存路径）
        if save_path:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
        
    def plot_sample_metric(self, x: np.ndarray, title: str = "样本-指标图",
                          xlabel: str = "样本", ylabel: str = "指标值",
                          figsize: Tuple[int, int] = (10, 6), plot_type: str = "bar", 
                          save_path: str = None, **kwargs) -> None:
        """
        5. 输入一个x向量,画样本-指标图,用于偏差和方差图,浓度图等
        
        Args:
            x: 指标值向量
            title: 图标题
            xlabel: x轴标签
            ylabel: y轴标签
            figsize: 图像大小
            plot_type: 图类型 ('bar', 'line', 'scatter')
            save_path: 保存路径，如果提供则保存图像到指定路径
        """
        plt.figure(figsize=figsize)
        
        sample_indices = np.arange(len(x))
        
        if plot_type == "bar":
            plt.bar(sample_indices, x, alpha=0.7, **kwargs)
        elif plot_type == "line":
            plt.plot(sample_indices, x, marker='o', **kwargs)
        elif plot_type == "scatter":
            plt.scatter(sample_indices, x, **kwargs)
        else:
            plt.plot(sample_indices, x, **kwargs)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图像（如果提供了保存路径）
        if save_path:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
        
    def plot_table(self, data: Union[pd.DataFrame, np.ndarray, List[List]], 
                  title: str = "数据表格", figsize: Tuple[int, int] = (10, 6),
                  column_names: Optional[List[str]] = None, **kwargs) -> None:
        """
        6. 画表格函数
        
        Args:
            data: 表格数据
            title: 表格标题
            figsize: 图像大小
            column_names: 列名
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        # 转换数据格式
        if isinstance(data, pd.DataFrame):
            table_data = data.values
            if column_names is None:
                column_names = data.columns.tolist()
        elif isinstance(data, np.ndarray):
            table_data = data
        else:
            table_data = data
            
        # 创建表格
        table = ax.table(cellText=table_data, colLabels=column_names,
                        cellLoc='center', loc='center', **kwargs)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title(title, pad=20)
        plt.tight_layout()
        plt.show()
        
    def plot_activations(self, activations: List[np.ndarray], layer_names: List[str],
                        input_sample: np.ndarray, title: str = "激活层显示",
                        figsize: Tuple[int, int] = (12, 8), **kwargs) -> None:
        """
        7. 激活层显示函数(来自于神经网络类)
        
        Args:
            activations: 各层激活值列表
            layer_names: 层名称列表
            input_sample: 输入样本
            title: 图标题
            figsize: 图像大小
        """
        n_layers = len(activations)
        fig, axes = plt.subplots(n_layers + 1, 1, figsize=(figsize[0], figsize[1] * (n_layers + 1) / 2))
        
        if n_layers == 0:
            axes = [axes]
        
        # 绘制原始输入
        axes[0].plot(input_sample, **kwargs)
        axes[0].set_title(f'{title} - 原始输入')
        axes[0].set_ylabel('强度')
        axes[0].grid(True, alpha=0.3)
        
        # 绘制各层激活
        for i, (activation, layer_name) in enumerate(zip(activations, layer_names)):
            ax = axes[i + 1]
            
            # 处理不同维度的激活值
            if activation.ndim == 3:  # (batch, length, channels)
                # 显示前几个通道
                n_channels = min(6, activation.shape[-1])
                for j in range(n_channels):
                    ax.plot(activation[0, :, j], alpha=0.7, label=f'Channel {j+1}')
                ax.legend()
            elif activation.ndim == 2:  # (batch, features)
                ax.bar(range(len(activation[0])), activation[0], alpha=0.7)
            else:
                ax.plot(activation.flatten(), **kwargs)
                
            ax.set_title(f'{title} - {layer_name}')
            ax.set_ylabel('激活值')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    # 保留一些兼容性方法
    def plot_spectra_by_sample(self, x: np.ndarray, Y: np.ndarray, sample_ids: np.ndarray,
                              config: Dict, max_samples_to_show: int = 10) -> None:
        """兼容性方法：调用plot_multi_line"""
        title = config.get('title', '光谱图')
        xlabel = config.get('xlabel', '波数')
        ylabel = config.get('ylabel', '吸光度')
        figsize = config.get('figsize', (12, 8))
        
        # 按样本分组
        unique_sample_ids = np.unique(sample_ids)
        if len(unique_sample_ids) > max_samples_to_show:
            selected_samples = np.random.choice(unique_sample_ids, max_samples_to_show, replace=False)
        else:
            selected_samples = unique_sample_ids
            
        selected_Y = []
        for sample_id in selected_samples:
            mask = sample_ids == sample_id
            selected_Y.append(Y[mask].mean(axis=0))  # 取样本均值
            
        self.plot_multi_line(x, np.array(selected_Y), title=title, xlabel=xlabel, ylabel=ylabel, figsize=figsize)
        
    def plot_prediction_comparison(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 sample_ids: np.ndarray, config: Dict,
                                 title: Optional[str] = None, show_error_bars: bool = True) -> None:
        """兼容性方法：调用相应的核心方法"""
        if title is None:
            title = config.get('title', '预测对比图')
        figsize = config.get('figsize', (8, 8))
        
        if show_error_bars:
            self.plot_prediction_with_error_bars(y_true, y_pred, sample_ids, title=title, figsize=figsize)
        else:
            self.plot_prediction_scatter(y_true, y_pred, title=title, figsize=figsize)
    
    # 简单专用绘图方法(基于基本函数组合)
    def plot_pls_prediction_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   r2_score: float, rmse_score: float,
                                   title: str = "PLS预测结果", item: str = "目标物质",
                                   figsize: Tuple[int, int] = (8, 8)) -> None:
        """PLS预测结果绘图(调用基本函数plot_prediction_scatter)"""
        footnote = f'R² = {r2_score:.3f}, RMSE = {rmse_score:.3f}'
        plot_title = f'{title} - {item}'
        self.plot_prediction_scatter(y_true, y_pred, title=plot_title, footnote=footnote, figsize=figsize)
    
    def plot_train_test_comparison(self, y_train_true: np.ndarray, y_train_pred: np.ndarray,
                                 y_test_true: np.ndarray, y_test_pred: np.ndarray,
                                 train_r2: float, train_rmse: float,
                                 test_r2: float, test_rmse: float,
                                 title: str = "训练测试结果对比", item: str = "目标物质",
                                 figsize: Tuple[int, int] = (15, 6)) -> None:
        """训练和测试结果对比绘图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 训练集结果
        ax1.scatter(y_train_true, y_train_pred, alpha=0.6, color='blue')
        min_val = min(y_train_true.min(), y_train_pred.min())
        max_val = max(y_train_true.max(), y_train_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax1.set_title(f'{item}训练集结果\nR² = {train_r2:.3f}, RMSE = {train_rmse:.3f}')
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.grid(True, alpha=0.3)
        
        # 测试集结果
        ax2.scatter(y_test_true, y_test_pred, alpha=0.6, color='green')
        min_val = min(y_test_true.min(), y_test_pred.min())
        max_val = max(y_test_true.max(), y_test_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax2.set_title(f'{item}测试集结果\nR² = {test_r2:.3f}, RMSE = {test_rmse:.3f}')
        ax2.set_xlabel('真实值')
        ax2.set_ylabel('预测值')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_results(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "预测结果",
                              figsize: Tuple[int, int] = (8, 8)) -> None:
        """基本预测结果绘图"""
        from sklearn.metrics import r2_score, mean_squared_error
        import numpy as np
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        footnote = f'R² = {r2:.3f}, RMSE = {rmse:.3f}'
        
        self.plot_prediction_scatter(y_true, y_pred, title=title, footnote=footnote, figsize=figsize)
    
    def plot_prediction_distribution(self, y_pred: np.ndarray, title: str = "预测结果分布",
                                   figsize: Tuple[int, int] = (10, 6)) -> None:
        """预测结果分布图"""
        plt.figure(figsize=figsize)
        plt.hist(y_pred, bins=30, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel('预测值')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(y_pred)
        std_val = np.std(y_pred)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.3f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_val + std_val:.3f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_val - std_val:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_pls_prediction_with_sample_ids(self, y_train, y_train_pred, train_sample_ids,
                                           y_val, y_val_pred, val_sample_ids,
                                           y_test, y_test_pred, test_sample_ids, item):
        """绘制PLS预测结果（三个数据集，同一样本同色显示）"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 获取所有唯一样本ID并分配颜色
        all_sample_ids = np.concatenate([train_sample_ids, val_sample_ids, test_sample_ids])
        unique_samples = np.unique(all_sample_ids)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_samples)))
        sample_color_map = {sample_id: colors[i] for i, sample_id in enumerate(unique_samples)}
        
        datasets = [
            (y_train, y_train_pred, train_sample_ids, '训练集', axes[0]),
            (y_val, y_val_pred, val_sample_ids, '验证集', axes[1]),
            (y_test, y_test_pred, test_sample_ids, '测试集', axes[2])
        ]
        
        for y_true, y_pred, sample_ids, dataset_name, ax in datasets:
            # 为每个样本分配颜色
            point_colors = [sample_color_map[sid] for sid in sample_ids]
            
            # 绘制散点图
            ax.scatter(y_true, y_pred, c=point_colors, alpha=0.6)
            
            # 绘制对角线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # 计算指标
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            ax.set_title(f'{item} {dataset_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{item} - PLS预测结果对比（同一样本同色显示）', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_sample_variance_analysis(self, Y_all, Y_all_pred, all_sample_ids, item):
        """绘制样本方差分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按样本分组计算方差
        unique_samples = np.unique(all_sample_ids)
        sample_variances_true = []
        sample_variances_pred = []
        sample_means_true = []
        sample_means_pred = []
        
        for sample_id in unique_samples:
            mask = all_sample_ids == sample_id
            sample_variances_true.append(Y_all[mask].var())
            sample_variances_pred.append(Y_all_pred[mask].var())
            sample_means_true.append(Y_all[mask].mean())
            sample_means_pred.append(Y_all_pred[mask].mean())
        
        # 左图：样本方差对比
        ax1.scatter(sample_variances_true, sample_variances_pred, alpha=0.6)
        min_var = min(min(sample_variances_true), min(sample_variances_pred))
        max_var = max(max(sample_variances_true), max(sample_variances_pred))
        ax1.plot([min_var, max_var], [min_var, max_var], 'r--', alpha=0.8)
        ax1.set_title(f'{item} - 样本方差对比')
        ax1.set_xlabel('真实值方差')
        ax1.set_ylabel('预测值方差')
        ax1.grid(True, alpha=0.3)
        
        # 右图：样本均值对比
        ax2.scatter(sample_means_true, sample_means_pred, alpha=0.6)
        min_mean = min(min(sample_means_true), min(sample_means_pred))
        max_mean = max(max(sample_means_true), max(sample_means_pred))
        ax2.plot([min_mean, max_mean], [min_mean, max_mean], 'r--', alpha=0.8)
        ax2.set_title(f'{item} - 样本均值对比')
        ax2.set_xlabel('真实值均值')
        ax2.set_ylabel('预测值均值')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_component_analysis(self, components_range: List[int], loss_values: List[float], 
                              r2_values: List[float], best_components: int,
                              title: str = "主成分分析结果", figsize: Tuple[int, int] = (12, 8)) -> None:
        """主成分分析结果绘图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 损失值图
        ax1.plot(components_range, loss_values, 'bo-', label='损失值')
        ax1.axvline(best_components, color='red', linestyle='--', alpha=0.7, label=f'最优组件数: {best_components}')
        ax1.set_title('损失值 vs 组件数')
        ax1.set_xlabel('组件数')
        ax1.set_ylabel('损失值')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # R²值图
        ax2.plot(components_range, r2_values, 'go-', label='R²值')
        ax2.axvline(best_components, color='red', linestyle='--', alpha=0.7, label=f'最优组件数: {best_components}')
        ax2.set_title('R² vs 组件数')
        ax2.set_xlabel('组件数')
        ax2.set_ylabel('R²值')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpecializedPlotManager类介绍

专用绘图管理器，处理复杂的多子图布局和特殊绘图需求
包含以下专用绘图方法：
1. plot_pls_prediction_results_with_train_test: PLS训练测试结果对比（双子图）
2. plot_pls_components_analysis: PLS成分分析绘图（多子图）
3. plot_residual_distribution_with_outliers: 残差分布与异常值检测绘图（双子图）

这些方法由于需要复杂的子图布局和特殊的绘图逻辑，无法简单地用基本函数组合实现。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class SpecializedPlotManager:
    """专用绘图管理器：处理复杂的多子图布局和特殊绘图需求"""
    
    def __init__(self):
        """初始化专用绘图管理器"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_pls_training_loss(self, components_range, train_scores, item):
        """绘制PLS训练损失图"""
        plt.figure(figsize=(10, 6))
        plt.plot(components_range, train_scores, 'b-o', linewidth=2, markersize=6, label='训练损失')
        plt.xlabel('主成分数量')
        plt.ylabel('损失值')
        plt.title(f'{item} - PLS训练损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 标记最优点
        min_idx = np.argmin(train_scores)
        optimal_components = components_range[min_idx]
        optimal_loss = train_scores[min_idx]
        plt.scatter(optimal_components, optimal_loss, color='red', s=100, zorder=5, label=f'最优点 (n={optimal_components})')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_pls_prediction_results_with_train_test(self, y_train_true: np.ndarray, y_train_pred: np.ndarray,
                                                   y_test_true: np.ndarray, y_test_pred: np.ndarray,
                                                   train_r2: float, train_rmse: float,
                                                   test_r2: float, test_rmse: float,
                                                   title: str = "PLS训练测试结果对比", item: str = "目标物质",
                                                   figsize: Tuple[int, int] = (15, 6)) -> None:
        """PLS训练和测试结果对比绘图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 训练集结果
        ax1.scatter(y_train_true, y_train_pred, alpha=0.6, color='blue')
        min_val = min(y_train_true.min(), y_train_pred.min())
        max_val = max(y_train_true.max(), y_train_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax1.set_title(f'{item}训练集结果\nR² = {train_r2:.3f}, RMSE = {train_rmse:.3f}')
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.grid(True, alpha=0.3)
        
        # 测试集结果
        ax2.scatter(y_test_true, y_test_pred, alpha=0.6, color='green')
        min_val = min(y_test_true.min(), y_test_pred.min())
        max_val = max(y_test_true.max(), y_test_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax2.set_title(f'{item}测试集结果\nR² = {test_r2:.3f}, RMSE = {test_rmse:.3f}')
        ax2.set_xlabel('真实值')
        ax2.set_ylabel('预测值')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_pls_components_analysis(self, components: np.ndarray, wavelengths: np.ndarray,
                                   n_components: int, title: str = "PLS成分分析",
                                   figsize: Tuple[int, int] = (12, 8)) -> None:
        """PLS成分分析绘图"""
        n_show = min(n_components, 4)  # 最多显示4个成分
        fig, axes = plt.subplots(n_show, 1, figsize=(figsize[0], figsize[1] * n_show / 2))
        
        if n_show == 1:
            axes = [axes]
        
        for i in range(n_show):
            axes[i].plot(wavelengths, components[:, i])
            axes[i].set_title(f'PLS成分 {i+1}')
            axes[i].set_ylabel('权重')
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('波数/cm⁻¹')
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_residual_distribution_with_outliers(self, residuals: np.ndarray, threshold: float,
                                               outlier_indices: np.ndarray, title: str = "残差分布与异常值检测",
                                               figsize: Tuple[int, int] = (12, 8)) -> None:
        """残差分布与异常值检测绘图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 残差直方图
        ax1.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=threshold, color='red', linestyle='--', label=f'阈值={threshold:.3f}')
        ax1.axvline(x=-threshold, color='red', linestyle='--')
        ax1.set_xlabel('残差')
        ax1.set_ylabel('频数')
        ax1.set_title('残差分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 残差散点图
        normal_mask = np.ones(len(residuals), dtype=bool)
        normal_mask[outlier_indices] = False
        
        ax2.scatter(np.arange(len(residuals))[normal_mask], residuals[normal_mask], 
                   alpha=0.6, color='blue', label='正常点')
        ax2.scatter(outlier_indices, residuals[outlier_indices], 
                   alpha=0.8, color='red', label='异常值')
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=-threshold, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('样本索引')
        ax2.set_ylabel('残差')
        ax2.set_title('残差分布图')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()