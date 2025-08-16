#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### 9. AbnormalValueDetector.py - 异常值检测器
**主要功能：异常值检测和剔除**

类功能说明：
- AbnormalValueDetector: 异常值检测类
  - 3σ原则异常值检测
  - IQR方法异常值检测
  - 异常值可视化
  - 数据清洗功能
"""

import numpy as np
import pandas as pd
from .LossManager import LossManager
from .EnhancedPlotManager import EnhancedPlotManager, SpecializedPlotManager

class AbnormalValueDetector:
    """
    异常值检测类
    
    功能：
    - 基于预测误差的异常值检测
    - 支持3σ原则和IQR方法
    - 异常值可视化和数据清洗
    """
    
    def __init__(self):
        self.loss_manager = LossManager()
    
    def detect_outliers_by_prediction_error(self, X_train, y_train, X_test, y_test, 
                                           y_train_pred, y_test_pred,
                                           threshold_method='iqr', threshold_factor=1.5, 
                                           plot_outliers=False, item="异丙醇"):
        """
        基于预测误差检测异常值
        
        参数:
        X_train: 训练集特征
        y_train: 训练集真实值
        X_test: 测试集特征
        y_test: 测试集真实值
        y_train_pred: 训练集预测值
        y_test_pred: 测试集预测值
        threshold_method: 阈值方法 ('3sigma' 或 'iqr')
        threshold_factor: 阈值因子
        plot_outliers: 是否绘制异常值图
        item: 物质名称
        
        返回:
        dict: 包含清洗后数据和异常值信息的字典
        """
        print(f"开始基于预测误差检测异常值 - 方法: {threshold_method}")
        
        # 计算残差
        train_residuals = np.abs(y_train.flatten() - y_train_pred.flatten())
        test_residuals = np.abs(y_test.flatten() - y_test_pred.flatten())
        
        # 检测异常值
        train_outliers = self._detect_outliers(train_residuals, threshold_method, threshold_factor)
        test_outliers = self._detect_outliers(test_residuals, threshold_method, threshold_factor)
        
        # 获取正常样本索引
        train_normal_idx = ~train_outliers
        test_normal_idx = ~test_outliers
        
        # 清洗数据
        X_train_clean = X_train[train_normal_idx]
        y_train_clean = y_train[train_normal_idx]
        X_test_clean = X_test[test_normal_idx]
        y_test_clean = y_test[test_normal_idx]
        
        # 统计信息
        train_outlier_count = np.sum(train_outliers)
        test_outlier_count = np.sum(test_outliers)
        
        print(f"训练集异常值数量: {train_outlier_count}/{len(y_train)} ({train_outlier_count/len(y_train)*100:.2f}%)")
        print(f"测试集异常值数量: {test_outlier_count}/{len(y_test)} ({test_outlier_count/len(y_test)*100:.2f}%)")
        
        # 绘制异常值图
        if plot_outliers:
            self._plot_outliers(train_residuals, test_residuals, train_outliers, test_outliers, item)
        
        return {
            'X_train_clean': X_train_clean,
            'y_train_clean': y_train_clean,
            'X_test_clean': X_test_clean,
            'y_test_clean': y_test_clean,
            'train_outliers': train_outliers,
            'test_outliers': test_outliers,
            'train_outlier_indices': np.where(train_outliers)[0],
            'test_outlier_indices': np.where(test_outliers)[0],
            'train_outlier_count': train_outlier_count,
            'test_outlier_count': test_outlier_count
        }
    
    def detect_outliers_3sigma(self, X_train, y_train, X_test, y_test, 
                              y_train_pred, y_test_pred, item="异丙醇"):
        """
        使用3σ原则检测异常值
        
        参数:
        X_train: 训练集特征
        y_train: 训练集真实值
        X_test: 测试集特征
        y_test: 测试集真实值
        y_train_pred: 训练集预测值
        y_test_pred: 测试集预测值
        item: 物质名称
        
        返回:
        dict: 包含清洗后数据和异常值信息的字典
        """
        return self.detect_outliers_by_prediction_error(
            X_train, y_train, X_test, y_test, y_train_pred, y_test_pred,
            threshold_method='3sigma', threshold_factor=3.0, 
            plot_outliers=True, item=item
        )
    
    def _detect_outliers(self, residuals, method, factor):
        """
        检测异常值的内部方法
        
        参数:
        residuals: 残差数组
        method: 检测方法 ('3sigma' 或 'iqr')
        factor: 阈值因子
        
        返回:
        outliers: 布尔数组，True表示异常值
        """
        if method == '3sigma':
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            threshold = mean_residual + factor * std_residual
            outliers = residuals > threshold
            
        elif method == 'iqr':
            Q1 = np.percentile(residuals, 25)
            Q3 = np.percentile(residuals, 75)
            IQR = Q3 - Q1
            threshold = Q3 + factor * IQR
            outliers = residuals > threshold
            
        else:
            raise ValueError(f"不支持的检测方法: {method}")
        
        return outliers
    
    def _plot_outliers(self, train_residuals, test_residuals, train_outliers, test_outliers, item="异丙醇"):
        """
        绘制异常值检测结果
        
        参数:
        train_residuals: 训练集残差
        test_residuals: 测试集残差
        train_outliers: 训练集异常值标记
        test_outliers: 测试集异常值标记
        item: 物质名称
        """
        try:
            specialized_plot_manager = SpecializedPlotManager()
            
            # 合并训练和测试残差
            all_residuals = np.concatenate([train_residuals, test_residuals])
            all_outliers = np.concatenate([train_outliers, test_outliers + len(train_residuals)])
            
            # 计算阈值（使用3σ原则）
            threshold = 3 * np.std(all_residuals)
            
            # 调用SpecializedPlotManager的异常值绘图方法
            specialized_plot_manager.plot_residual_distribution_with_outliers(
                all_residuals, threshold, all_outliers, 
                title=f"{item} - 残差分布与异常值检测"
            )
            
        except ImportError:
            print("警告：无法导入SpecializedPlotManager，跳过异常值可视化")
    
    def calculate_metrics_after_cleaning(self, y_true, y_pred, model_name="模型"):
        """
        计算清洗后数据的评估指标
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        
        返回:
        metrics: 评估指标字典
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # 确保数据形状一致
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # 计算各种指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        huber = self.loss_manager.huber_loss(y_true, y_pred)
        
        # 计算相对误差
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Huber_Loss': huber,
            'MAPE': mape,
            'Sample_Count': len(y_true)
        }
        
        print(f"\n{model_name} 清洗后评估指标:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Huber Loss: {huber:.6f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"样本数量: {len(y_true)}")
        
        return metrics
    
    def get_outlier_info(self, outlier_indices, X_data, y_data, y_pred, dataset_name="数据集"):
        """
        获取异常值的详细信息
        
        参数:
        outlier_indices: 异常值索引
        X_data: 特征数据
        y_data: 真实值
        y_pred: 预测值
        dataset_name: 数据集名称
        
        返回:
        outlier_info: 异常值信息DataFrame
        """
        if len(outlier_indices) == 0:
            print(f"{dataset_name}中没有检测到异常值")
            return pd.DataFrame()
        
        outlier_info = []
        for idx in outlier_indices:
            info = {
                'Index': idx,
                'True_Value': y_data[idx] if len(y_data.shape) == 1 else y_data[idx, 0],
                'Predicted_Value': y_pred[idx] if len(y_pred.shape) == 1 else y_pred[idx, 0],
                'Absolute_Error': abs(y_data[idx] - y_pred[idx]) if len(y_data.shape) == 1 else abs(y_data[idx, 0] - y_pred[idx, 0]),
                'Relative_Error_Percent': abs((y_data[idx] - y_pred[idx]) / y_data[idx] * 100) if len(y_data.shape) == 1 else abs((y_data[idx, 0] - y_pred[idx, 0]) / y_data[idx, 0] * 100)
            }
            outlier_info.append(info)
        
        outlier_df = pd.DataFrame(outlier_info)
        
        print(f"\n{dataset_name}异常值详细信息:")
        print(outlier_df.to_string(index=False))
        
        return outlier_df