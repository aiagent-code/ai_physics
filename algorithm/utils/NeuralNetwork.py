#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuralNetwork类介绍

一.函数
1. 训练函数,用EnhancedPlotManager画出来
2. 评估函数(需要返回评估结果),用EnhancedPlotManager画出来
3. 贝叶斯优化,调用训练函数进行,现在是分开的,容易造成标准混乱,用EnhancedPlotManager画出来
4. 1DCNN模型
5. 激活层显示函数
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np
import pandas as pd
import os
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from utils.LossManager import LossManager
from utils.EnhancedPlotManager import EnhancedPlotManager
from typing import Dict, List, Optional, Tuple, Union

# 从utils_tf.py合并的工具函数
def huber_loss_tf(y_true, y_pred, delta=1.0):
    """
    TensorFlow版本的Huber Loss，用于模型训练
    参数:
    y_true: 真实值
    y_pred: 预测值
    delta: Huber Loss的阈值参数，默认为3.0
    """
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)

def weighted_huber_loss_tf(y_true, y_pred, delta=1.0):
    """
    加权版本的Huber Loss，真实值为0的数据点权重变为1/2
    参数:
    y_true: 真实值
    y_pred: 预测值
    delta: Huber Loss的阈值参数，默认为1.0
    """
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    
    # 计算基础损失
    loss = 0.5 * tf.square(quadratic) + delta * linear
    
    # 创建权重：真实值为0时权重为0.5，否则为1.0
    weights = tf.where(tf.equal(y_true, 0.0), 0.5, 1.0)
    
    # 应用权重
    weighted_loss = loss * weights
    
    return tf.reduce_mean(weighted_loss)

def fit_line(x, y):
    """拟合直线，返回斜率和截距"""
    k = np.cov(x, y)[0,1] / np.var(x)
    b = np.mean(y) - k * np.mean(x)
    return k, b


class NeuralNetwork:
    """神经网络类：统一的CNN神经网络实现"""
    
    def __init__(self, plot_manager: Optional[EnhancedPlotManager] = None):
        """
        初始化神经网络类
        
        Args:
            plot_manager: 绘图管理器实例
        """
        from sklearn.preprocessing import StandardScaler
        self.loss_manager = LossManager()
        self.plot_manager = plot_manager or EnhancedPlotManager()
        self.best_params = None
        self.model = None
        self.training_history = None
        self.y_scaler = StandardScaler()  # 目标变量标准化器
        self.y_scaled = False  # 标记是否已对y进行标准化
        
    def create_1dcnn_model(self, DenseN: int, DropoutR: float, C1_K: int, C1_S: int, 
                          C2_K: int, C2_S: int, input_dim: int, learning_rate: float = 0.001, 
                          loss_function: str = 'mean_squared_error') -> tf.keras.Model:
        """
        1DCNN模型构建
        
        Args:
            DenseN: 全连接层神经元数
            DropoutR: Dropout比率
            C1_K: 第一卷积层核大小
            C1_S: 第一卷积层步长
            C2_K: 第二卷积层核大小
            C2_S: 第二卷积层步长
            input_dim: 输入维度
            learning_rate: 学习率
            loss_function: 损失函数类型
            
        Returns:
            构建的1DCNN模型
        """
        activation = 'relu'
        model = keras.Sequential([
            keras.layers.GaussianNoise(0.01, input_shape=(input_dim,)),
            keras.layers.Reshape((input_dim, 1)),
            keras.layers.Conv1D(C1_K, C1_S, padding='same', activation=activation),
            keras.layers.Conv1D(C2_K, C2_S, padding='same', activation=activation),
            keras.layers.Flatten(),
            keras.layers.Dropout(DropoutR),
            keras.layers.Dense(DenseN, activation=activation),
            keras.layers.Dense(1, activation='linear')
        ])
        
        # 损失函数映射
        loss_mapping = {
            'mean_squared_error': 'mean_squared_error',
            'huber_loss_tf': huber_loss_tf,
            'weighted_huber_loss_tf': weighted_huber_loss_tf
        }
        
        # 获取损失函数
        loss_func = loss_mapping.get(loss_function, 'mean_squared_error')
        
        model.compile(
            loss=loss_func,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['mean_absolute_error']
        )
        
        return model
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray,
                   model_params: Dict, epochs: int = 200, batch_size: int = 32,
                   item: str = "异丙醇", save_path: str = None, 
                   loss_function: str = 'mean_squared_error',is_callback:bool=True) -> Dict:

        """
        训练函数,用EnhancedPlotManager画出来
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            model_params: 模型参数字典
            epochs: 训练轮数
            batch_size: 批次大小
            item: 物质名称
            save_path: 模型保存路径
            
        Returns:
            训练结果字典
        """
        print(f"\n=== 开始训练 {item} 神经网络模型 ===")
        
        # 不使用目标变量标准化，直接使用原始值
        self.y_scaled = False
        print(f"使用原始目标变量，范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        # 创建模型
        self.model = self.create_1dcnn_model(
            DenseN=model_params['DenseN'],
            DropoutR=model_params['DropoutR'],
            C1_K=model_params['C1_K'],
            C1_S=model_params['C1_S'],
            C2_K=model_params['C2_K'],
            C2_S=model_params['C2_S'],
            input_dim=X_train.shape[1],
            learning_rate=model_params.get('learning_rate', 0.001),
            loss_function=loss_function
        )
        if is_callback==True:
            # 设置回调函数
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, min_lr=1e-7)
            ]
            
            # 训练模型（使用原始y值）
            # 在贝叶斯优化期间设置适当的输出级别
            is_bayesian_trial = item.endswith('_trial_') or '_trial_' in item
            verbose_level = 2 if is_bayesian_trial else 1  # 2表示每个epoch显示一行进度
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose_level
            )
        else:          
            # 训练模型（使用原始y值）
            # 在贝叶斯优化期间设置适当的输出级别
            is_bayesian_trial = item.endswith('_trial_') or '_trial_' in item
            verbose_level = 2 if is_bayesian_trial else 1  # 2表示每个epoch显示一行进度
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose_level
            )
        
        self.training_history = history
        
        # 保存模型
        if save_path:
            self.model.save(save_path)
            print(f"模型已保存到: {save_path}")
            
        # 绘制训练过程
        self._plot_training_history(history, item)
        
        # 计算训练结果（直接使用原始预测值）
        train_pred = self.model.predict(X_train).flatten()
        test_pred = self.model.predict(X_test).flatten()
        
        # 在贝叶斯优化期间使用最小化指标计算
        is_bayesian_trial = item.endswith('_trial_') or '_trial_' in item
        train_metrics = self._calculate_metrics(y_train, train_pred, "训练集", minimal=is_bayesian_trial)
        test_metrics = self._calculate_metrics(y_test, test_pred, "测试集", minimal=is_bayesian_trial)
        
        # 在贝叶斯优化期间减少调试输出
        if not (item.endswith('_trial_') or '_trial_' in item):
            # 添加调试信息：预测值范围和标准差
            print(f"训练集预测值范围: [{train_pred.min():.4f}, {train_pred.max():.4f}], 标准差: {train_pred.std():.4f}")
            print(f"测试集预测值范围: [{test_pred.min():.4f}, {test_pred.max():.4f}], 标准差: {test_pred.std():.4f}")
            
            # 显示损失值（原始尺度）
            print(f"结果: Train R²={train_metrics['r2']:.4f}, Test R²={test_metrics['r2']:.4f}")
            print(f"Huber Loss: Train={train_metrics['huber_loss']:.4f}, Test={test_metrics['huber_loss']:.4f}")
        
        results = {
            'model': self.model,
            'history': history,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'model_path': save_path if save_path else 'models/best_model.h5'
        }
        
        print(f"训练完成 - 训练集R²: {train_metrics['r2']:.4f}, 测试集R²: {test_metrics['r2']:.4f}")
        
        return results
        
    def evaluate_model(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      model_path: str = None, item: str = "异丙醇",
                      train_sample_ids: np.ndarray = None,
                      test_sample_ids: np.ndarray = None,
                      enable_plotting: bool = True) -> Dict:
        """
        评估函数(需要返回评估结果),用EnhancedPlotManager画出来
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            model_path: 模型路径
            item: 物质名称
            train_sample_ids: 训练集样本ID
            test_sample_ids: 测试集样本ID
            
        Returns:
            评估结果字典
        """
        print(f"\n=== 评估 {item} 神经网络模型 ===")
        
        # 加载模型（如果提供路径）
        if model_path and os.path.exists(model_path):
            # 导入损失函数
            from utils.NeuralNetwork import huber_loss_tf, weighted_huber_loss_tf
            self.model = keras.models.load_model(model_path, 
                                                custom_objects={
                                                    'huber_loss_tf': huber_loss_tf,
                                                    'weighted_huber_loss_tf': weighted_huber_loss_tf
                                                })
            print(f"已加载模型: {model_path}")
        elif self.model is None:
            raise ValueError("没有可用的模型进行评估")
            
        # 预测
        train_pred_scaled = self.model.predict(X_train)
        test_pred_scaled = self.model.predict(X_test)
        
        # 如果模型使用了y标准化，需要反标准化
        if self.y_scaled:
            train_pred = self.y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
            test_pred = self.y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        else:
            train_pred = train_pred_scaled.flatten()
            test_pred = test_pred_scaled.flatten()
        
        # 计算指标
        train_metrics = self._calculate_metrics(y_train, train_pred, "训练集")
        test_metrics = self._calculate_metrics(y_test, test_pred, "测试集")
        
        # 绘制评估结果（可选）
        if enable_plotting:
            # 训练集预测散点图
            self.plot_manager.plot_prediction_scatter(
                y_true=y_train.flatten(),
                y_pred=train_pred.flatten(),
                title=f'{item} - 训练集预测结果'
            )
            
            # 测试集预测散点图
            self.plot_manager.plot_prediction_scatter(
                y_true=y_test.flatten(),
                y_pred=test_pred.flatten(),
                title=f'{item} - 测试集预测结果'
            )
        
        # 打印评估结果
        print(f"\n=== {item} 模型评估结果 ===")
        print(f"训练集 - R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}")
        print(f"测试集 - R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'model': self.model
        }
        
    def bayesian_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             n_trials: int = 10, epochs: int = 100,
                             batch_size: int = 32, item: str = "异丙醇",
                             loss_function: str = 'mean_squared_error',
                             previous_results_file: str = None,
                             n_repeats: int = 2, enable_plotting: bool = True) -> Dict:
        """
        贝叶斯优化,可选择是否输出训练过程图片
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            n_trials: 优化试验次数
            epochs: 每次训练的轮数
            batch_size: 批次大小
            item: 物质名称
            loss_function: 损失函数类型
            previous_results_file: 历史结果文件路径
            n_repeats: 每个参数组合重复训练次数，默认为2
            enable_plotting: 是否绘制训练历史图像，默认为True
            
        Returns:
            优化结果字典
            y_test: 测试集标签
            n_trials: 优化试验次数
            epochs: 每次训练的轮数
            batch_size: 批次大小
            item: 物质名称
            loss_function: 损失函数类型
            previous_results_file: 历史优化结果CSV文件路径
            
        Returns:
            优化结果字典
        """
        print(f"\n=== 开始 {item} 贝叶斯优化 ===")
        print(f"优化试验次数: {n_trials}")
        
        # 存储优化历史
        optimization_history = []
        
        # 加载历史优化结果
        if previous_results_file and os.path.exists(previous_results_file):
            print(f"正在加载历史优化结果: {previous_results_file}")
            try:
                previous_df = pd.read_csv(previous_results_file)
                print(f"加载了 {len(previous_df)} 条历史记录")
                
                # 转换历史数据为optimization_history格式
                for _, row in previous_df.iterrows():
                    params = {}
                    for param in ['DenseN', 'DropoutR', 'C1_K', 'C1_S', 'C2_K', 'C2_S']:
                        if param in row:
                            params[param] = row[param]
                    
                    trial_result = {
                        'trial': int(row['trial']),
                        'params': params,
                        'train_r2': row['train_r2'],
                        'test_r2': row['test_r2'],
                        'huber_loss': row['huber_loss']
                    }
                    
                    # 添加新的huber_loss列（如果存在）
                    if 'train_huber_loss' in row and pd.notna(row['train_huber_loss']):
                        trial_result['train_huber_loss'] = row['train_huber_loss']
                    if 'test_huber_loss' in row and pd.notna(row['test_huber_loss']):
                        trial_result['test_huber_loss'] = row['test_huber_loss']
                    
                    if 'test_rmse' in row and pd.notna(row['test_rmse']):
                        trial_result['test_rmse'] = row['test_rmse']
                    
                    optimization_history.append(trial_result)
                    
            except Exception as e:
                print(f"加载历史结果失败: {e}")
                print("将从头开始优化")
        
        def objective(trial):
            # 定义超参数搜索空间
            params = {
                'DenseN': trial.suggest_int('DenseN', 32, 128, step=16),
                'DropoutR': trial.suggest_float('DropoutR', 0.0, 0.1),
                'C1_K': trial.suggest_int('C1_K', 2, 16, step=2),
                'C1_S': trial.suggest_int('C1_S', 4, 64, step=4),
                'C2_K': trial.suggest_int('C2_K', 4, 32, step=4),
                'C2_S': trial.suggest_int('C2_S', 4, 64, step=4)
            }
            
            try:
                # 计算当前试验的总编号（包括历史记录）
                current_trial_num = len([h for h in optimization_history if 'trial' in h]) + trial.number + 1
                print(f"\n--- Trial {current_trial_num}/{len(optimization_history) + n_trials} ---")
                print(f"参数: {params}")
                print(f"重复训练次数: {n_repeats}")
                
                # 临时禁用绘图功能
                original_plot_manager = self.plot_manager
                self.plot_manager = None
                
                # 多次训练并收集结果
                all_results = []
                for repeat_idx in range(n_repeats):
                    print(f"  第 {repeat_idx + 1}/{n_repeats} 次训练...")
                    
                    # 调用训练函数
                    results = self.train_model(
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        model_params=params,
                        epochs=epochs, batch_size=batch_size,
                        item=f"{item}_trial_{current_trial_num}_repeat_{repeat_idx + 1}",
                        loss_function=loss_function,
                        is_callback=False
                    )
                    all_results.append(results)
                    
                    # 显示单次结果
                    train_huber = results['train_metrics']['huber_loss']
                    test_huber = results['test_metrics']['huber_loss']
                    test_r2 = results['test_metrics']['r2']
                    print(f"    第{repeat_idx + 1}次: Test R²={test_r2:.4f}, Train Huber={train_huber:.4f}, Test Huber={test_huber:.4f}")
                
                # 恢复绘图功能
                self.plot_manager = original_plot_manager
                
                # 计算平均结果
                avg_train_r2 = np.mean([r['train_metrics']['r2'] for r in all_results])
                avg_test_r2 = np.mean([r['test_metrics']['r2'] for r in all_results])
                avg_train_huber_loss = np.mean([r['train_metrics']['huber_loss'] for r in all_results])
                avg_test_huber_loss = np.mean([r['test_metrics']['huber_loss'] for r in all_results])
                avg_huber_loss = (avg_train_huber_loss + avg_test_huber_loss) / 2
                
                # 计算标准差
                std_train_r2 = np.std([r['train_metrics']['r2'] for r in all_results])
                std_test_r2 = np.std([r['test_metrics']['r2'] for r in all_results])
                std_train_huber_loss = np.std([r['train_metrics']['huber_loss'] for r in all_results])
                std_test_huber_loss = np.std([r['test_metrics']['huber_loss'] for r in all_results])
                
                # 绘制最后一次训练的历史图像（代表该参数组合的训练过程）
                if enable_plotting and self.plot_manager is not None and all_results:
                    last_result = all_results[-1]  # 使用最后一次训练的结果
                    plot_item_name = f"{item}_trial_{current_trial_num}_avg_result"
                    self._plot_training_history(last_result['history'], plot_item_name)
                    print(f"已绘制Trial {current_trial_num}的训练历史图像")
                elif not enable_plotting:
                    print(f"Trial {current_trial_num} 完成（跳过绘图）")
                
                trial_result = {
                    'trial': current_trial_num - 1,  # 保持从0开始的编号
                    'params': params,
                    'n_repeats': n_repeats,
                    'train_r2': avg_train_r2,
                    'test_r2': avg_test_r2,
                    'train_huber_loss': avg_train_huber_loss,
                    'test_huber_loss': avg_test_huber_loss,
                    'huber_loss': avg_huber_loss,
                    'train_r2_std': std_train_r2,
                    'test_r2_std': std_test_r2,
                    'train_huber_loss_std': std_train_huber_loss,
                    'test_huber_loss_std': std_test_huber_loss
                }
                
                # 如果有rmse信息则添加（非minimal模式）
                if 'rmse' in all_results[0]['test_metrics']:
                    avg_test_rmse = np.mean([r['test_metrics']['rmse'] for r in all_results])
                    std_test_rmse = np.std([r['test_metrics']['rmse'] for r in all_results])
                    trial_result['test_rmse'] = avg_test_rmse
                    trial_result['test_rmse_std'] = std_test_rmse
                    
                optimization_history.append(trial_result)
                
                # 立即保存当前试验结果到CSV文件
                self._save_single_trial_result(trial_result, item, previous_results_file)
                
                # 显示平均结果
                print(f"平均结果: Test R²={avg_test_r2:.4f}(±{std_test_r2:.4f}), Train Huber={avg_train_huber_loss:.4f}(±{std_train_huber_loss:.4f}), Test Huber={avg_test_huber_loss:.4f}(±{std_test_huber_loss:.4f}), Avg Huber={avg_huber_loss:.4f}")
                
                # 返回优化目标（最小化平均测试集Huber损失）
                return avg_test_huber_loss
                
            except Exception as e:
                print(f"Trial {current_trial_num} 失败: {e}")
                # 恢复绘图功能
                self.plot_manager = original_plot_manager
                return float('inf')
                
        # 创建优化器
        study = optuna.create_study(direction='minimize')
        
        # 如果有历史结果，将其添加到study中
        if optimization_history:
            print(f"将 {len(optimization_history)} 条历史记录添加到优化器中")
            for hist in optimization_history:
                try:
                    # 创建trial并设置参数
                    trial = optuna.trial.create_trial(
                        params=hist['params'],
                        distributions={
                            'DenseN': optuna.distributions.IntDistribution(32, 128, step=16),
                            'DropoutR': optuna.distributions.FloatDistribution(0.0, 0.1),
                            'C1_K': optuna.distributions.IntDistribution(2, 16, step=2),
                            'C1_S': optuna.distributions.IntDistribution(4, 64, step=4),
                            'C2_K': optuna.distributions.IntDistribution(4, 32, step=4),
                            'C2_S': optuna.distributions.IntDistribution(4, 64, step=4)
                        },
                        value=hist['huber_loss']
                    )
                    study.add_trial(trial)
                except Exception as e:
                    print(f"添加历史记录失败 (trial {hist['trial']}): {e}")
        
        # 继续优化
        current_trial_count = len(study.trials)
        remaining_trials = max(0, n_trials - current_trial_count)
        
        if remaining_trials > 0:
            print(f"继续进行 {remaining_trials} 次新的优化试验")
            study.optimize(objective, n_trials=remaining_trials)
        else:
            print("已达到指定的试验次数，无需进行新的优化")
        
        # 保存最佳参数
        self.best_params = study.best_params
        
        # 保存优化结果到CSV并生成热力图
        self._save_optimization_results(optimization_history, item)
        
        print(f"\n=== {item} 贝叶斯优化完成 ===")
        print(f"最佳参数: {self.best_params}")
        print(f"最佳Huber损失: {study.best_value:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_value': study.best_value,
            'study': study,
            'optimization_history': optimization_history
        }
        
    def visualize_activations(self, X_test: np.ndarray, wavelengths: np.ndarray = None,
                             item: str = "异丙醇", sample_indices: List[int] = None) -> None:
        """
        激活层显示函数
        
        Args:
            X_test: 测试数据
            wavelengths: 波长数组
            item: 物质名称
            sample_indices: 要可视化的样本索引列表
        """
        if self.model is None:
            raise ValueError("没有可用的模型进行激活层可视化")
            
        print(f"\n=== {item} CNN激活层可视化 ===")
        
        # 如果没有指定样本，随机选择几个
        if sample_indices is None:
            sample_indices = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)
            
        # 创建激活层模型
        layer_outputs = []
        layer_names = []
        
        for i, layer in enumerate(self.model.layers):
            if 'conv1d' in layer.name.lower():
                layer_outputs.append(layer.output)
                layer_names.append(f"Conv1D_{i+1}")
                
        if not layer_outputs:
            print("模型中没有找到卷积层")
            return
            
        activation_model = keras.Model(inputs=self.model.input, outputs=layer_outputs)
        
        # 对选定样本进行激活层可视化
        for idx in sample_indices:
            sample_input = X_test[idx:idx+1]
            activations = activation_model.predict(sample_input)
            
            self._plot_activations(activations, layer_names, sample_input[0], 
                                 wavelengths, f"{item}_sample_{idx}")
                                 
    def _plot_training_history(self, history, item: str) -> None:
        """绘制训练历史"""
        # 如果plot_manager为None，跳过绘图
        if self.plot_manager is None:
            return
            
        import matplotlib.pyplot as plt
        
        epochs = list(range(1, len(history.history['loss']) + 1))
        
        # 创建包含训练和验证损失的综合图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制损失曲线
        ax1.plot(epochs, history.history['loss'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        
        # 获取最终损失值
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        # 在图上标注最终损失值
        ax1.annotate(f'最终训练损失: {final_train_loss:.4f}', 
                    xy=(epochs[-1], final_train_loss), 
                    xytext=(epochs[-1]*0.7, final_train_loss*1.1),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=10, color='blue')
        
        ax1.annotate(f'最终验证损失: {final_val_loss:.4f}', 
                    xy=(epochs[-1], final_val_loss), 
                    xytext=(epochs[-1]*0.7, final_val_loss*0.9),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        ax1.set_title(f'{item} - 训练过程损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Huber Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制MAE曲线
        ax2.plot(epochs, history.history['mean_absolute_error'], 'g-', label='训练MAE', linewidth=2)
        ax2.plot(epochs, history.history['val_mean_absolute_error'], 'm-', label='验证MAE', linewidth=2)
        
        # 获取最终MAE值
        final_train_mae = history.history['mean_absolute_error'][-1]
        final_val_mae = history.history['val_mean_absolute_error'][-1]
        
        # 在图上标注最终MAE值
        ax2.annotate(f'最终训练MAE: {final_train_mae:.4f}', 
                    xy=(epochs[-1], final_train_mae), 
                    xytext=(epochs[-1]*0.7, final_train_mae*1.1),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green')
        
        ax2.annotate(f'最终验证MAE: {final_val_mae:.4f}', 
                    xy=(epochs[-1], final_val_mae), 
                    xytext=(epochs[-1]*0.7, final_val_mae*0.9),
                    arrowprops=dict(arrowstyle='->', color='magenta'),
                    fontsize=10, color='magenta')
        
        ax2.set_title(f'{item} - 训练过程MAE曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plots_dir = './plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        plt.savefig(f'{plots_dir}/{item}_神经网络训练过程.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"神经网络训练过程图已保存: {plots_dir}/{item}_神经网络训练过程.png")
        print(f"最终训练损失: {final_train_loss:.4f}, 最终验证损失: {final_val_loss:.4f}")
        print(f"最终训练MAE: {final_train_mae:.4f}, 最终验证MAE: {final_val_mae:.4f}")
        
    def _save_optimization_results(self, history: List[Dict], item: str) -> None:
        """保存优化结果到CSV并生成热力图"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 创建png2文件夹
        png2_dir = 'png2'
        os.makedirs(png2_dir, exist_ok=True)
        
        # 转换为DataFrame
        df_results = []
        for h in history:
            row = {
                'trial': h['trial'],
                'train_r2': h['train_r2'],
                'test_r2': h['test_r2'],
                'train_huber_loss': h.get('train_huber_loss', h.get('huber_loss', 0)),  # 向后兼容
                'test_huber_loss': h.get('test_huber_loss', h.get('huber_loss', 0)),   # 向后兼容
                'huber_loss': h['huber_loss']
            }
            # 添加可选的rmse信息
            if 'test_rmse' in h:
                row['test_rmse'] = h['test_rmse']
            
            # 添加标准差信息（如果存在）
            if 'train_r2_std' in h:
                row['train_r2_std'] = h['train_r2_std']
            if 'test_r2_std' in h:
                row['test_r2_std'] = h['test_r2_std']
            if 'train_huber_loss_std' in h:
                row['train_huber_loss_std'] = h['train_huber_loss_std']
            if 'test_huber_loss_std' in h:
                row['test_huber_loss_std'] = h['test_huber_loss_std']
            if 'huber_loss_std' in h:
                row['huber_loss_std'] = h['huber_loss_std']
            
            # 添加参数
            for param_name, param_value in h['params'].items():
                row[param_name] = param_value
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # 保存到CSV文件
        csv_filename = f'bayesian_optimization_{item}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"优化结果已保存到: {csv_filename}")
        
        # 生成参数热力图
        # 动态获取参数列名（排除非参数列）
        non_param_cols = ['trial', 'train_r2', 'test_r2', 'test_rmse', 'train_huber_loss', 'test_huber_loss', 'huber_loss',
                         'train_r2_std', 'test_r2_std', 'train_huber_loss_std', 'test_huber_loss_std', 'huber_loss_std']
        param_cols = [col for col in df.columns if col not in non_param_cols]
        
        # 检查是否有足够的数据进行相关性分析
        if len(param_cols) == 0 or len(df) < 2:
            print("数据不足，跳过热力图生成")
            return
        
        # 创建相关性矩阵（只使用存在的列）
        available_cols = param_cols + [col for col in ['test_rmse', 'test_r2', 'train_huber_loss', 'test_huber_loss', 'huber_loss'] if col in df.columns]
        corr_data = df[available_cols].corr()
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'{item} - 贝叶斯优化参数相关性热力图')
        plt.tight_layout()
        
        # 保存热力图
        heatmap_filename = os.path.join(png2_dir, f'{item}_bayesian_optimization_heatmap.png')
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"参数相关性热力图已保存到: {heatmap_filename}")
        
        # 生成参数vs性能的热力图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(param_cols):
            if i < len(axes):
                # 创建参数值与Huber Loss的散点图
                scatter = axes[i].scatter(df[param], df['huber_loss'], 
                                        c=df['test_r2'], cmap='viridis', alpha=0.7)
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Huber Loss')
                axes[i].set_title(f'{param} vs Huber Loss')
                plt.colorbar(scatter, ax=axes[i], label='Test R²')
        
        # 隐藏多余的子图
        if len(param_cols) < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle(f'{item} - 贝叶斯优化参数性能分析')
        plt.tight_layout()
        
        # 保存参数性能图
        param_perf_filename = os.path.join(png2_dir, f'{item}_bayesian_optimization_params.png')
        plt.savefig(param_perf_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"参数性能分析图已保存到: {param_perf_filename}")
        
    def _save_single_trial_result(self, trial_result: Dict, item: str, csv_filename: str = None) -> None:
        """立即保存单个试验结果到CSV文件"""
        if csv_filename is None:
            csv_filename = f'bayesian_optimization_{item}.csv'
        
        # 准备单行数据
        row = {
            'trial': trial_result['trial'],
            'train_r2': trial_result['train_r2'],
            'test_r2': trial_result['test_r2'],
            'train_huber_loss': trial_result.get('train_huber_loss', trial_result.get('huber_loss', 0)),
            'test_huber_loss': trial_result.get('test_huber_loss', trial_result.get('huber_loss', 0)),
            'huber_loss': trial_result['huber_loss']
        }
        
        # 添加可选的rmse信息
        if 'test_rmse' in trial_result:
            row['test_rmse'] = trial_result['test_rmse']
        
        # 添加标准差信息（如果存在）
        if 'train_r2_std' in trial_result:
            row['train_r2_std'] = trial_result['train_r2_std']
        if 'test_r2_std' in trial_result:
            row['test_r2_std'] = trial_result['test_r2_std']
        if 'train_huber_loss_std' in trial_result:
            row['train_huber_loss_std'] = trial_result['train_huber_loss_std']
        if 'test_huber_loss_std' in trial_result:
            row['test_huber_loss_std'] = trial_result['test_huber_loss_std']
        if 'huber_loss_std' in trial_result:
            row['huber_loss_std'] = trial_result['huber_loss_std']
        
        # 添加参数
        for param_name, param_value in trial_result['params'].items():
            row[param_name] = param_value
        
        # 转换为DataFrame
        df_single = pd.DataFrame([row])
        
        # 检查文件是否存在
        if os.path.exists(csv_filename):
            # 文件存在，追加数据
            df_single.to_csv(csv_filename, mode='a', header=False, index=False)
        else:
            # 文件不存在，创建新文件
            df_single.to_csv(csv_filename, mode='w', header=True, index=False)
        
        print(f"Trial {trial_result['trial'] + 1} 结果已保存到: {csv_filename}")
        
    def _plot_activations(self, activations: List, layer_names: List[str], 
                         input_sample: np.ndarray, wavelengths: np.ndarray, title: str) -> None:
        """绘制激活层"""
        # 使用EnhancedPlotManager的plot_activations方法
        self.plot_manager.plot_activations(
            activations=activations,
            layer_names=layer_names,
            input_sample=input_sample,
            wavelengths=wavelengths,
            title=title
        )
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str, minimal: bool = False) -> Dict:
        """计算评估指标，使用Huber损失保持与训练一致性"""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # 计算原始尺度的指标
        huber_loss = self.loss_manager.huber_loss(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 在贝叶斯优化期间只计算必要指标
        if minimal:
            return {
                'huber_loss': huber_loss,
                'r2': r2,
                'dataset': dataset_name
            }
        
        # 完整指标计算
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'huber_loss': huber_loss,  # 原始尺度的Huber损失
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'dataset': dataset_name
        }
        
    def get_model_summary(self) -> str:
        """获取模型摘要"""
        if self.model is None:
            return "没有可用的模型"
        return self.model.summary()
        
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可用的模型进行保存")
        self.model.save(filepath)
        print(f"模型已保存到: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        self.model = tf.keras.models.load_model(filepath, custom_objects={
            'huber_loss_tf': huber_loss_tf,
            'weighted_huber_loss_tf': weighted_huber_loss_tf
        })
        print(f"模型已从 {filepath} 加载")
    
    def evaluate_saved_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray, 
                           model_path: str = 'best_model.h5', item: str = '异丙醇') -> Dict:
        """
        评估保存的模型 - 从utils_tf.py合并
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            model_path: 模型路径
            item: 物质名称
            
        Returns:
            评估结果字典
        """
        # 加载模型
        self.load_model(model_path)
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 计算指标
        train_metrics = self._calculate_metrics(y_train, y_train_pred, "训练集")
        test_metrics = self._calculate_metrics(y_test, y_test_pred, "测试集")
        
        # 使用EnhancedPlotManager绘制结果
        self.plot_manager.plot_train_test_comparison(
            y_train.flatten(), y_train_pred.flatten(),
            y_test.flatten(), y_test_pred.flatten(),
            train_metrics['r2'], train_metrics['rmse'],
            test_metrics['r2'], test_metrics['rmse'],
            f"{item} 模型评估结果", item
        )
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_predictions': y_train_pred,
            'test_predictions': y_test_pred
        }
    


    
    def predict_new_samples(self, X_new: np.ndarray, model_path: str = None) -> np.ndarray:
        """
        使用训练好的模型预测新样本
        
        Args:
            X_new: 新样本特征
            model_path: 模型路径（如果为None，使用当前模型）
        
        Returns:
            predictions: 预测结果
        """
        if model_path is not None:
            # 加载模型
            model = tf.keras.models.load_model(model_path, custom_objects={'huber_loss_tf': huber_loss_tf})
        else:
            if self.model is None:
                raise ValueError("没有可用的模型，请先训练模型或提供模型路径")
            model = self.model
        
        # 预测并直接返回原始结果
        predictions = model.predict(X_new).flatten()
        
        return predictions
    
    def predict(self, X):
        """预测新数据"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 标准化输入特征
        X_scaled = self.X_scaler.transform(X)
        
        # 预测并直接返回原始结果
        y_pred = self.model.predict(X_scaled).flatten()
        
        return y_pred