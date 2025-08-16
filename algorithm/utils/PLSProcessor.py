#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### 4. pls_processor.py - PLS算法
**主要功能：PLS回归算法实现**

类功能说明：
- PLSProcessor: PLS算法处理类
  - PLS回归模型训练和预测
  - 自动选择最优组件数
  - 异常值检测和剔除（3σ原则）
  - 模型性能评估
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

# 自定义huber_loss函数
def huber_loss(y_true, y_pred, delta=1.0):
    """计算Huber损失"""
    residual = np.abs(y_true - y_pred)
    condition = residual <= delta
    squared_loss = 0.5 * (residual ** 2)
    linear_loss = delta * residual - 0.5 * (delta ** 2)
    return np.mean(np.where(condition, squared_loss, linear_loss))
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt  # 移除直接使用plt，改用EnhancedPlotManager
import warnings
warnings.filterwarnings('ignore')
from .LossManager import LossManager
from .AbnormalValueDetector import AbnormalValueDetector
from .EnhancedPlotManager import SpecializedPlotManager

class PLSProcessor:
    """
    PLS算法处理类
    
    功能：
    - PLS回归模型训练和预测
    - 自动选择最优组件数
    - 异常值检测和剔除
    - 模型性能评估
    """
    
    def __init__(self):
        self.best_components = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.loss_manager = LossManager()
        self.outlier_detector = AbnormalValueDetector()
    
    def pls_train(self, X_calib, Y_calib, X_valid, Y_valid, n_components=None, loss='huber', item="异丙醇", train_sample_ids=None, test_sample_ids=None):
        """
        PLS训练函数 - 使用EnhancedPlotManager画出训练结果
        
        参数:
        X_calib: 校准集特征
        Y_calib: 校准集标签
        X_valid: 验证集特征
        Y_valid: 验证集标签
        n_components: 指定组件数，如果为None则自动选择
        loss: 损失函数类型
        item: 物质名称
        
        返回:
        model: 训练好的模型
        Y_calib_pred: 校准集预测值
        Y_valid_pred: 验证集预测值
        """
        print("开始PLS训练...")
        
        # 数据标准化
        X_calib_scaled = self.scaler.fit_transform(X_calib)
        X_valid_scaled = self.scaler.transform(X_valid)
        
        # 选择组件数
        if n_components is None:
            best_components = self._select_optimal_components(X_calib_scaled, Y_calib, loss)
            self.best_components = best_components
            print(f"自动选择的最优组件数: {best_components}")
        else:
            best_components = n_components
            print(f"使用指定组件数: {best_components}")
        
        # 训练模型
        model = PLSRegression(n_components=best_components)
        model.fit(X_calib_scaled, Y_calib)
        self.best_model = model
        
        # 预测
        Y_calib_pred = model.predict(X_calib_scaled)
        Y_valid_pred = model.predict(X_valid_scaled)
        
        # 使用SpecializedPlotManager绘制PLS专用图像
        from .EnhancedPlotManager import SpecializedPlotManager
        specialized_plot_manager = SpecializedPlotManager()
        
        # 计算评估指标
        train_r2 = r2_score(Y_calib, Y_calib_pred)
        train_rmse = np.sqrt(mean_squared_error(Y_calib, Y_calib_pred))
        test_r2 = r2_score(Y_valid, Y_valid_pred)
        test_rmse = np.sqrt(mean_squared_error(Y_valid, Y_valid_pred))
        
        # 绘制PLS训练和测试结果对比图（用户期望的格式）
        specialized_plot_manager.plot_pls_prediction_results_with_train_test(
            Y_calib.flatten(), Y_calib_pred.flatten(),
            Y_valid.flatten(), Y_valid_pred.flatten(),
            train_r2, train_rmse, test_r2, test_rmse,
            f"{item} PLS回归结果", item
        )
        
        # 如果是自动选择组件数，绘制主成分分析曲线
        if n_components is None:
            self.pls_component_analysis(X_calib_scaled, Y_calib, X_valid_scaled, Y_valid, item=item)
        
        return model, Y_calib_pred, Y_valid_pred
    
    def pls_predict(self, X_new, model=None):
        """
        PLS预测函数 - 使用EnhancedPlotManager画出预测结果
        
        参数:
        X_new: 新样本特征
        model: 训练好的模型，如果为None则使用self.best_model
        
        返回:
        Y_pred: 预测值
        """
        if model is None:
            if self.best_model is None:
                raise ValueError("没有可用的训练模型，请先调用pls_train函数")
            model = self.best_model
        
        # 数据标准化
        X_new_scaled = self.scaler.transform(X_new)
        
        # 预测
        Y_pred = model.predict(X_new_scaled)
        
        print(f"预测完成，共预测 {len(Y_pred)} 个样本")
        
        # 使用EnhancedPlotManager绘制预测结果分布
        from .EnhancedPlotManager import EnhancedPlotManager
        plot_manager = EnhancedPlotManager()
        plot_manager.plot_prediction_distribution(Y_pred.flatten(), "PLS预测结果分布")
        
        return Y_pred
    
    def pls_component_analysis(self, X_calib, Y_calib, X_valid, Y_valid, max_components=20, loss='huber', item="异丙醇"):
        """
        主成分回归函数 - 调用训练函数多次，求得主成分数量-损失值对应关系，并用EnhancedPlotManager画出来
        
        参数:
        X_calib: 校准集特征
        Y_calib: 校准集标签
        X_valid: 验证集特征
        Y_valid: 验证集标签
        max_components: 最大组件数
        loss: 损失函数类型
        item: 物质名称
        
        返回:
        components_range: 组件数范围
        loss_values: 对应的损失值
        best_components: 最优组件数
        """
        print(f"开始主成分分析，最大组件数: {max_components}")
        
        # 数据标准化
        X_calib_scaled = self.scaler.fit_transform(X_calib)
        X_valid_scaled = self.scaler.transform(X_valid)
        
        max_components = min(max_components, X_calib.shape[1], X_calib.shape[0] - 1)
        components_range = list(range(1, max_components + 1))
        loss_values = []
        r2_values = []
        
        for n_comp in components_range:
            print(f"测试组件数: {n_comp}")
            
            # 训练模型
            model = PLSRegression(n_components=n_comp)
            model.fit(X_calib_scaled, Y_calib)
            
            # 预测
            Y_valid_pred = model.predict(X_valid_scaled)
            
            # 计算损失值
            if loss == 'mse':
                loss_val = mean_squared_error(Y_valid, Y_valid_pred)
            else:
                loss_val = self.loss_manager.huber_loss(Y_valid, Y_valid_pred)
            
            r2_val = r2_score(Y_valid, Y_valid_pred)
            
            loss_values.append(loss_val)
            r2_values.append(r2_val)
        
        # 找到最优组件数
        best_idx = np.argmin(loss_values)
        best_components = components_range[best_idx]
        
        print(f"最优组件数: {best_components}, 最小损失值: {loss_values[best_idx]:.4f}")
        
        # 使用EnhancedPlotManager绘制组件分析结果
        from .EnhancedPlotManager import EnhancedPlotManager
        plot_manager = EnhancedPlotManager()
        
        # 绘制损失值-组件数关系图
        plot_manager.plot_component_analysis(
            components_range, loss_values, r2_values,
            best_components, f"{item} 主成分分析结果"
        )
        
        return components_range, loss_values, best_components
    
    def _select_optimal_components(self, X_calib, Y_calib, loss='huber'):
        """
        自动选择最优组件数
        
        参数:
        X_calib: 校准集特征
        Y_calib: 校准集标签
        loss: 损失函数类型
        
        返回:
        best_components: 最优组件数
        """
        max_components = min(X_calib.shape[1], X_calib.shape[0] - 1, 20)
        components_range = range(1, max_components + 1)
        
        mse_scores = []
        huber_scores = []
        
        for n_comp in components_range:
            # 使用交叉验证
            from sklearn.model_selection import cross_val_predict
            y_pred_cv = cross_val_predict(PLSRegression(n_components=n_comp), X_calib, Y_calib, cv=5)
            
            # 计算指标
            mse = mean_squared_error(Y_calib, y_pred_cv)
            huber = self.loss_manager.huber_loss(Y_calib, y_pred_cv)
            
            mse_scores.append(mse)
            huber_scores.append(huber)
        
        # 选择最优组件数
        if loss == 'mse':
            best_components = components_range[np.argmin(mse_scores)]
        else:
            best_components = components_range[np.argmin(huber_scores)]
        
        return best_components
    
    def _evaluate_model(self, y_true=None, y_pred=None, item="异丙醇", plot_results=True,
                           y_train_true=None, y_train_pred=None, y_test_true=None, y_test_pred=None,
                           train_sample_ids=None, test_sample_ids=None):
        """
        统一的模型评估方法，支持单一数据集和训练测试对比两种模式
        
        参数:
        # 单一数据集模式
        y_true: 真实值
        y_pred: 预测值
        item: 物质名称
        plot_results: 是否绘制预测结果图
        
        # 训练测试对比模式
        y_train_true: 训练集真实值
        y_train_pred: 训练集预测值
        y_test_true: 测试集真实值
        y_test_pred: 测试集预测值
        train_sample_ids: 训练集样本ID
        test_sample_ids: 测试集样本ID
        """
        # 判断使用哪种模式
        if y_train_true is not None and y_test_true is not None:
            # 训练测试对比模式
            return self._evaluate_train_test_mode(
                y_train_true, y_train_pred, y_test_true, y_test_pred,
                train_sample_ids, test_sample_ids, item
            )
        elif y_true is not None and y_pred is not None:
            # 单一数据集模式
            return self._evaluate_single_mode(y_true, y_pred, item, plot_results)
        else:
            raise ValueError("必须提供单一数据集参数(y_true, y_pred)或训练测试对比参数")
    
    def _evaluate_single_mode(self, y_true, y_pred, item, plot_results):
        """单一数据集评估模式"""
        # 计算指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        huber = huber_loss(y_true, y_pred)
        sep = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        print(f"\n测试集指标:")
        print(f"R^2: {r2:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Huber Loss: {huber:.3f}")
        print(f"SEP: {sep:.3f}")
        
        # 可视化预测结果
        if plot_results:
            try:
                from .EnhancedPlotManager import EnhancedPlotManager
                plot_manager = EnhancedPlotManager()
                
                config = {
                    'title': f'{item} - 神经网络预测结果',
                    'figsize': (15, 6)
                }
                
                plot_manager.plot_pls_prediction_results(
                    y_true, y_pred, config, model_name="神经网络"
                )
            except Exception as e:
                print(f"绘图时出现错误: {e}")
        
        return {
            'R2': r2,
            'RMSE': rmse,
            'Huber_Loss': huber,
            'SEP': sep
        }
    
    def _evaluate_train_test_mode(self, y_train_true, y_train_pred, y_test_true, y_test_pred,
                                 train_sample_ids, test_sample_ids, item):
        """训练测试对比评估模式"""
        # 计算测试集指标
        r2 = r2_score(y_test_true, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
        huber = huber_loss(y_test_true, y_test_pred)
        sep = np.sqrt(np.mean((y_test_true - y_test_pred) ** 2))
        
        # 计算训练集指标
        train_r2 = r2_score(y_train_true, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
        
        print(f"\n测试集指标:")
        print(f"R^2: {r2:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Huber Loss: {huber:.3f}")
        print(f"SEP: {sep:.3f}")
        
        # 可视化预测结果（包含训练集和测试集）
        try:
            from .EnhancedPlotManager import EnhancedPlotManager
            plot_manager = EnhancedPlotManager()
            
            # 将样本标签转换为数值型ID
            if train_sample_ids is not None and test_sample_ids is not None:
                unique_train_labels = np.unique(train_sample_ids)
                unique_test_labels = np.unique(test_sample_ids)
                all_unique_labels = np.unique(np.concatenate([unique_train_labels, unique_test_labels]))
                
                # 创建标签到ID的映射
                label_to_id = {label: i for i, label in enumerate(all_unique_labels)}
                
                # 转换样本标签为数值型ID
                train_ids_numeric = np.array([label_to_id[label] for label in train_sample_ids])
                test_ids_numeric = np.array([label_to_id[label] for label in test_sample_ids])
            
            config = {
                'title': f'{item} - 神经网络预测结果分析',
                'figsize': (15, 6)
            }
            
            specialized_plot_manager = SpecializedPlotManager()
            specialized_plot_manager.plot_pls_prediction_results_with_train_test(
                y_train_true.flatten(), y_train_pred.flatten(),
                y_test_true.flatten(), y_test_pred.flatten(),
                train_r2, train_rmse, r2, rmse,
                title=f'{item} - 训练测试结果对比', item=item
            )
        except Exception as e:
            print(f"绘图时出现错误: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'R2': r2,
            'RMSE': rmse,
            'Huber_Loss': huber,
            'SEP': sep,
            'train_R2': train_r2,
            'train_RMSE': train_rmse
        }
    
    def _plot_components(self, X_calib, Y_calib, X_valid, Y_valid, item="异丙醇"):
        """
        PLS组件分析和可视化
        
        参数:
        X_calib: 校准集特征
        Y_calib: 校准集标签
        X_valid: 验证集特征
        Y_valid: 验证集标签
        item: 物质名称
        """
        max_components = min(X_calib.shape[1], X_calib.shape[0] - 1, 20)
        components_range = range(1, max_components + 1)
        
        mse_scores = []
        huber_scores = []
        
        for n_comp in components_range:
            # 使用交叉验证
            from sklearn.model_selection import cross_val_predict
            y_pred_cv = cross_val_predict(PLSRegression(n_components=n_comp), X_calib, Y_calib, cv=5)
            
            # 计算指标
            mse = mean_squared_error(Y_calib, y_pred_cv)
            huber = huber_loss(Y_calib, y_pred_cv)
            
            mse_scores.append(mse)
            huber_scores.append(huber)
        
        print(f"\n{item} - PLS组件分析统计:")
        print(f"最优组件数: {self.best_components}")
        print(f"MSE范围: {min(mse_scores):.4f} - {max(mse_scores):.4f}")
        print(f"Huber Loss范围: {min(huber_scores):.4f} - {max(huber_scores):.4f}")
        
        # 调用EnhancedPlotManager进行可视化
        try:
            from .EnhancedPlotManager import EnhancedPlotManager
            plot_manager = EnhancedPlotManager()
            
            config = {
                'title': f'{item} - PLS组件分析',
                'figsize': (15, 6)
            }
            
            specialized_plot_manager = SpecializedPlotManager()
            # 注意：这里需要根据实际的PLS模型获取components和wavelengths
            # 暂时使用占位符，实际使用时需要传入正确的参数
            print("PLS成分分析绘图需要components和wavelengths参数，请使用SpecializedPlotManager.plot_pls_components_analysis方法")
        except Exception as e:
            print(f"绘图时出现错误: {e}")
            print("详细的组件分析图表请使用EnhancedPlotManager类进行绘制")
    

    
    def predict_new_samples(self, X_new, model_path=None):
        """
        使用训练好的模型预测新样本
        
        参数:
        X_new: 新样本特征
        model_path: 模型路径（如果为None，使用当前模型）
        
        返回:
        predictions: 预测结果
        """
        if model_path is not None:
            # 加载模型
            import joblib
            model = joblib.load(model_path)
            scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
        else:
            if self.best_model is None:
                raise ValueError("没有可用的模型，请先训练模型或提供模型路径")
            model = self.best_model
            scaler = self.scaler
        
        # 数据标准化
        X_new_scaled = scaler.transform(X_new)
        
        # 预测
        predictions = model.predict(X_new_scaled)
        
        return predictions
    
    def save_model(self, model_path):
        """
        保存模型
        
        参数:
        model_path: 模型保存路径
        """
        if self.best_model is None:
            raise ValueError("没有可用的模型，请先训练模型")
        
        import joblib
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, model_path.replace('.pkl', '_scaler.pkl'))
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        参数:
        model_path: 模型路径
        """
        import joblib
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
        print(f"模型已从 {model_path} 加载")

    def pls_training(self, X_calib, Y_calib, loss='huber', item="异丙醇"):
        """
        PLS训练方法 - 仅使用训练集进行训练并输出训练损失图
        
        参数:
        X_calib: 校准集特征
        Y_calib: 校准集标签
        loss: 损失函数类型
        item: 物质名称
        
        返回:
        model: 训练好的模型
        Y_calib_pred: 校准集预测值
        """
        print("开始PLS训练（仅训练集）...")
        
        # 数据标准化
        X_calib_scaled = self.scaler.fit_transform(X_calib)
        
        # 选择最优组件数
        best_components = self._select_optimal_components(X_calib_scaled, Y_calib, loss)
        self.best_components = best_components
        print(f"自动选择的最优组件数: {best_components}")
        
        # 训练模型
        model = PLSRegression(n_components=best_components)
        model.fit(X_calib_scaled, Y_calib)
        self.best_model = model
        
        # 预测训练集
        Y_calib_pred = model.predict(X_calib_scaled)
        
        # 绘制训练损失图（主成分分析曲线）
        self.pls_component_analysis_training_only(X_calib_scaled, Y_calib, item=item)
        
        return model, Y_calib_pred
    
    def pls_component_analysis_training_only(self, X_calib, Y_calib, max_components=20, loss='huber', item="异丙醇"):
        """
        PLS主成分分析 - 仅训练集版本，用于训练损失图
        """
        print("开始PLS主成分分析（训练集）...")
        
        max_components = min(max_components, X_calib.shape[1], X_calib.shape[0]-1)
        
        train_scores = []
        components_range = range(1, max_components + 1)
        
        for n_comp in components_range:
            # 训练模型
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_calib, Y_calib)
            
            # 预测训练集
            Y_train_pred = pls.predict(X_calib)
            
            # 计算训练集损失
            if loss == 'huber':
                train_loss = self.loss_manager.huber_loss(Y_calib, Y_train_pred)
            else:
                train_loss = mean_squared_error(Y_calib, Y_train_pred)
            
            train_scores.append(train_loss)
        
        # 绘制训练损失图
        from .EnhancedPlotManager import SpecializedPlotManager
        specialized_plot_manager = SpecializedPlotManager()
        specialized_plot_manager.plot_pls_training_loss(
            components_range, train_scores, item
        )
    
    def pls_prediction(self, x_train, y_train, x_val, y_val, x_test, y_test, 
                      train_sample_ids, val_sample_ids, test_sample_ids, 
                      loss='huber', item="异丙醇"):
        """
        PLS预测方法 - 传入所有数据进行样本点预测，输出预测图像和样本方差图
        
        参数:
        x_train: 训练集特征
        y_train: 训练集标签
        x_val: 验证集特征
        y_val: 验证集标签
        x_test: 测试集特征
        y_test: 测试集标签
        train_sample_ids: 训练集样本ID
        val_sample_ids: 验证集样本ID
        test_sample_ids: 测试集样本ID
        loss: 损失函数类型
        item: 物质名称
        
        返回:
        predictions: 包含所有预测结果的字典
        """
        print("开始PLS预测（所有数据集）...")
        
        # 合并所有数据
        X_all = np.vstack([x_train, x_val, x_test])
        # 确保Y数组是一维的
        y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train
        y_val_flat = y_val.flatten() if y_val.ndim > 1 else y_val
        y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
        Y_all = np.concatenate([y_train_flat, y_val_flat, y_test_flat])
        all_sample_ids = np.concatenate([train_sample_ids, val_sample_ids, test_sample_ids])
        
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(x_train)
        X_all_scaled = self.scaler.transform(X_all)
        
        # 使用训练集训练模型
        if self.best_model is None:
            # 如果没有预训练模型，先训练
            best_components = self._select_optimal_components(X_train_scaled, y_train_flat, loss)
            self.best_components = best_components
            model = PLSRegression(n_components=best_components)
            model.fit(X_train_scaled, y_train_flat)
            self.best_model = model
        else:
            model = self.best_model
        
        # 对所有数据进行预测
        Y_all_pred = model.predict(X_all_scaled)
        
        # 分别获取各数据集的预测结果
        n_train = len(x_train)
        n_val = len(x_val)
        
        y_train_pred = Y_all_pred[:n_train]
        y_val_pred = Y_all_pred[n_train:n_train+n_val]
        y_test_pred = Y_all_pred[n_train+n_val:]
        
        # 使用EnhancedPlotManager绘制预测结果图和样本方差图
        try:
            from .EnhancedPlotManager import EnhancedPlotManager
            plot_manager = EnhancedPlotManager()
            
            # 绘制三个预测图像（训练集、验证集、测试集）
            plot_manager.plot_pls_prediction_with_sample_ids(
                y_train, y_train_pred, train_sample_ids,
                y_val, y_val_pred, val_sample_ids,
                y_test, y_test_pred, test_sample_ids,
                item
            )
            
            # 样本方差图已移至步骤5.5中生成
            
        except Exception as e:
            print(f"绘制预测图像时出现错误: {e}")
        
        # 返回预测结果
        predictions = {
            'y_train_pred': y_train_pred,
            'y_val_pred': y_val_pred,
            'y_test_pred': y_test_pred,
            'y_all_pred': Y_all_pred,
            'model': model
        }
        
        return predictions

# 注意：便捷函数已移除，请直接使用对应的类方法
# 例如：使用 PLSProcessor().pls_prediction() 而不是 pls_prediction()
# 这样可以保持代码的一致性和可维护性