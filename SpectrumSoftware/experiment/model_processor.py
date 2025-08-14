# -*- coding: utf-8 -*-
"""
模型处理器 - 处理神经网络模型加载和浓度预测
Function: 加载训练好的神经网络模型，对光谱数据进行浓度预测
Purpose:
- 加载和管理神经网络模型文件
- 对输入的光谱数据进行预处理
- 调用模型进行浓度预测
- 返回预测的浓度值
Stored Data:
- 模型对象 (model)
- 模型路径 (model_path)
- 预处理参数 (preprocessing_params)
"""

import numpy as np
import os
from typing import Optional, Tuple, Dict, Any
import pickle
import json

class ModelProcessor:
    """模型处理器 - 处理神经网络模型的加载和预测"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.preprocessing_params = None
        self.is_loaded = False
        
    def load_model(self, model_path: str) -> bool:
        """
        加载神经网络模型
        Args:
            model_path: 模型文件路径
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 根据文件扩展名选择加载方式
            file_ext = os.path.splitext(model_path)[1].lower()
            
            if file_ext == '.pkl':
                # 加载pickle格式的模型
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict):
                        self.model = model_data.get('model')
                        self.preprocessing_params = model_data.get('preprocessing_params', {})
                    else:
                        self.model = model_data
                        self.preprocessing_params = {}
                        
            elif file_ext in ['.h5', '.hdf5']:
                # 加载Keras/TensorFlow模型
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_path)
                    # 尝试加载预处理参数
                    params_path = model_path.replace(file_ext, '_params.json')
                    if os.path.exists(params_path):
                        with open(params_path, 'r') as f:
                            self.preprocessing_params = json.load(f)
                    else:
                        self.preprocessing_params = {}
                except ImportError:
                    raise ImportError("需要安装tensorflow来加载.h5/.hdf5格式的模型")
                    
            elif file_ext == '.joblib':
                # 加载joblib格式的模型
                try:
                    import joblib
                    model_data = joblib.load(model_path)
                    if isinstance(model_data, dict):
                        self.model = model_data.get('model')
                        self.preprocessing_params = model_data.get('preprocessing_params', {})
                    else:
                        self.model = model_data
                        self.preprocessing_params = {}
                except ImportError:
                    raise ImportError("需要安装joblib来加载.joblib格式的模型")
                    
            else:
                raise ValueError(f"不支持的模型文件格式: {file_ext}")
            
            if self.model is None:
                raise ValueError("模型加载失败，模型对象为空")
                
            self.model_path = model_path
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.model = None
            self.model_path = None
            self.preprocessing_params = None
            self.is_loaded = False
            return False
    
    def preprocess_spectrum(self, spectrum_data: np.ndarray, wavelengths: np.ndarray = None) -> np.ndarray:
        """
        预处理光谱数据
        Args:
            spectrum_data: 光谱强度数据
            wavelengths: 波长数据（可选）
        Returns:
            np.ndarray: 预处理后的数据
        """
        if spectrum_data is None:
            raise ValueError("光谱数据不能为空")
            
        # 确保数据为numpy数组
        data = np.array(spectrum_data)
        
        # 应用预处理参数
        if self.preprocessing_params:
            # 归一化
            if 'normalize' in self.preprocessing_params and self.preprocessing_params['normalize']:
                data_min = self.preprocessing_params.get('data_min', np.min(data))
                data_max = self.preprocessing_params.get('data_max', np.max(data))
                if data_max != data_min:
                    data = (data - data_min) / (data_max - data_min)
            
            # 标准化
            if 'standardize' in self.preprocessing_params and self.preprocessing_params['standardize']:
                data_mean = self.preprocessing_params.get('data_mean', np.mean(data))
                data_std = self.preprocessing_params.get('data_std', np.std(data))
                if data_std != 0:
                    data = (data - data_mean) / data_std
            
            # 波长范围选择
            if wavelengths is not None and 'wavelength_range' in self.preprocessing_params:
                wl_range = self.preprocessing_params['wavelength_range']
                if len(wl_range) == 2:
                    mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
                    data = data[mask]
        
        # 确保数据形状正确（添加batch维度）
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            
        return data
    
    def predict_concentration(self, spectrum_data: np.ndarray, wavelengths: np.ndarray = None) -> float:
        """
        预测浓度值
        Args:
            spectrum_data: 光谱强度数据
            wavelengths: 波长数据（可选）
        Returns:
            float: 预测的浓度值
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先加载模型")
            
        try:
            # 预处理数据
            processed_data = self.preprocess_spectrum(spectrum_data, wavelengths)
            
            # 进行预测
            if hasattr(self.model, 'predict'):
                # Keras/TensorFlow模型或sklearn模型
                prediction = self.model.predict(processed_data)
            elif hasattr(self.model, '__call__'):
                # 可调用对象
                prediction = self.model(processed_data)
            else:
                raise ValueError("模型对象不支持预测操作")
            
            # 处理预测结果
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    # 多维输出，取第一个值
                    concentration = float(prediction[0, 0])
                else:
                    concentration = float(prediction[0])
            else:
                concentration = float(prediction)
            
            return concentration
            
        except Exception as e:
            print(f"浓度预测失败: {str(e)}")
            raise RuntimeError(f"浓度预测失败: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        Returns:
            Dict: 模型信息字典
        """
        info = {
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'model_type': type(self.model).__name__ if self.model else None,
            'preprocessing_params': self.preprocessing_params
        }
        
        if self.is_loaded and hasattr(self.model, 'summary'):
            # Keras模型
            try:
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    self.model.summary()
                info['model_summary'] = f.getvalue()
            except:
                pass
                
        return info
    
    def unload_model(self):
        """卸载模型"""
        self.model = None
        self.model_path = None
        self.preprocessing_params = None
        self.is_loaded = False