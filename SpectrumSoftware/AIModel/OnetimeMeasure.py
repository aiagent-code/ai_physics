#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### OnetimeMeasure.py - 单次测量预测模块
**主要功能：对单次测量数据进行预处理和神经网络预测**

主要功能：
- 单次测量数据预处理（波段选择、面积归一化）
- 神经网络模型加载和预测
- 返回预测结果

主要方法：
- predict_single_measurement(): 单次测量预测主函数
- preprocess_single_spectrum(): 单个光谱预处理
- area_normalize(): 面积归一化
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class OnetimeMeasure:
    """单次测量预测器：用于单个光谱的预处理和预测"""
    
    def __init__(self):
        """
        初始化单次测量预测器
        """
        self.model = None
        self.model_path = None
    
    def area_normalize(self, spectrum: np.ndarray) -> np.ndarray:
        """
        面积归一化
        
        参数:
        spectrum: 光谱数据数组
        
        返回:
        np.ndarray: 归一化后的光谱数据
        """
        # 计算光谱面积（积分）
        area = np.trapz(np.abs(spectrum))
        
        # 避免除零错误
        if area == 0:
            return spectrum
        
        # 面积归一化
        normalized_spectrum = spectrum / area
        
        return normalized_spectrum
    
    def preprocess_single_spectrum(self, wavenumbers: List[float], 
                                 intensities: List[float]) -> np.ndarray:
        """
        单个光谱预处理
        
        参数:
        wavenumbers: 波数列表（暂时不使用，但保留接口）
        intensities: 相对光强列表
        
        返回:
        np.ndarray: 预处理后的光谱数据
        """
        # 转换为numpy数组
        spectrum = np.array(intensities)
        
        # 选择30-430范围的数据点（假设这是索引范围）
        if len(spectrum) > 430:
            spectrum_selected = spectrum[30:430]
        else:
            # 如果数据长度不足，使用全部数据
            print(f"警告：数据长度({len(spectrum)})小于430，使用全部数据")
            spectrum_selected = spectrum
        
        # 面积归一化
        spectrum_normalized = self.area_normalize(spectrum_selected)
        
        return spectrum_normalized
    
    def load_model(self, model_path: str) -> bool:
        """
        加载神经网络模型
        
        参数:
        model_path: 模型文件路径
        
        返回:
        bool: 是否成功加载模型
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            print(f"模型加载成功: {model_path}")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def predict_single_measurement(self, wavenumbers: List[float], 
                                 intensities: List[float], 
                                 model_path: str) -> Optional[float]:
        """
        单次测量预测主函数
        
        参数:
        wavenumbers: 波数列表
        intensities: 相对光强列表
        model_path: 模型文件路径
        
        返回:
        Optional[float]: 预测结果，失败时返回None
        """
        try:
            # 加载模型（如果路径不同或模型未加载）
            if self.model is None or self.model_path != model_path:
                if not self.load_model(model_path):
                    return None
            
            # 预处理光谱数据
            processed_spectrum = self.preprocess_single_spectrum(wavenumbers, intensities)
            
            # 重塑数据为模型输入格式 (1, features)
            input_data = processed_spectrum.reshape(1, -1)
            
            # 进行预测
            prediction = self.model.predict(input_data, verbose=0)
            
            # 返回预测结果（假设是回归任务，返回标量值）
            result = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
            
            print(f"预测完成，结果: {result}")
            return result
            
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """
        获取当前加载模型的信息
        
        返回:
        dict: 模型信息字典
        """
        if self.model is None:
            return {"status": "未加载模型"}
        
        try:
            return {
                "status": "已加载",
                "model_path": self.model_path,
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "total_params": self.model.count_params()
            }
        except Exception as e:
            return {"status": f"获取模型信息失败: {e}"}


# 便捷函数接口
def predict_single_measurement(wavenumbers: List[float], 
                             intensities: List[float], 
                             model_path: str) -> Optional[float]:
    """
    便捷函数：单次测量预测
    
    参数:
    wavenumbers: 波数列表
    intensities: 相对光强列表
    model_path: 模型文件路径
    
    返回:
    Optional[float]: 预测结果，失败时返回None
    """
    predictor = OnetimeMeasure()
    return predictor.predict_single_measurement(wavenumbers, intensities, model_path)


# 测试代码
if __name__ == "__main__":
    # 示例用法
    print("=== OnetimeMeasure 测试 ===")
    
    # 创建示例数据
    example_wavenumbers = list(range(1000))  # 示例波数
    example_intensities = np.random.rand(1000).tolist()  # 示例光强
    example_model_path = "./models/best_model.h5"  # 示例模型路径
    
    # 使用类接口
    predictor = OnetimeMeasure()
    result = predictor.predict_single_measurement(
        example_wavenumbers, 
        example_intensities, 
        example_model_path
    )
    
    if result is not None:
        print(f"预测结果: {result}")
    else:
        print("预测失败")
    
    # 显示模型信息
    model_info = predictor.get_model_info()
    print(f"模型信息: {model_info}")
    
    print("\n=== 使用便捷函数接口 ===")
    # 使用便捷函数接口
    result2 = predict_single_measurement(
        example_wavenumbers, 
        example_intensities, 
        example_model_path
    )
    
    if result2 is not None:
        print(f"便捷函数预测结果: {result2}")
    else:
        print("便捷函数预测失败")