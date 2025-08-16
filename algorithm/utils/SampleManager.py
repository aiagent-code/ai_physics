#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### 2. sample_processor.py - 样本管理与数据集分离
**主要功能：样本ID生成、样本级别数据分割**

类功能说明：
- SampleManager: 样本管理器，负责样本ID生成、分组、数据集分割等核心逻辑
  - 根据浓度值生成唯一样本ID
  - 按样本级别分割数据集，确保同一样本的不同测量点不被分离
  - 提供样本统计信息
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional


class SampleManager:
    """样本管理器：负责样本ID生成、分组、数据集分割等核心逻辑"""
    
    def __init__(self, random_seed: int = 42):
        """
        初始化样本管理器
        
        Args:
            random_seed: 随机种子，确保结果可重现
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.sample_mapping = {}  # 浓度值 -> 样本ID的映射
        self.reverse_mapping = {}  # 样本ID -> 浓度值的映射
        
    def generate_sample_ids(self, concentrations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据浓度值生成样本ID和测量次数
        
        Args:
            concentrations: 浓度值数组，形状为(n_samples, n_components)
            
        Returns:
            sample_ids: 样本ID数组，形状为(n_samples,)
            measurement_counts: 测量次数数组，形状为(n_samples,)
        """
        # 确保输入是numpy数组，并转换为数值类型
        concentrations = np.asarray(concentrations, dtype=np.float64)
        
        # 确保concentrations是二维数组
        if concentrations.ndim == 1:
            concentrations = concentrations.reshape(-1, 1)
        
        # 将浓度值转换为字符串，用于生成唯一标识
        concentration_strings = []
        for conc in concentrations:
            # 将浓度值四舍五入到小数点后2位，避免浮点数精度问题
            conc_rounded = np.round(conc, 2)
            if conc_rounded.ndim == 0:  # 标量情况
                conc_str = f"{conc_rounded:.2f}"
            else:  # 数组情况
                conc_str = '_'.join([f"{x:.2f}" for x in conc_rounded])
            concentration_strings.append(conc_str)
        
        # 生成样本ID映射
        unique_concentrations = list(set(concentration_strings))
        sample_ids = []
        measurement_counts = []
        
        for i, conc_str in enumerate(concentration_strings):
            if conc_str not in self.sample_mapping:
                # 新的样本，分配新的ID
                sample_id = len(self.sample_mapping)
                self.sample_mapping[conc_str] = sample_id
                self.reverse_mapping[sample_id] = conc_str
            else:
                sample_id = self.sample_mapping[conc_str]
            
            sample_ids.append(sample_id)
        
        # 计算每个样本的测量次数
        sample_id_counts = {}
        for sample_id in sample_ids:
            sample_id_counts[sample_id] = sample_id_counts.get(sample_id, 0) + 1
        
        measurement_counts = [sample_id_counts[sid] for sid in sample_ids]
        
        return np.array(sample_ids), np.array(measurement_counts)
    
    def split_by_sample_id(self, X: np.ndarray, y: np.ndarray, sample_ids: np.ndarray,
                          train_ratio: float = 0.7, test_ratio: float = 0.2, 
                          val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                          np.ndarray, np.ndarray, np.ndarray]:
        """
        按样本ID分割数据集，确保同一样本的所有测量都在同一个集合中
        
        Args:
            X: 特征数据，形状为(n_samples, n_features)
            y: 标签数据，形状为(n_samples, n_targets)
            sample_ids: 样本ID数组，形状为(n_samples,)
            train_ratio: 训练集比例
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            
        Returns:
            (X_train, y_train, X_test, y_test, X_val, y_val)
        """
        # 检查比例是否合理
        total_ratio = train_ratio + test_ratio + val_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"警告：比例总和不等于1.0 ({total_ratio})，将自动归一化")
            train_ratio /= total_ratio
            test_ratio /= total_ratio
            val_ratio /= total_ratio
        
        # 获取唯一的样本ID
        unique_sample_ids = np.unique(sample_ids)
        n_unique_samples = len(unique_sample_ids)
        
        print(f"总样本数: {len(X)}, 唯一样本数: {n_unique_samples}")
        
        # 随机打乱样本ID顺序
        np.random.shuffle(unique_sample_ids)
        
        # 计算每个集合应该包含的样本数量
        n_train_samples = int(n_unique_samples * train_ratio)
        n_test_samples = int(n_unique_samples * test_ratio)
        n_val_samples = n_unique_samples - n_train_samples - n_test_samples
        
        # 分割样本ID
        train_sample_ids = unique_sample_ids[:n_train_samples]
        test_sample_ids = unique_sample_ids[n_train_samples:n_train_samples + n_test_samples]
        val_sample_ids = unique_sample_ids[n_train_samples + n_test_samples:]
        
        # 根据样本ID筛选数据
        train_mask = np.isin(sample_ids, train_sample_ids)
        test_mask = np.isin(sample_ids, test_sample_ids)
        val_mask = np.isin(sample_ids, val_sample_ids)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        # 打印分割结果
        print(f"训练集: {len(X_train)} 个数据点 ({len(train_sample_ids)} 个样本)")
        print(f"测试集: {len(X_test)} 个数据点 ({len(test_sample_ids)} 个样本)")
        print(f"验证集: {len(X_val)} 个数据点 ({len(val_sample_ids)} 个样本)")
        
        return X_train, y_train, X_test, y_test, X_val, y_val
    
    def get_sample_statistics(self, y: np.ndarray, sample_ids: np.ndarray) -> Dict:
        """
        计算每个样本的统计信息
        
        Args:
            y: 标签数据
            sample_ids: 样本ID数组
            
        Returns:
            包含每个样本统计信息的字典
        """
        stats = {}
        unique_sample_ids = np.unique(sample_ids)
        
        for sample_id in unique_sample_ids:
            mask = sample_ids == sample_id
            sample_values = y[mask]
            
            stats[sample_id] = {
                'count': len(sample_values),
                'mean': np.mean(sample_values),
                'std': np.std(sample_values),
                'min': np.min(sample_values),
                'max': np.max(sample_values),
                'values': sample_values
            }
        
        return stats
    
    def save_sample_info(self, output_dir: str, filename: str = "sample_info.csv"):
        """
        保存样本信息到CSV文件
        
        Args:
            output_dir: 输出目录
            filename: 文件名
        """
        if not self.sample_mapping:
            print("没有样本信息可保存")
            return
        
        # 创建样本信息DataFrame
        sample_info = []
        for sample_id, conc_str in self.reverse_mapping.items():
            # 解析浓度字符串
            concentrations = [float(x) for x in conc_str.split('_')]
            
            info = {
                'sample_id': sample_id,
                'concentration_string': conc_str
            }
            
            # 添加各个组分的浓度
            for i, conc in enumerate(concentrations):
                info[f'component_{i+1}'] = conc
            
            sample_info.append(info)
        
        df = pd.DataFrame(sample_info)
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"样本信息已保存到: {output_path}")
    
    def get_sample_id_by_concentration(self, concentration: np.ndarray) -> Optional[int]:
        """
        根据浓度值获取样本ID
        
        Args:
            concentration: 浓度值数组
            
        Returns:
            样本ID，如果不存在则返回None
        """
        conc_str = '_'.join([f"{x:.2f}" for x in np.round(concentration, 2)])
        return self.sample_mapping.get(conc_str)
    
    def get_concentration_by_sample_id(self, sample_id: int) -> Optional[np.ndarray]:
        """
        根据样本ID获取浓度值
        
        Args:
            sample_id: 样本ID
            
        Returns:
            浓度值数组，如果不存在则返回None
        """
        conc_str = self.reverse_mapping.get(sample_id)
        if conc_str:
            return np.array([float(x) for x in conc_str.split('_')])
        return None


def create_sample_aware_data(x_path: str, y_path: str, output_dir: str, 
                           sample_manager: SampleManager) -> Tuple[str, str]:
    """
    创建包含样本ID的数据文件
    
    Args:
        x_path: 原始X数据文件路径
        y_path: 原始Y数据文件路径
        output_dir: 输出目录
        sample_manager: 样本管理器实例
        
    Returns:
        (x_enhanced_path, y_enhanced_path) 增强后的数据文件路径
    """
    # 读取原始数据
    X = pd.read_csv(x_path, header=None).values
    y = pd.read_csv(y_path, header=None).values
    
    # 生成样本ID和测量次数
    sample_ids, measurement_counts = sample_manager.generate_sample_ids(y)
    
    # 创建增强的Y数据（包含浓度、样本ID、测量次数）
    y_enhanced = np.column_stack([y, sample_ids, measurement_counts])
    
    # 保存增强的数据
    x_enhanced_path = os.path.join(output_dir, 'x_enhanced.csv')
    y_enhanced_path = os.path.join(output_dir, 'y_enhanced.csv')
    
    pd.DataFrame(X).to_csv(x_enhanced_path, index=False, header=False)
    pd.DataFrame(y_enhanced).to_csv(y_enhanced_path, index=False, header=False)
    
    # 保存样本信息
    sample_manager.save_sample_info(output_dir)
    
    print(f"增强数据已保存:")
    print(f"X: {x_enhanced_path}")
    print(f"Y: {y_enhanced_path}")
    print(f"Y数据包含: 浓度值 + 样本ID + 测量次数")
    
    return x_enhanced_path, y_enhanced_path


def load_sample_aware_data(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载包含样本ID的数据
    
    Args:
        x_path: X数据文件路径
        y_path: Y数据文件路径
        
    Returns:
        (X, y, sample_ids, measurement_counts)
    """
    X = pd.read_csv(x_path, header=None).values
    y_enhanced = pd.read_csv(y_path, header=None).values
    
    # 分离浓度值、样本ID和测量次数
    n_components = y_enhanced.shape[1] - 2  # 减去样本ID和测量次数列
    
    y = y_enhanced[:, :n_components]  # 浓度值
    sample_ids = y_enhanced[:, n_components].astype(int)  # 样本ID
    measurement_counts = y_enhanced[:, n_components + 1].astype(int)  # 测量次数
    
    return X, y, sample_ids, measurement_counts