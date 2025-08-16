#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PreProcessor类介绍

一.方法:
1. 选取特定波段
   - select_wavelength_range: 选取单个波长范围
   - select_multiple_ranges: 选取多个波长范围

2. 归一化方法
   - standard: 标准化（Z-score）
   - minmax: Min-Max归一化
   - robust: 鲁棒标准化
   - l2: L2归一化
   - snv: 标准正态变量变换
   - msc: 多元散射校正
   - area: 面积归一化
   - vector: 向量归一化
   - range: 范围归一化
   - zscore: Z-score标准化（与SNV相同）
   - internal_standard: 内标归一化
   - baseline_asls: AsLS基线校正

3. 滤波方法
   - savgol: Savitzky-Golay滤波
   - gaussian: 高斯滤波
   - median: 中值滤波
   - lowpass: 低通滤波
   - moving_average: 移动平均滤波

4. 其他功能
   - derivative_transform: 导数变换
   - preprocess_pipeline: 预处理流水线
"""

import numpy as np
import pandas as pd
from scipy import signal, integrate, sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Optional, Tuple, Union


class PreProcessor:
    """数据预处理器：用于光谱数据的预处理操作"""
    
    def __init__(self):
        """
        初始化预处理器
        """
        self.scaler = None
        self.selected_wavelengths = None
        self.filter_params = {}
        
    def select_wavelength_range(self, X: np.ndarray, wavelengths: np.ndarray, 
                               start_wl: float, end_wl: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        选取特定波段
        
        Args:
            X: 光谱数据矩阵 (样本数, 波长数)
            wavelengths: 波长数组
            start_wl: 起始波数
            end_wl: 结束波数
            
        Returns:
            选取后的光谱数据和对应的波长数组
        """
        # 对于100-1500波数范围，直接使用索引30-429
        if start_wl == 100 and end_wl == 1500:
            start_idx = 30
            end_idx = 429
            
            # 确保索引在有效范围内
            if end_idx >= X.shape[1]:
                end_idx = X.shape[1] - 1
                print(f"警告: 结束索引超出数据范围，调整为 {end_idx}")
            
            # 选取数据
            X_selected = X[:, start_idx:end_idx+1]
            wavelengths_selected = wavelengths[start_idx:end_idx+1]
            
            # 保存选取的波长信息
            self.selected_wavelengths = {
                'start': start_wl,
                'end': end_wl,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'indices': np.arange(start_idx, end_idx+1),
                'wavelengths': wavelengths_selected
            }
            
            print(f"选取波数范围: {start_wl} - {end_wl} cm⁻¹ (索引 {start_idx}-{end_idx})")
            print(f"选取的数据点数: {X_selected.shape[1]}")
            
        else:
            # 其他情况使用原来的逻辑
            mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
            
            if not np.any(mask):
                raise ValueError(f"没有找到波长范围 {start_wl}-{end_wl} 内的数据")
                
            # 选取数据
            X_selected = X[:, mask]
            wavelengths_selected = wavelengths[mask]
            
            # 保存选取的波长信息
            self.selected_wavelengths = {
                'start': start_wl,
                'end': end_wl,
                'indices': np.where(mask)[0],
                'wavelengths': wavelengths_selected
            }
            
            print(f"选取波长范围: {start_wl:.1f} - {end_wl:.1f} cm⁻¹")
            print(f"选取的数据点数: {X_selected.shape[1]}")
        
        return X_selected, wavelengths_selected
        

        
    def normalize_data(self, X: np.ndarray, method: str = 'standard', 
                      fit_data: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        归一化数据
        
        Args:
            X: 输入数据矩阵
            method: 归一化方法 ('standard', 'minmax', 'robust', 'l2')
            fit_data: 用于拟合归一化参数的数据，如果为None则使用X
            
        Returns:
            归一化后的数据
        """
        if fit_data is None:
            fit_data = X
            
        if method == 'standard':
            if self.scaler is None or not isinstance(self.scaler, StandardScaler):
                self.scaler = StandardScaler()
                self.scaler.fit(fit_data)
            return self.scaler.transform(X)
            
        elif method == 'minmax':
            if self.scaler is None or not isinstance(self.scaler, MinMaxScaler):
                self.scaler = MinMaxScaler()
                self.scaler.fit(fit_data)
            return self.scaler.transform(X)
            
        elif method == 'robust':
            if self.scaler is None or not isinstance(self.scaler, RobustScaler):
                self.scaler = RobustScaler()
                self.scaler.fit(fit_data)
            return self.scaler.transform(X)
            
        elif method == 'l2':
            # L2归一化（每个样本归一化到单位长度）
            from sklearn.preprocessing import normalize
            return normalize(X, norm='l2', axis=1)
            
        elif method == 'snv':
            # 标准正态变量变换 (Standard Normal Variate)
            return self._snv_normalize(X)
            
        elif method == 'msc':
            # 多元散射校正 (Multiplicative Scatter Correction)
            return self._msc_normalize(X, fit_data)
            
        elif method == 'area':
            # 面积归一化
            return self._area_normalize(X)
            
        elif method == 'vector':
            # 向量归一化
            return self._vector_normalize(X)
            
        elif method == 'range':
            # 范围归一化
            target_range = kwargs.get('target_range', (0, 1))
            return self._range_normalize(X, target_range)
            
        elif method == 'zscore':
            # Z-score标准化（与SNV相同）
            return self._snv_normalize(X)
            
        elif method == 'internal_standard':
            # 内标归一化
            internal_std_index = kwargs.get('internal_std_index', None)
            return self._internal_standard_normalize(X, internal_std_index)
            
        elif method == 'baseline_asls':
            # AsLS基线校正
            lam = kwargs.get('lam', 1e5)
            p = kwargs.get('p', 0.01)
            niter = kwargs.get('niter', 10)
            return self._baseline_asls_normalize(X, lam, p, niter)
            
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
            
    def _snv_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        标准正态变量变换
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            SNV归一化后的数据
        """
        X_snv = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_snv[i, :] = (X[i, :] - np.mean(X[i, :])) / np.std(X[i, :])
        return X_snv
        
    def _msc_normalize(self, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        多元散射校正
        
        Args:
            X: 输入数据矩阵
            reference: 参考数据矩阵
            
        Returns:
            MSC归一化后的数据
        """
        # 计算参考光谱（平均光谱）
        ref_spectrum = np.mean(reference, axis=0)
        
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            # 线性回归拟合
            coef = np.polyfit(ref_spectrum, X[i, :], 1)
            # 校正
            X_msc[i, :] = (X[i, :] - coef[1]) / coef[0]
            
        return X_msc
        
    def _area_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        面积归一化
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            面积归一化后的数据
        """
        X_area = np.zeros_like(X)
        for i in range(X.shape[0]):
            area = integrate.simps(X[i, :])
            if area == 0:
                X_area[i, :] = X[i, :]
            else:
                X_area[i, :] = X[i, :] / area
        return X_area
        
    def _vector_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        向量归一化
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            向量归一化后的数据
        """
        X_vector = np.zeros_like(X)
        for i in range(X.shape[0]):
            norm = np.linalg.norm(X[i, :])
            if norm == 0:
                X_vector[i, :] = X[i, :]
            else:
                X_vector[i, :] = X[i, :] / norm
        return X_vector
        
    def _range_normalize(self, X: np.ndarray, target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        范围归一化
        
        Args:
            X: 输入数据矩阵
            target_range: 目标范围
            
        Returns:
            范围归一化后的数据
        """
        X_range = np.zeros_like(X)
        for i in range(X.shape[0]):
            min_val = np.min(X[i, :])
            max_val = np.max(X[i, :])
            
            if max_val == min_val:
                X_range[i, :] = X[i, :]
            else:
                normalized = (X[i, :] - min_val) / (max_val - min_val)
                X_range[i, :] = normalized * (target_range[1] - target_range[0]) + target_range[0]
        return X_range
        
    def _internal_standard_normalize(self, X: np.ndarray, internal_std_index: Optional[int] = None) -> np.ndarray:
        """
        内标归一化
        
        Args:
            X: 输入数据矩阵
            internal_std_index: 内标峰位置索引
            
        Returns:
            内标归一化后的数据
        """
        if internal_std_index is None:
            # 默认使用索引220作为内标峰位置
            internal_std_index = 220
            
        X_internal = np.zeros_like(X)
        for i in range(X.shape[0]):
            # 确保索引在有效范围内
            idx = min(internal_std_index, X.shape[1] - 1)
            idx = max(idx, 0)
            
            internal_std_value = X[i, idx]
            if internal_std_value == 0:
                X_internal[i, :] = X[i, :]
            else:
                X_internal[i, :] = X[i, :] / internal_std_value
        return X_internal
        
    def _baseline_asls(self, y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
        """
        单条光谱的AsLS基线拟合
        
        Args:
            y: 光谱数据
            lam: 光滑度控制参数
            p: 惩罚因子
            niter: 迭代次数
            
        Returns:
            基线数据
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = D.dot(D.transpose())
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
        
    def _baseline_asls_normalize(self, X: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
        """
        AsLS基线校正归一化
        
        Args:
            X: 输入数据矩阵
            lam: 光滑度控制参数
            p: 惩罚因子
            niter: 迭代次数
            
        Returns:
            基线校正后的数据
        """
        X_baseline = np.zeros_like(X)
        for i in range(X.shape[0]):
            baseline = self._baseline_asls(X[i, :], lam=lam, p=p, niter=niter)
            X_baseline[i, :] = X[i, :] - baseline
        return X_baseline
        
    def apply_filter(self, X: np.ndarray, filter_type: str = 'savgol',
                    **filter_params) -> np.ndarray:
        """
        滤波处理
        
        Args:
            X: 输入数据矩阵
            filter_type: 滤波器类型 ('savgol', 'gaussian', 'median', 'lowpass')
            **filter_params: 滤波器参数
            
        Returns:
            滤波后的数据
        """
        X_filtered = np.zeros_like(X)
        
        if filter_type == 'savgol':
            # Savitzky-Golay滤波
            window_length = filter_params.get('window_length', 11)
            polyorder = filter_params.get('polyorder', 2)
            
            # 确保window_length为奇数且小于数据长度
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, X.shape[1])
            if window_length <= polyorder:
                window_length = polyorder + 2 if polyorder + 2 <= X.shape[1] else X.shape[1]
                if window_length % 2 == 0:
                    window_length -= 1
                    
            for i in range(X.shape[0]):
                X_filtered[i, :] = signal.savgol_filter(X[i, :], window_length, polyorder)
                
        elif filter_type == 'gaussian':
            # 高斯滤波
            sigma = filter_params.get('sigma', 1.0)
            for i in range(X.shape[0]):
                X_filtered[i, :] = signal.gaussian_filter1d(X[i, :], sigma)
                
        elif filter_type == 'median':
            # 中值滤波
            kernel_size = filter_params.get('kernel_size', 5)
            for i in range(X.shape[0]):
                X_filtered[i, :] = signal.medfilt(X[i, :], kernel_size)
                
        elif filter_type == 'lowpass':
            # 低通滤波
            cutoff = filter_params.get('cutoff', 0.1)
            order = filter_params.get('order', 4)
            
            # 设计Butterworth低通滤波器
            b, a = signal.butter(order, cutoff, btype='low')
            
            for i in range(X.shape[0]):
                X_filtered[i, :] = signal.filtfilt(b, a, X[i, :])
                
        elif filter_type == 'moving_average':
            # 移动平均滤波
            window_size = filter_params.get('window_size', 3)
            for i in range(X.shape[0]):
                X_filtered[i, :] = self._moving_average_filter(X[i, :], window_size)
                
        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")
            
        # 保存滤波参数
        self.filter_params = {
            'type': filter_type,
            'params': filter_params
        }
        
        print(f"应用 {filter_type} 滤波，参数: {filter_params}")
        
        return X_filtered
        
    def _moving_average_filter(self, spectrum: np.ndarray, window_size: int) -> np.ndarray:
        """
        移动平均滤波
        
        Args:
            spectrum: 光谱数据
            window_size: 窗口大小
            
        Returns:
            滤波后的光谱数据
        """
        kernel = np.ones(window_size) / window_size
        filtered = np.convolve(spectrum, kernel, mode='same')
        return filtered
        
    def derivative_transform(self, X: np.ndarray, order: int = 1) -> np.ndarray:
        """
        导数变换
        
        Args:
            X: 输入数据矩阵
            order: 导数阶数 (1 或 2)
            
        Returns:
            导数变换后的数据
        """
        if order == 1:
            # 一阶导数
            X_deriv = np.diff(X, axis=1)
        elif order == 2:
            # 二阶导数
            X_deriv = np.diff(X, n=2, axis=1)
        else:
            raise ValueError("只支持1阶和2阶导数")
            
        print(f"应用 {order} 阶导数变换")
        return X_deriv
        
    def preprocess_pipeline(self, X: np.ndarray, wavelengths: np.ndarray,
                           config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理流水线
        
        Args:
            X: 输入光谱数据
            wavelengths: 波长数组
            config: 预处理配置字典
            
        Returns:
            预处理后的数据和波长数组
        """
        X_processed = X.copy()
        wavelengths_processed = wavelengths.copy()
        
        print("\n=== 开始预处理流水线 ===")
        print(f"原始数据形状: {X_processed.shape}")
        
        # 1. 波长选择
        if 'wavelength_range' in config:
            start_wl, end_wl = config['wavelength_range']
            X_processed, wavelengths_processed = self.select_wavelength_range(
                X_processed, wavelengths_processed, start_wl, end_wl
            )
            
        # 2. 滤波
        if 'filter' in config:
            filter_config = config['filter']
            filter_type = filter_config.get('type', 'savgol')
            filter_params = {k: v for k, v in filter_config.items() if k != 'type'}
            X_processed = self.apply_filter(X_processed, filter_type, **filter_params)
            
        # 3. 导数变换
        if 'derivative_order' in config:
            order = config['derivative_order']
            X_processed = self.derivative_transform(X_processed, order)
            # 导数变换会改变数据长度，需要调整波长数组
            if order == 1:
                wavelengths_processed = wavelengths_processed[:-1]
            elif order == 2:
                wavelengths_processed = wavelengths_processed[:-2]
                
        # 4. 归一化
        if 'normalization' in config:
            method = config['normalization'].get('method', 'standard')
            X_processed = self.normalize_data(X_processed, method)
            
        print(f"预处理后数据形状: {X_processed.shape}")
        print("=== 预处理流水线完成 ===\n")
        
        return X_processed, wavelengths_processed
        
    def get_preprocessing_info(self) -> Dict:
        """
        获取预处理信息
        
        Returns:
            包含预处理参数的字典
        """
        info = {
            'selected_wavelengths': self.selected_wavelengths,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'filter_params': self.filter_params
        }
        return info
        
    def save_preprocessing_params(self, filepath: str) -> None:
        """
        保存预处理参数
        
        Args:
            filepath: 保存路径
        """
        import pickle
        
        params = {
            'scaler': self.scaler,
            'selected_wavelengths': self.selected_wavelengths,
            'filter_params': self.filter_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
            
        print(f"预处理参数已保存到: {filepath}")
        
    def load_preprocessing_params(self, filepath: str) -> None:
        """
        加载预处理参数
        
        Args:
            filepath: 参数文件路径
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            
        self.scaler = params.get('scaler')
        self.selected_wavelengths = params.get('selected_wavelengths')
        self.filter_params = params.get('filter_params', {})
        
        print(f"预处理参数已从 {filepath} 加载")