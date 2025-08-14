# -*- coding: utf-8 -*-
"""
光谱数据处理器 - 提供光谱数据的处理和分析功能
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from scipy import signal
from scipy.interpolate import interp1d
import json
import csv
from datetime import datetime
import os

class SpectrumProcessor:
    """光谱数据处理器 - 纯工具函数类，无状态"""
    
    @staticmethod
    def dark_correction(spectrum: np.ndarray, dark_spectrum: np.ndarray) -> np.ndarray:
        """暗光谱校正"""
        if dark_spectrum is not None and len(spectrum) == len(dark_spectrum):
            return spectrum - dark_spectrum
        return spectrum
        
    @staticmethod
    def wavelength_to_raman_shift(wavelengths: np.ndarray, excitation_wavelength_nm: float) -> np.ndarray:
        """
        将波长转换为拉曼位移（cm^-1）
        wavelengths: 波长数组（单位nm）
        excitation_wavelength_nm: 激发光波长（单位nm）
        """
        if wavelengths is None:
            raise ValueError("波长数据未提供")
        wl = np.array(wavelengths)
        wl0 = excitation_wavelength_nm
        raman_shift = 1e7 / wl0 - 1e7 / wl
        return raman_shift

    @staticmethod
    def baseline_remove_asls(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asymmetric Least Squares Smoothing (asls) 基线去除
        y: 输入光谱
        lam: 平滑参数
        p: 惩罚参数
        niter: 迭代次数
        return: 去基线后的光谱, 基线
        """
        L = len(y)
        # 创建二阶差分矩阵
        D = np.zeros((L-2, L))
        for i in range(L-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
        
        w = np.ones(L)
        for i in range(niter):
            W = np.diag(w)
            Z = W + lam * D.T @ D
            baseline = np.linalg.solve(Z, w * y)
            w = p * (y > baseline) + (1 - p) * (y < baseline)
        return y - baseline, baseline