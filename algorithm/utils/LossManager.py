#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### 3. LossManager.py - 损失函数管理器
**主要功能：统一的损失函数定义和管理**

类功能说明：
- LossManager: 损失函数管理类
  - 提供统一的损失函数接口
  - 支持numpy和TensorFlow两种实现
  - 包含常用的回归损失函数
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import make_scorer

class LossManager:
    """
    损失函数管理类
    
    功能：
    - 统一的损失函数定义
    - 支持numpy和TensorFlow实现
    - 提供sklearn兼容的scorer
    """
    
    @staticmethod
    def huber_loss(y_true, y_pred, delta=3):
        """
        Huber Loss的numpy实现，用于评估
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        delta: Huber Loss的阈值参数，默认为1.0
        
        返回:
        loss: Huber损失值
        """
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic**2 + delta * linear)
    
    @staticmethod
    def piecewise_loss(y_true, y_pred, threshold=1.0):
        """
        分段损失函数的numpy实现
        当|预测值-真实值|小于threshold时，该数据点的损失为0；否则为差值的绝对值
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        threshold: 阈值参数，默认为1.0
        
        返回:
        loss: 分段损失值的平均值
        """
        error = np.abs(y_pred - y_true)
        # 当误差小于阈值时，损失为0；否则为误差的绝对值
        loss = np.where(error < threshold, 0.0, error)
        return np.mean(loss)
    
    @staticmethod
    def piecewise_loss_tf(y_true, y_pred, threshold=1.0):
        """
        分段损失函数的TensorFlow实现，用于模型训练
        当|预测值-真实值|小于threshold时，该数据点的损失为0；否则为差值的绝对值
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        threshold: 阈值参数，默认为1.0
        
        返回:
        loss: 分段损失值的平均值
        """
        error = tf.abs(y_pred - y_true)
        # 当误差小于阈值时，损失为0；否则为误差的绝对值
        loss = tf.where(error < threshold, 0.0, error)
        return tf.reduce_mean(loss)
    
    @staticmethod
    def huber_loss_tf(y_true, y_pred, delta=3.0):
        """
        Huber Loss的TensorFlow实现，用于模型训练
        
        参数:
        y_true: 真实值张量
        y_pred: 预测值张量
        delta: Huber Loss的阈值参数，默认为3.0
        
        返回:
        loss: Huber损失张量
        """
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)
    
    @staticmethod
    def mse_loss(y_true, y_pred):
        """
        均方误差损失
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        
        返回:
        loss: MSE损失值
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae_loss(y_true, y_pred):
        """
        平均绝对误差损失
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        
        返回:
        loss: MAE损失值
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @classmethod
    def get_huber_scorer(cls, delta=1.0):
        """
        获取sklearn兼容的Huber损失scorer
        
        参数:
        delta: Huber Loss的阈值参数
        
        返回:
        scorer: sklearn兼容的scorer对象
        """
        def huber_scorer(y_true, y_pred):
            return cls.huber_loss(y_true, y_pred, delta)
        
        return make_scorer(huber_scorer, greater_is_better=False)
    
    @staticmethod
    def fit_line(x, y):
        """
        拟合直线，返回斜率和截距
        
        参数:
        x: 自变量
        y: 因变量
        
        返回:
        k: 斜率
        b: 截距
        """
        k = np.cov(x, y)[0, 1] / np.var(x)
        b = np.mean(y) - k * np.mean(x)
        return k, b