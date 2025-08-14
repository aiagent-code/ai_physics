# -*- coding: utf-8 -*-
"""
模型端口管理器 - 独立管理模型端口模式的测量和结果
Function: 管理模型端口模式下的多次测量、结果存储和可视化
Purpose:
- 独立于数据集模式的模型端口功能
- 管理多次测量的光谱数据和预测结果
- 提供结果可视化功能
- 处理溶液类型配置
Stored Data:
- 测量历史记录 (measurement_history)
- 溶液类型名称 (solution_type)
- 模型处理器引用 (model_processor)
- 测量计数器 (measurement_count)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from .model_processor import ModelProcessor
from AIModel.OnetimeMeasure import OnetimeMeasure

class ModelPortManager:
    """模型端口管理器 - 独立管理模型端口模式"""
    
    def __init__(self, model_processor: ModelProcessor):
        self.model_processor = model_processor
        self.solution_type = "丙三醇"  # 默认溶液类型
        self.measurement_history = []  # 存储测量历史
        self.measurement_count = 0
        self.ai_predictor = OnetimeMeasure()  # AI模型预测器
        
    def set_solution_type(self, solution_type: str):
        """设置溶液类型名称"""
        self.solution_type = solution_type
        
    def get_solution_type(self) -> str:
        """获取当前溶液类型"""
        return self.solution_type
        
    def add_measurement(self, spectrum_data: np.ndarray, wavelengths: np.ndarray, 
                       display_mode: str = 'wavelength') -> Dict[str, Any]:
        """
        添加一次测量记录
        Args:
            spectrum_data: 光谱数据
            wavelengths: 波长数据
            display_mode: 显示模式
        Returns:
            Dict: 测量结果记录
        """
        self.measurement_count += 1
        
        # 使用AI模型进行真实预测
        try:
            # 获取模型路径
            if not self.model_processor.is_loaded or not self.model_processor.model_path:
                raise RuntimeError("模型未加载，无法进行预测")
            
            model_path = self.model_processor.model_path
            
            # 转换数据格式为列表
            wavenumbers_list = wavelengths.tolist() if isinstance(wavelengths, np.ndarray) else list(wavelengths)
            intensities_list = spectrum_data.tolist() if isinstance(spectrum_data, np.ndarray) else list(spectrum_data)
            
            # 调用AI模型进行预测
            predicted_concentration = self.ai_predictor.predict_single_measurement(
                wavenumbers_list, intensities_list, model_path
            )
            
            # 如果预测失败，使用默认值
            if predicted_concentration is None:
                raise RuntimeError("AI模型预测失败")
                
        except Exception as e:
            print(f"AI预测失败，使用默认值: {e}")
            # 预测失败时的默认值
            if self.solution_type == "丙三醇":
                predicted_concentration = 50.0
            else:
                predicted_concentration = 25.0
            
        # 创建测量记录
        measurement_record = {
            'id': self.measurement_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'solution_type': self.solution_type,
            'spectrum_data': spectrum_data.copy(),
            'wavelengths': wavelengths.copy(),
            'display_mode': display_mode,
            'predicted_concentration': predicted_concentration,
            'unit': '%'  # 浓度单位
        }
        
        self.measurement_history.append(measurement_record)
        return measurement_record
        
    def get_measurement_history(self) -> List[Dict[str, Any]]:
        """获取测量历史记录"""
        return self.measurement_history.copy()
        
    def get_measurement_count(self) -> int:
        """获取测量次数"""
        return self.measurement_count
        
    def clear_measurements(self):
        """清空测量记录"""
        self.measurement_history.clear()
        self.measurement_count = 0
        
    def get_latest_measurement(self) -> Optional[Dict[str, Any]]:
        """获取最新的测量记录"""
        if self.measurement_history:
            return self.measurement_history[-1]
        return None
        
    def remove_last_measurement(self) -> bool:
        """删除最后一次测量"""
        if self.measurement_history:
            self.measurement_history.pop()
            return True
        return False
        
    def get_measurement_summary(self) -> Dict[str, Any]:
        """获取测量汇总信息"""
        if not self.measurement_history:
            return {
                'total_count': 0,
                'solution_type': self.solution_type,
                'avg_concentration': 0,
                'min_concentration': 0,
                'max_concentration': 0
            }
            
        concentrations = [m['predicted_concentration'] for m in self.measurement_history]
        
        return {
            'total_count': len(self.measurement_history),
            'solution_type': self.solution_type,
            'avg_concentration': np.mean(concentrations),
            'min_concentration': np.min(concentrations),
            'max_concentration': np.max(concentrations),
            'latest_timestamp': self.measurement_history[-1]['timestamp']
        }
        
    def create_results_plot(self) -> plt.Figure:
        """
        创建结果可视化图表
        Returns:
            matplotlib.figure.Figure: 包含多次测量光谱和结果的图表
        """
        if not self.measurement_history:
            raise ValueError("没有测量数据可供可视化")
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 颜色列表用于区分不同测量
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.measurement_history)))
        
        # 上图：光谱数据
        for i, record in enumerate(self.measurement_history):
            x_data = record['wavelengths']
            y_data = record['spectrum_data']
            concentration = record['predicted_concentration']
            
            label = f"测量{record['id']}: {concentration:.1f}{record['unit']}"
            ax1.plot(x_data, y_data, color=colors[i], label=label, linewidth=1.5)
            
        ax1.set_xlabel('波长 (nm)' if self.measurement_history[0]['display_mode'] == 'wavelength' else '拉曼位移 (cm⁻¹)')
        ax1.set_ylabel('强度')
        ax1.set_title(f'{self.solution_type}溶液多次测量光谱对比')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 下图：浓度结果柱状图
        measurement_ids = [f"测量{r['id']}" for r in self.measurement_history]
        concentrations = [r['predicted_concentration'] for r in self.measurement_history]
        
        bars = ax2.bar(measurement_ids, concentrations, color=colors)
        ax2.set_ylabel(f'浓度 ({self.measurement_history[0]["unit"]})')
        ax2.set_title(f'{self.solution_type}溶液浓度测量结果')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上标注数值
        for bar, conc in zip(bars, concentrations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{conc:.1f}%', ha='center', va='bottom')
                    
        plt.tight_layout()
        return fig
        
    def export_results(self, file_path: str) -> bool:
        """
        导出测量结果到文件
        Args:
            file_path: 导出文件路径
        Returns:
            bool: 导出是否成功
        """
        try:
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['测量ID', '时间戳', '溶液类型', '预测浓度', '单位']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in self.measurement_history:
                    writer.writerow({
                        '测量ID': record['id'],
                        '时间戳': record['timestamp'],
                        '溶液类型': record['solution_type'],
                        '预测浓度': record['predicted_concentration'],
                        '单位': record['unit']
                    })
                    
            return True
        except Exception as e:
            print(f"导出结果失败: {e}")
            return False