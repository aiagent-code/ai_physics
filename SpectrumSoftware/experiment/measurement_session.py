from typing import List, Dict, Optional, Callable
import numpy as np

class MeasurementSession:
    """测量会话管理器 - 负责单次测量的流程控制"""
    
    def __init__(self, dataset_manager, spectrometer_panel):
        self.dataset_manager = dataset_manager
        self.spectrometer_panel = spectrometer_panel
        self.current_variable_values = []
        self.variable_names = []
        
    def set_variables(self, variable_names: List[str]):
        """设置变量名称"""
        self.variable_names = variable_names
        self.dataset_manager.set_variables([f"var_{i}" for i in range(len(variable_names))], variable_names)
        
    def set_variable_values(self, values: List[float]):
        """设置当前变量值"""
        if not self.dataset_manager.validate_variable_values(values):
            raise ValueError("变量值验证失败")
        self.current_variable_values = values
        
    def measure_and_save(self,data_save_function, on_complete: Callable = None, apply_baseline=False) -> str:
        """
        测量并保存数据，统一调用DataSavePanel.save_current_spectrum，并保存元数据
        """
        if not self.variable_names:
            raise RuntimeError("请先设置测量变量")
        if not self.current_variable_values:
            raise RuntimeError("请先设置变量值")
        if not hasattr(self.spectrometer_panel, 'spectrometer') or not self.spectrometer_panel.spectrometer:
            raise RuntimeError("光谱仪未连接")
        spectrum = self.spectrometer_panel.spectrometer.get_formatted_spectrum()
        if spectrum is None or len(spectrum) == 0:
            raise RuntimeError("无法获取光谱数据")
        metadata = self._build_metadata()
        import os
        dataset_path = getattr(self.dataset_manager, 'dataset_path', './experiment_data')
        prefix = getattr(self.dataset_manager, 'file_prefix', 'spectral')
        # 获取当前文件数量作为序号
        file_count = len([f for f in os.listdir(dataset_path) if f.startswith(prefix)]) + 1
        filename = os.path.join(dataset_path, f"{prefix}_{file_count}.csv")
        saved_file = data_save_function(filename=filename, apply_baseline=apply_baseline)
        # 保存元数据
        self.dataset_manager.save_metadata_csv(self.current_variable_values, metadata)
        self.spectrometer_panel.current_spectrum = spectrum
        if on_complete:
            on_complete(saved_file)
        return saved_file
        
    def _build_metadata(self):
        """构建元数据"""
        metadata = {
            'integration_time_ms': float(self.spectrometer_panel.integration_time_var.get()),
            'scans_to_average': int(self.spectrometer_panel.scans_var.get()),
            'device_serial': self.spectrometer_panel.spectrometer.get_serial_number(),
            'device_model': self.spectrometer_panel.spectrometer.get_model()
        }
        return metadata
        
    def get_measurement_info(self) -> Dict:
        """获取测量信息"""
        return {
            'variable_names': self.variable_names,
            'current_values': self.current_variable_values,
            'dataset_info': self.dataset_manager.get_dataset_info()
        }
        
    def validate_measurement_ready(self) -> bool:
        """验证是否准备好进行测量"""
        return (
            bool(self.variable_names) and
            bool(self.current_variable_values) and
            hasattr(self.spectrometer_panel, 'spectrometer') and
            self.spectrometer_panel.spectrometer is not None
        )