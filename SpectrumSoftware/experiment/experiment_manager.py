import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class ExperimentManager:
    """实验管理器 - 负责实验状态管理和配置管理"""
    
    def __init__(self):
        self.experiment_config = {}
        self.is_running = False
        self.current_step = 0
        self.measurement_count = 0
        self.experiment_start_time = None
        self.experiment_end_time = None
        
    def load_config(self, config_file: str = "experiment_config.json"):
        """加载实验配置"""
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.experiment_config = json.load(f)
        else:
            self.create_default_config(config_file)
            
    def create_default_config(self, config_file: str = "experiment_config.json"):
        """创建默认配置"""
        self.experiment_config = {
            'dataset_path': './experiment_data',
            'last_updated': datetime.now().isoformat()
        }
        self.save_config(config_file)
        
    def save_config(self, config_file: str = "experiment_config.json"):
        """保存实验配置"""
        self.experiment_config['last_updated'] = datetime.now().isoformat()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_config, f, indent=2, ensure_ascii=False)
            
    def start_experiment(self):
        """开始实验"""
        if self.is_running:
            raise RuntimeError("实验已在运行中")
            
        self.is_running = True
        self.current_step = 1
        self.measurement_count = 0
        self.experiment_start_time = datetime.now()
        self.experiment_end_time = None
        
    def stop_experiment(self):
        """停止实验"""
        if not self.is_running:
            raise RuntimeError("实验未在运行")
            
        self.is_running = False
        self.current_step = 0
        self.experiment_end_time = datetime.now()
        
    def increment_measurement(self):
        """增加测量计数"""
        if not self.is_running:
            raise RuntimeError("实验未在运行")
        self.measurement_count += 1
        
    def get_experiment_status(self) -> Dict:
        """获取实验状态"""
        return {
            'is_running': self.is_running,
            'current_step': self.current_step,
            'measurement_count': self.measurement_count,
            'start_time': self.experiment_start_time.isoformat() if self.experiment_start_time else None,
            'end_time': self.experiment_end_time.isoformat() if self.experiment_end_time else None,
            'duration': self._calculate_duration()
        }
        
    def _calculate_duration(self) -> Optional[str]:
        """计算实验持续时间"""
        if not self.experiment_start_time:
            return None
            
        end_time = self.experiment_end_time or datetime.now()
        duration = end_time - self.experiment_start_time
        return str(duration)
        
    def reset_experiment(self):
        """重置实验状态"""
        self.is_running = False
        self.current_step = 0
        self.measurement_count = 0
        self.experiment_start_time = None
        self.experiment_end_time = None 