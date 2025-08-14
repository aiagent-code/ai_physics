import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional

class DatasetManager:
    """数据集管理器 - 负责数据集配置、文件管理和数据保存"""
    
    def __init__(self, dataset_path: str = "./experiment_data"):
        self.dataset_path = dataset_path
        self.variables = []
        self.variable_names = []
        self.measurement_count = 0
        self.dataset_created_time = None
        self.file_prefix = "spectral"  # 默认文件前缀
        self.ensure_dataset_path()
        
    def ensure_dataset_path(self):
        """确保数据集路径存在"""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            self.dataset_created_time = datetime.now()
            
    def set_variables(self, variables: List[str], variable_names: List[str]):
        """设置变量和变量名"""
        if len(variables) != len(variable_names):
            raise ValueError("变量列表和变量名列表长度必须相同")
            
        self.variables = variables
        self.variable_names = variable_names
        self.measurement_count = 0
        
        # 创建数据集信息文件
        self._save_dataset_info()
        
    def set_file_prefix(self, prefix: str):
        """设置文件前缀"""
        self.file_prefix = prefix.strip() if prefix.strip() else "spectral"
        
    def _save_dataset_info(self):
        """保存数据集信息"""
        info = {
            'created_time': datetime.now().isoformat(),
            'variables': self.variables,
            'variable_names': self.variable_names,
            'dataset_path': self.dataset_path,
            'file_prefix': self.file_prefix,
            'total_measurements': 0
        }
        
        info_file = os.path.join(self.dataset_path, 'dataset_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
            
    def generate_filename(self, variable_values: List[float]) -> str:
        """生成文件名 - 使用简化的命名方式"""
        if len(variable_values) != len(self.variable_names):
            raise ValueError("变量值数量与变量名数量不匹配")
            
        self.measurement_count += 1
        filename = f"{self.file_prefix}_{self.measurement_count}.csv"
        return os.path.join(self.dataset_path, filename)
        
    def _update_dataset_info(self):
        """更新数据集信息"""
        info_file = os.path.join(self.dataset_path, 'dataset_info.json')
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
        else:
            info = {}
            
        info['total_measurements'] = self.measurement_count
        info['last_updated'] = datetime.now().isoformat()
        info['file_prefix'] = self.file_prefix
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
    def delete_last_measurement(self) -> bool:
        """删除最后一个测量文件和metadata.csv最后一行"""
        if self.measurement_count <= 0:
            return False
            
        try:
            # 删除光谱数据文件
            spectrum_file = os.path.join(self.dataset_path, f"{self.file_prefix}_{self.measurement_count}.csv")
            if os.path.exists(spectrum_file):
                os.remove(spectrum_file)
                
            # 删除metadata.csv最后一行
            meta_file = os.path.join(self.dataset_path, 'metadata.csv')
            if os.path.exists(meta_file):
                self._remove_last_line_from_csv(meta_file)
                
            # 减少计数
            self.measurement_count -= 1
            
            # 更新数据集信息
            self._update_dataset_info()
            
            return True
            
        except Exception as e:
            print(f"删除最后一个测量文件失败: {e}")
            return False
            
    def _remove_last_line_from_csv(self, filepath: str):
        """从CSV文件中删除最后一行"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > 1:  # 保留表头
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines[:-1])
        except Exception as e:
            print(f"删除CSV最后一行失败: {e}")
        
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv') and not f == 'metadata.csv']
        return {
            'dataset_path': self.dataset_path,
            'total_measurements': len(files),
            'current_measurement_count': self.measurement_count,
            'variables': self.variables,
            'variable_names': self.variable_names,
            'file_prefix': self.file_prefix,
            'created_time': self.dataset_created_time.isoformat() if self.dataset_created_time else None
        }
        
    def validate_variable_values(self, values: List[float]) -> bool:
        """验证变量值"""
        if len(values) != len(self.variable_names):
            return False
        # 检查是否为有效数值
        for value in values:
            if not isinstance(value, (int, float)) or str(value).lower() in ['nan', 'inf', '-inf']:
                return False
        return True 

    def save_metadata_csv(self, variable_values: List[float], metadata: Dict = None):
        """保存测量环境信息到metadata.csv"""
        import csv
        import os
        from datetime import datetime
        
        # 增加测量计数
        self.measurement_count += 1
        
        meta_file = os.path.join(self.dataset_path, 'metadata.csv')
        file_exists = os.path.exists(meta_file)
        with open(meta_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            if not file_exists:
                header = ['测量序号', '时间戳'] + self.variable_names + ['积分时间(ms)', '扫描次数', '平滑宽度']
                writer.writerow(header)
            # 写入数据行
            row = [self.measurement_count, datetime.now().isoformat()] + variable_values
            if metadata:
                row.extend([
                    metadata.get('integration_time_ms', ''),
                    metadata.get('scans_to_average', '')
                ])
            else:
                row.extend(['', '', ''])
            writer.writerow(row)
        
        # 更新数据集信息
        self._update_dataset_info()