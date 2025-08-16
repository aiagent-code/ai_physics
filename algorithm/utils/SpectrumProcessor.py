#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIBS光谱数据预处理模块

### 1. preprocessor.py - 数据读取与预处理
**主要功能：数据读取、格式转换、光谱预处理**

类功能说明：
1. PathManager: 路径管理类，管理项目目录结构（已移至SampleManager.py）
2. NameProcessor: 文件名处理类，支持多种命名格式的解析和标准化
3. DataProcessor: 数据处理类，负责CSV数据读取和基本处理（功能已整合到SpectrumProcessor中）
4. SpectrumProcessor: 光谱处理类，负责光谱数据的完整处理流程，生成x.csv和y.csv
5. SpectrumNormalizer: 光谱归一化类（已移至PreProcessor.py）
6. DataSplitter: 数据分割类（已移至SampleManager.py）
"""

import numpy as np
import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import sklearn
import re
import matplotlib.pyplot as plt

# PathManager类已移至SampleManager.py，请使用SampleManager中的路径管理功能

class NameProcessor:
    """
    文件名处理类
    功能：
    - 检测和解析多种文件命名格式
    - 生成标准化文件名
    - 支持逗号分隔格式（如：b1,c1,d3,1.txt）
    """
    
    def __init__(self, num_solutions=5):
        self.num_solutions = num_solutions
    
    def detect_naming_convention(self, filename):
        """检测文件命名约定"""
        base_name = os.path.splitext(filename)[0]
        
        # 检测逗号分隔格式：b1,c1,d3,1
        if ',' in base_name:
            return 'comma_separated'
        # 检测标准格式：字母_数字_字母_数字
        elif self._is_standard_format(base_name):
            return 'standard'
        # 检测旧约定：字母数字字母数字
        elif self._is_old_convention(base_name):
            return 'old'
        # 检测新约定：字母_数字_字母
        elif self._is_new_convention(base_name):
            return 'new'
        else:
            return 'unknown'
    
    def _is_standard_format(self, base_name):
        """检查是否为标准格式：字母_数字_字母_数字"""
        pattern = r'^[A-Za-z]+_\d+_[A-Za-z]+_\d+$'
        return bool(re.match(pattern, base_name))
    
    def _is_old_convention(self, base_name):
        """检查是否为旧约定：字母数字字母数字 或 字母数字（简化格式）"""
        # 完整格式：AB12CD34
        full_pattern = r'^[A-Za-z]+\d+[A-Za-z]+\d+$'
        # 简化格式：AD3
        simple_pattern = r'^[A-Za-z]+\d+$'
        return bool(re.match(full_pattern, base_name)) or bool(re.match(simple_pattern, base_name))
    
    def _is_new_convention(self, base_name):
        """检查是否为新约定：字母_数字_字母"""
        pattern = r'^[A-Za-z]+_\d+_[A-Za-z]+$'
        return bool(re.match(pattern, base_name))
    
    def parse_filename(self, filename):
        """解析文件名，返回结构化信息"""
        convention = self.detect_naming_convention(filename)
        
        if convention == 'comma_separated':
            return self._parse_comma_separated(filename)
        elif convention == 'standard':
            return self._parse_standard_format(filename)
        elif convention == 'old':
            return self._parse_old_convention(filename)
        elif convention == 'new':
            return self._parse_new_convention(filename)
        else:
            return {'convention': 'unknown', 'filename': filename}
    
    def _parse_comma_separated(self, filename):
        """解析逗号分隔格式文件名"""
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split(',')
        
        if len(parts) >= 2:
            # 处理两种情况：
            # 1. b1,c1,d3,1 - 多个元素+重复次数
            # 2. BE31,2 - 单个复合元素+重复次数
            elements = parts[:-1]  # 除最后一个数字外的所有部分
            replicate = int(parts[-1])  # 最后一个数字作为重复次数
            
            # 如果第一个元素包含多个字母和数字（如BE31），需要进一步解析
            if len(elements) == 1 and len(elements[0]) > 2:
                element = elements[0]
                # 解析BE31这样的格式
                parsed_elements = []
                i = 0
                while i < len(element):
                    if element[i].isalpha():
                        letter = element[i]
                        i += 1
                        # 提取数字
                        num_str = ''
                        while i < len(element) and element[i].isdigit():
                            num_str += element[i]
                            i += 1
                        if num_str:
                            parsed_elements.append(letter.lower() + num_str)
                        else:
                            parsed_elements.append(letter.lower() + '1')
                    else:
                        i += 1
                elements = parsed_elements
            
            return {
                'convention': 'comma_separated',
                'filename': filename,
                'elements': elements,
                'replicate': replicate
            }
        return {'convention': 'comma_separated', 'filename': filename, 'elements': [], 'replicate': 1}
    
    def _parse_old_convention(self, filename):
        """解析旧约定文件名"""
        base_name = os.path.splitext(filename)[0]
        # 先尝试匹配完整的旧约定格式：字母数字字母数字
        pattern = r'^([A-Za-z]+)(\d+)([A-Za-z]+)(\d+)$'
        match = re.match(pattern, base_name)
        
        if match:
            return {
                'convention': 'old',
                'filename': filename,
                'first_letter': match.group(1),
                'first_number': int(match.group(2)),
                'second_letter': match.group(3),
                'second_number': int(match.group(4))
            }
        
        # 尝试匹配简化格式：字母数字（如BE3）
        simple_pattern = r'^([A-Za-z]+)(\d+)$'
        simple_match = re.match(simple_pattern, base_name)
        
        if simple_match:
            letters = simple_match.group(1)
            number = int(simple_match.group(2))
            
            # 如果是多个字母，假设每个字母代表一个元素，数字是测量次数
            if len(letters) > 1:
                # 每个字母分配相等的比例（50%:50%对于两个字母）
                ratio_per_letter = 1  # 每个字母的比例权重相等
                return {
                    'convention': 'old',
                    'filename': filename,
                    'letters': letters,
                    'total_number': number,
                    'ratio_per_letter': ratio_per_letter
                }
            else:
                # 单个字母的情况
                return {
                    'convention': 'old',
                    'filename': filename,
                    'first_letter': letters,
                    'first_number': number,
                    'second_letter': '',
                    'second_number': 0
                }
        
        return {'convention': 'old', 'filename': filename}
    
    def _parse_new_convention(self, filename):
        """解析新约定文件名"""
        base_name = os.path.splitext(filename)[0]
        pattern = r'^([A-Za-z]+)_(\d+)_([A-Za-z]+)$'
        match = re.match(pattern, base_name)
        
        if match:
            return {
                'convention': 'new',
                'filename': filename,
                'first_letter': match.group(1),
                'number': int(match.group(2)),
                'second_letter': match.group(3)
            }
        return {'convention': 'new', 'filename': filename}
    
    def _parse_standard_format(self, filename):
        """解析标准格式文件名"""
        base_name = os.path.splitext(filename)[0]
        pattern = r'^([A-Za-z]+)_(\d+)_([A-Za-z]+)_(\d+)$'
        match = re.match(pattern, base_name)
        
        if match:
            return {
                'convention': 'standard',
                'filename': filename,
                'first_letter': match.group(1),
                'first_number': int(match.group(2)),
                'second_letter': match.group(3),
                'second_number': int(match.group(4))
            }
        return {'convention': 'standard', 'filename': filename}
    
    def generate_standard_filename(self, filename):
        """生成标准格式文件名 - 格式: a_b_c_d_e_f_g,replicate"""
        parsed = self.parse_filename(filename)
        
        # 初始化7个物质的浓度值
        concentrations = [0] * 7  # 对应a,b,c,d,e,f,g
        replicate = 1
        
        if parsed['convention'] == 'comma_separated':
            # 处理b1,c1,d3,1.txt或BE31,2.txt格式
            elements = parsed.get('elements', [])
            replicate = parsed.get('replicate', 1)
            
            # 解析元素和比例
            total_ratio = 0
            element_ratios = {}
            
            for element in elements:
                if len(element) >= 2 and element[0].isalpha():
                    letter = element[0].lower()
                    try:
                        ratio = int(element[1:])
                        element_ratios[letter] = ratio
                        total_ratio += ratio
                    except ValueError:
                        continue
            
            # 计算百分比浓度 - CE31,1表示C:E = 1:3的比例
            if total_ratio > 0:
                for letter, ratio in element_ratios.items():
                    if letter in 'abcdefg':
                        col_index = ord(letter) - ord('a')
                        concentrations[col_index] = int((ratio / total_ratio) * 100)
        
        elif parsed['convention'] == 'old' or parsed['convention'] == 'new':
            # 处理AD1.txt或AD13,1.txt或BE3.txt格式
            if parsed['convention'] == 'old':
                # 检查是否是简化格式（如BE3）
                if 'letters' in parsed:
                    # 处理BE3这样的格式
                    letters = parsed.get('letters', '').lower()
                    total_number = parsed.get('total_number', 1)
                    ratio_per_letter = parsed.get('ratio_per_letter', 1)
                    
                    # 为每个字母分配相等的比例
                    total_ratio = len(letters) * ratio_per_letter
                    if total_ratio > 0:
                        for letter in letters:
                            if letter in 'abcdefg':
                                col_index = ord(letter) - ord('a')
                                concentrations[col_index] = int((ratio_per_letter / total_ratio) * 100)
                elif 'second_letter' in parsed and 'second_number' in parsed:
                    # 处理标准的旧约定格式 (如 CE31)
                    first_letter = parsed.get('first_letter', '').lower()
                    first_number = parsed.get('first_number', 1)
                    second_letter = parsed.get('second_letter', '').lower()
                    second_number = parsed.get('second_number', 0)
                    total_ratio = first_number + second_number
                    
                    # 计算百分比浓度
                    if total_ratio > 0:
                        if first_letter in 'abcdefg':
                            col_index = ord(first_letter) - ord('a')
                            concentrations[col_index] = int((first_number / total_ratio) * 100)
                        if second_letter and second_letter in 'abcdefg':
                            col_index = ord(second_letter) - ord('a')
                            concentrations[col_index] = int((second_number / total_ratio) * 100)
                else:
                    # 处理简化的旧约定格式 (如 AD3)
                    first_letter = parsed.get('first_letter', '').lower()
                    first_number = parsed.get('first_number', 1)
                    
                    # 对于简化格式，假设浓度为100%
                    if first_letter in 'abcdefg':
                        col_index = ord(first_letter) - ord('a')
                        concentrations[col_index] = 100
            else:  # new convention
                first_letter = parsed.get('first_letter', '').lower()
                number = parsed.get('number', 1)
                second_letter = parsed.get('second_letter', '').lower()
                first_number = 1
                second_number = number
                total_ratio = first_number + second_number
                
                # 计算百分比浓度
                if total_ratio > 0:
                    if first_letter in 'abcdefg':
                        col_index = ord(first_letter) - ord('a')
                        concentrations[col_index] = int((first_number / total_ratio) * 100)
                    if second_letter and second_letter in 'abcdefg':
                        col_index = ord(second_letter) - ord('a')
                        concentrations[col_index] = int((second_number / total_ratio) * 100)
        
        # 生成标准格式文件名
        conc_str = '_'.join(map(str, concentrations))
        return f"{conc_str},{replicate}"
    
    def _generate_label_from_elements(self, elements):
        """从元素列表生成标签"""
        # 这里可以根据具体业务逻辑来实现
        # 暂时返回元素数量作为简单标签
        return float(len(elements))

# DataProcessor类功能已整合到SpectrumProcessor中，无需单独使用

class SpectrumProcessor:
    """
    光谱处理器：负责完整的光谱数据处理流程
    
    功能：
    - 文件格式转换
    - XY文件生成
    - 数据过滤
    """
    
    def __init__(self, source_dir, target_dir, data_dir, num_solutions=7, selected_substance='c'):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.data_dir = data_dir
        self.num_solutions = num_solutions
        self.selected_substance = selected_substance
        self.name_processor = NameProcessor(num_solutions)
        
        # 物质映射字典
        self.substance_mapping = {
            "a": "异丙醇",
            "b": "丙三醇", 
            "c": "乙二醇",
            "d": "聚乙二醇",
            "e": "乙酸",
            "f": "二甲基乙枫",
            "g": "三乙醇胺"
        }
        
        # 保存波数信息作为类属性
        self.wavenumber = None
        
        print(f"当前选择的物质: {self.substance_mapping.get(selected_substance, '未知物质')}")
    
    def convert_to_standard_format(self):
        """转换为标准格式"""
        if not os.path.exists(self.source_dir):
            print(f"源目录不存在: {self.source_dir}")
            return False
        
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir, exist_ok=True)
        
        converted_count = 0
        for filename in os.listdir(self.source_dir):
            if filename.endswith('.txt') or filename.endswith('.csv'):
                # 跳过已经是标准格式的文件和配置文件
                if filename in ['x.csv', 'y.csv', 'y_selected.csv'] or filename.lower() in ['config.csv', 'readme.txt', 'info.txt']:
                    continue
                    
                source_path = os.path.join(self.source_dir, filename)
                standard_name = self.name_processor.generate_standard_filename(filename)
                target_path = os.path.join(self.target_dir, f"{standard_name}.csv")
                
                # 如果目标文件已存在且与源文件相同，跳过
                if os.path.exists(target_path) and os.path.samefile(source_path, target_path):
                    continue
                
                try:
                    # 复制并重命名文件
                    self._copy_and_rename_file(source_path, target_path)
                    converted_count += 1
                    print(f"已转换: {filename} -> {standard_name}.csv")
                except Exception as e:
                    print(f"转换失败 {filename}: {e}")
        
        print(f"转换完成，共处理 {converted_count} 个文件")
        return True
    
    def _copy_and_rename_file(self, source_path, target_path):
        """复制并重命名文件"""
        import shutil
        shutil.copy2(source_path, target_path)
    
    def generate_xy_files(self):
        """生成X和Y文件"""
        try:
            # 查找所有CSV文件，排除已生成的x.csv和y相关文件
            csv_files = [f for f in os.listdir(self.target_dir) 
                        if f.endswith('.csv') and not f.startswith('x') and not f.startswith('y')]
            if not csv_files:
                print("警告：目标目录中没有找到CSV文件")
                return None, None
            
            all_spectra = []
            all_concentrations = []  # 存储7列浓度数据
            sample_ids = []  # 存储样本ID
            wavenumber_saved = False  # 标记是否已保存波数信息
            
            # 处理每个CSV文件
            for csv_file in csv_files:
                csv_path = os.path.join(self.target_dir, csv_file)
                
                try:
                    # 读取光谱数据
                    data = pd.read_csv(csv_path, header=None, sep='\t')
                    
                    # 跳过第一行（标题行）
                    if data.shape[0] > 1:
                        data = data.iloc[1:]
                    
                    # 检查数据格式：应该有两列（拉曼位移和强度）
                    if data.shape[1] >= 2:
                        # 第一列是波数信息，第二列是光谱强度数据
                        wavenumber = data.iloc[:, 0].values
                        spectrum = data.iloc[:, 1].values
                        
                        # 过滤掉NaN值并转换为数值
                        wavenumber = pd.to_numeric(wavenumber, errors='coerce')
                        spectrum = pd.to_numeric(spectrum, errors='coerce')
                        
                        # 找到有效数据的索引
                        valid_indices = ~(pd.isna(wavenumber) | pd.isna(spectrum))
                        wavenumber = wavenumber[valid_indices]
                        spectrum = spectrum[valid_indices]
                        
                        if len(spectrum) > 0:
                            all_spectra.append(spectrum)
                            
                            # 保存波数信息（只需要保存一次，因为所有文件的波数应该相同）
                            if not wavenumber_saved:
                                self.wavenumber = wavenumber
                                wavenumber_saved = True
                                print(f"波数信息已保存，范围: {wavenumber[0]:.2f} - {wavenumber[-1]:.2f} cm^-1，数据点数: {len(wavenumber)}")
                            
                            # 从文件名解析浓度信息
                            parsed = self.name_processor.parse_filename(csv_file)
                            concentrations = self._extract_concentrations_from_filename(csv_file)
                            all_concentrations.append(concentrations)
                            
                            # 生成样本ID（基于文件名的前缀部分）
                            sample_id = self._generate_sample_id(csv_file)
                            sample_ids.append(sample_id)
                        else:
                            print(f"文件 {csv_file} 中没有有效的光谱数据")
                    else:
                        print(f"文件 {csv_file} 格式不正确，列数: {data.shape[1]}")
                            
                except Exception as e:
                    print(f"处理文件 {csv_file} 时出错: {e}")
                    continue
            
            if not all_spectra:
                print("错误：没有成功读取任何光谱数据")
                return None, None
            
            # 检查所有光谱数据的长度是否一致
            spectrum_lengths = [len(spectrum) for spectrum in all_spectra]
            if len(set(spectrum_lengths)) > 1:
                print(f"警告：光谱数据长度不一致: {set(spectrum_lengths)}")
                # 使用最小长度进行截断
                min_length = min(spectrum_lengths)
                all_spectra = [spectrum[:min_length] for spectrum in all_spectra]
                print(f"已将所有光谱数据截断到长度: {min_length}")
            
            # 生成X文件（光谱数据矩阵）
            x_data = np.array(all_spectra)
            x_path = os.path.join(self.target_dir, "x.csv")
            
            # 使用波数作为列名
            if self.wavenumber is not None and len(self.wavenumber) == x_data.shape[1]:
                # 将波数转换为字符串作为列名
                column_names = [str(round(w, 2)) for w in self.wavenumber]
                x_df = pd.DataFrame(x_data, columns=column_names)
                x_df.to_csv(x_path, index=False)
                print(f"X文件已生成: {x_path}，形状: {x_data.shape}，波数范围: {self.wavenumber[0]:.2f} - {self.wavenumber[-1]:.2f} cm^-1")
            else:
                # 如果没有波数信息，使用默认方式
                pd.DataFrame(x_data).to_csv(x_path, header=False, index=False)
                print(f"X文件已生成: {x_path}，形状: {x_data.shape}（警告：没有波数信息）")
            
            # 生成Y文件（7列浓度数据 + 样本标签列）
            y_data_with_labels = []
            for i, (conc, sample_id) in enumerate(zip(all_concentrations, sample_ids)):
                # 将7列浓度数据和样本标签组合
                row = list(conc) + [sample_id]
                y_data_with_labels.append(row)
            
            y_data = np.array(y_data_with_labels, dtype=object)
            y_path = os.path.join(self.target_dir, "y.csv")
            pd.DataFrame(y_data).to_csv(y_path, header=False, index=False)
            print(f"Y文件已生成: {y_path}，形状: {y_data.shape}（包含7列浓度数据 + 1列样本标签）")
            
            # 生成Y_selected文件（选定物质浓度 + 样本标签）
            # 安全检查：确保selected_substance不为空且为有效字符
            if not self.selected_substance or len(self.selected_substance) == 0:
                print("警告：selected_substance为空，使用默认值'c'")
                self.selected_substance = 'c'
            
            selected_col_index = ord(self.selected_substance.lower()) - ord('a')  # 将a,b,c...转换为0,1,2...
            if selected_col_index < len(all_concentrations[0]):
                selected_concentrations = [conc[selected_col_index] for conc in all_concentrations]
                y_selected_data = np.column_stack([selected_concentrations, sample_ids])
                y_selected_path = os.path.join(self.target_dir, "y_selected.csv")
                pd.DataFrame(y_selected_data).to_csv(y_selected_path, header=False, index=False)
                print(f"Y_selected文件已生成: {y_selected_path}，形状: {y_selected_data.shape}")
                print(f"选择的物质: {self.substance_mapping[self.selected_substance]}")
                
                return x_path, y_selected_path
            else:
                print(f"错误：选择的物质索引 {selected_col_index} 超出范围")
                return x_path, y_path
            
        except Exception as e:
            print(f"生成XY文件失败: {e}")
            return None, None
    
    def _extract_concentrations_from_filename(self, filename):
        """从标准格式文件名中提取7列浓度信息"""
        try:
            # 移除文件扩展名
            name_part = filename.replace('.txt.csv', '').replace('.csv', '')
            
            # 初始化7列浓度数据 (对应a,b,c,d,e,f,g)
            concentrations = [0.0] * 7
            
            # 检查是否为标准格式（a_b_c_d_e_f_g,replicate）
            if ',' in name_part:
                parts = name_part.split(',')
                if len(parts) >= 1:
                    conc_part = parts[0]
                    if '_' in conc_part:
                        conc_values = conc_part.split('_')
                        # 直接使用7个浓度值
                        for i, val in enumerate(conc_values[:7]):
                            try:
                                concentrations[i] = float(val)
                            except ValueError:
                                concentrations[i] = 0.0
                        return concentrations
            
            # 如果不是标准格式，尝试从原始文件名解析
            # 使用NameProcessor重新解析
            parsed = self.name_processor.parse_filename(filename)
            
            if parsed['convention'] == 'comma_separated':
                elements = parsed.get('elements', [])
                # 解析元素和比例
                total_ratio = 0
                element_ratios = {}
                
                for element in elements:
                    if len(element) >= 2 and element[0].isalpha():
                        letter = element[0].lower()
                        try:
                            ratio = int(element[1:])
                            element_ratios[letter] = ratio
                            total_ratio += ratio
                        except ValueError:
                            continue
                
                # 计算百分比浓度 - CE31,1表示C:E = 1:3的比例
                if total_ratio > 0:
                    for letter, ratio in element_ratios.items():
                        if letter in 'abcdefg':
                            col_index = ord(letter) - ord('a')
                            concentrations[col_index] = float((ratio / total_ratio) * 100)
            
            elif parsed['convention'] == 'old' or parsed['convention'] == 'new':
                # 处理AD1.txt或AD13,1.txt格式
                if parsed['convention'] == 'old':
                    # 检查是否为多字母格式（如BC1, BD2等）
                    if 'letters' in parsed:
                        letters = parsed.get('letters', '').lower()
                        ratio_per_letter = parsed.get('ratio_per_letter', 1)
                        
                        # 每个字母分配相等的比例
                        if letters and len(letters) > 0:
                            percentage_per_letter = 100.0 / len(letters)
                            for letter in letters:
                                if letter in 'abcdefg':
                                    col_index = ord(letter) - ord('a')
                                    concentrations[col_index] = percentage_per_letter
                    else:
                        # 处理单字母或双字母格式
                        first_letter = parsed.get('first_letter', '').lower()
                        first_number = parsed.get('first_number', 1)
                        second_letter = parsed.get('second_letter', '').lower()
                        second_number = parsed.get('second_number', 0)
                        
                        # 检查是否为简化格式（只有一个字母和数字）
                        if second_letter == '':
                            # 简化格式，假设浓度为100%
                            if first_letter in 'abcdefg':
                                col_index = ord(first_letter) - ord('a')
                                concentrations[col_index] = 100.0
                        else:
                            # 标准格式，计算比例
                            total_ratio = first_number + second_number
                            if total_ratio > 0:
                                if first_letter in 'abcdefg':
                                    col_index = ord(first_letter) - ord('a')
                                    concentrations[col_index] = float((first_number / total_ratio) * 100)
                                if second_letter in 'abcdefg':
                                    col_index = ord(second_letter) - ord('a')
                                    concentrations[col_index] = float((second_number / total_ratio) * 100)
                else:  # new convention
                    first_letter = parsed.get('first_letter', '').lower()
                    number = parsed.get('number', 1)
                    second_letter = parsed.get('second_letter', '').lower()
                    first_number = 1
                    second_number = number
                    total_ratio = first_number + second_number
                    
                    # 计算百分比浓度
                    if total_ratio > 0:
                        if first_letter in 'abcdefg':
                            col_index = ord(first_letter) - ord('a')
                            concentrations[col_index] = float((first_number / total_ratio) * 100)
                        if second_letter and second_letter in 'abcdefg':
                            col_index = ord(second_letter) - ord('a')
                            concentrations[col_index] = float((second_number / total_ratio) * 100)
            
            return concentrations
            
        except Exception as e:
            print(f"解析文件名 {filename} 的浓度信息时出错: {e}")
            return [0.0] * 7
    
    def _generate_sample_id(self, filename):
        """从文件名生成样本ID"""
        try:
            # 移除文件扩展名
            name_part = filename.replace('.txt.csv', '').replace('.csv', '')
            
            # 如果是标准格式（a_b_c_d_e_f_g,replicate），提取浓度组合作为样本基础ID
            if ',' in name_part:
                parts = name_part.split(',')
                conc_part = parts[0]  # 浓度部分
                # 生成样本ID格式：sample_浓度组合（不包含重复次数，相同浓度的样本应该有相同ID）
                return f"sample_{conc_part}"
            else:
                # 对于非标准格式，使用原文件名生成样本ID
                return f"sample_{name_part}"
                
        except Exception as e:
            print(f"生成文件 {filename} 的样本ID时出错: {e}")
            return f"sample_{filename}"
    
    def full_processing_pipeline(self, plot_config=None):
        """完整处理流程"""
        print("开始完整处理流程...")
        
        try:
            # 1. 转换为标准格式
            if self.convert_to_standard_format():
                print("格式转换完成")
            
            # 2. 生成XY文件
            x_path, y_path = self.generate_xy_files()
            if x_path and y_path:
                print("XY文件生成完成")
                print(f"X文件路径: {x_path}")
                print(f"Y文件路径: {y_path}")
                return x_path, y_path
            else:
                print("XY文件生成失败")
                return None, None
                
        except Exception as e:
            print(f"完整处理流程失败: {e}")
            return None, None
    
    def filter_spectrum_data(self, input_csv_path, output_csv_path=None, window_size=3):
        """过滤光谱数据"""
        try:
            data = pd.read_csv(input_csv_path, header=None)
            filtered_data = np.zeros_like(data.values)
            
            for i in range(data.shape[0]):
                filtered_data[i, :] = self._moving_average_filter(data.iloc[i, :].values, window_size)
            
            if output_csv_path is None:
                output_csv_path = input_csv_path.replace('.csv', '_filtered.csv')
            
            filtered_df = pd.DataFrame(filtered_data)
            filtered_df.to_csv(output_csv_path, index=False, header=False)
            
            print(f"光谱过滤完成，输出文件: {output_csv_path}")
            return output_csv_path
            
        except Exception as e:
            print(f"光谱过滤失败: {e}")
            return None
    
    def _moving_average_filter(self, spectrum, window_size):
        """移动平均滤波"""
        kernel = np.ones(window_size) / window_size
        filtered = np.convolve(spectrum, kernel, mode='same')
        return filtered

# DataSplitter类功能已移至SampleManager.py，请使用SampleManager进行样本级别的数据分割
# 这样可以确保同一样本的不同测量点不会被分离到不同的数据集中

# 注意：便捷函数已移除，请直接使用对应的类方法
# 例如：使用 SpectrumNormalizer().normalize_spectrum() 而不是 normalize_spectrum_simple()
# 这样可以保持代码的一致性和可维护性