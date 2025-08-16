#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### precenter.py - 处理中心具体实现
**主要功能：实现ProcessCenter的具体处理逻辑**

类功能说明：
- PreCenter: 处理中心的具体实现，包含所有步骤的详细逻辑

流程概述（九步）:
1. 读取文件内容，生成整合数据文件
2. 数据预处理（选取特定波段，进行归一化等）
3. 数据集分离
4. 训练前总画图
5. PLS回归分析
6. 超参数优化（贝叶斯优化）
7. 神经网络基于测量数据点训练
8. 模型评估基于样本点预测
9. 新样本预测
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class PreCenter:
    """
    LIBS光谱数据处理中心具体实现类
    
    功能说明:
    - 实现所有数据处理步骤的具体逻辑
    - 管理数据缓存和工具类实例
    - 提供完整的九步处理流程
    
    属性:
    - tools: 工具类字典，包含各种专业处理工具
    - data_cache: 数据缓存字典，存储各步骤的处理结果
    - selected_substance: 选择的物质字母
    - substance_name: 物质名称
    - target_element_index: 目标元素索引
    """
    
    def __init__(self, selected_substance='b', loss_function='huber_loss_tf'):
        """初始化处理中心实现
        
        参数:
        selected_substance: 选择的物质字母 (a-g)，默认为'b'(丙三醇)
        loss_function: 损失函数类型，可选：'mean_squared_error', 'huber_loss_tf', 'weighted_huber_loss_tf'
        """
        self.tools = None
        self.data_cache = {}
        self.selected_substance = selected_substance
        self.loss_function = loss_function
        
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
        self.substance_name = self.substance_mapping.get(self.selected_substance, '未知物质')
        self.target_element_index = ord(self.selected_substance.lower()) - ord('a')
    
    def setup_environment(self):
        """环境配置和包导入"""
        print("=== 第0步：环境配置和包导入 ===")
        # 设置数学文本渲染
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        # 导入工具类
        from utils.SpectrumProcessor import SpectrumProcessor
        from utils.PreProcessor import PreProcessor
        from utils.PLSProcessor import PLSProcessor
        from utils.NeuralNetwork import NeuralNetwork
        from utils.SampleManager import SampleManager
        from utils.EnhancedPlotManager import EnhancedPlotManager
        from utils.AbnormalValueDetector import AbnormalValueDetector
        
        # 设置工作目录
        current_path = os.getcwd()
        work_dir = current_path
        output_dir = os.path.join(work_dir, 'data')
        output_dir2 = os.path.join(work_dir, 'raman_row')
        
        # 创建必要的目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir2, exist_ok=True)
        os.makedirs('./raman_row', exist_ok=True)
        
        print(f"当前工作目录: {current_path}")
        print("环境配置完成！")
        
        self.tools = {
            'work_dir': work_dir,
            'output_dir': output_dir,
            'output_dir2': output_dir2,
            'SpectrumProcessor': SpectrumProcessor,
            'PreProcessor': PreProcessor,
            'PLSProcessor': PLSProcessor,
            'NeuralNetwork': NeuralNetwork,
            'SampleManager': SampleManager,
            'EnhancedPlotManager': EnhancedPlotManager,
            'AbnormalValueDetector': AbnormalValueDetector
        }
        
        return self.tools
    
    def step1_read_and_integrate_data(self):
        """第1步：读取和集成数据"""
        print("\n=== 第1步：读取和集成数据 ===")
        
        # 更新选择的物质
        print(f"选择的预测物质: {self.substance_name} ({self.selected_substance})")
        
        # 检查环境是否已设置
        if self.tools is None:
            print("环境未设置，正在初始化...")
            self.setup_environment()
        
        # 数据读取和集成
        processor = self.tools['SpectrumProcessor'](
            source_dir="./raman3", 
            target_dir="./raman_row", 
            data_dir="./raman_row",
            num_solutions=7,
            selected_substance=self.selected_substance
        )
        
        print(f"SpectrumProcessor配置: 选择物质={self.substance_name}, 索引={self.target_element_index}")
        
        # 执行完整处理流水线
        print("开始光谱数据处理流水线...")
        result = processor.full_processing_pipeline()
        
        if result is None or len(result) != 2:
            print("错误：数据处理流水线失败，无法获取X和Y文件路径")
            return None, None
        
        x_path, y_selected_path = result
        
        # 使用完整的y.csv而不是y_selected.csv，以保留所有7列浓度数据
        y_path = os.path.join(processor.target_dir, "y.csv")
        if not os.path.exists(y_path):
            print(f"警告：完整Y文件不存在，使用选定的Y文件: {y_selected_path}")
            y_path = y_selected_path
        else:
            print(f"使用完整Y文件: {y_path}（包含所有7列浓度数据）")
        
        if x_path is None or y_path is None:
            print("错误：X或Y文件路径为空")
            return None, None
        
        print(f"数据处理完成！")
        print(f"X数据路径: {x_path}")
        print(f"Y数据路径: {y_path}")
        
        # 缓存数据路径和processor实例
        self.data_cache['x_path'] = x_path
        self.data_cache['y_path'] = y_path
        self.data_cache['y_selected_path'] = y_selected_path
        self.data_cache['processor'] = processor
        
        return x_path, y_path
    
    def step2_data_preprocessing(self):
        """第2步：数据预处理"""
        print("\n=== 第2步：数据预处理 ===")
        
        # 检查是否有数据路径
        if 'x_path' not in self.data_cache or 'y_path' not in self.data_cache:
            print("错误：第1步未完成，无法进行数据预处理")
            return None
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 数据预处理
        preprocessor = self.tools['PreProcessor']()
        
        print("开始数据预处理...")
        
        # 加载数据
        df = pd.read_csv(self.data_cache['x_path'], index_col=0)
        X = df.values
        
        # 安全转换列名为波长，跳过无效值
        wavelengths = []
        valid_columns = []
        for i, col in enumerate(df.columns):
            try:
                wavelength = float(col)
                wavelengths.append(wavelength)
                valid_columns.append(i)
            except ValueError:
                print(f"跳过无效波长列: {col}")
                continue
        
        # 只保留有效的波长列
        X = X[:, valid_columns]
        wavelengths = np.array(wavelengths)
        
        # 预处理配置
        config = {
            'wavelength_range': (100, 1500),  # 100-1500波数范围
            'normalization': {'method': 'area'}  # 标准化（Z-score）
        }
        
        # 执行预处理
        X_processed, wavelengths_processed = preprocessor.preprocess_pipeline(X, wavelengths, config)
        
        # 保存预处理后的数据
        processed_df = pd.DataFrame(X_processed, columns=wavelengths_processed)
        x_processed_path = './raman_row/x_processed.csv'
        processed_df.to_csv(x_processed_path)
        
        if x_processed_path is None:
            print("错误：数据预处理失败")
            return None
        
        print(f"数据预处理完成！")
        print(f"处理后X数据路径: {x_processed_path}")
        
        # 缓存处理后的数据路径和波长信息
        self.data_cache['x_processed_path'] = x_processed_path
        self.data_cache['preprocessor'] = preprocessor
        self.data_cache['selected_wavelengths'] = preprocessor.selected_wavelengths
        
        return x_processed_path
    
    def step3_data_splitting(self):
        """第3步：数据集分离"""
        print("\n=== 第3步：数据集分离 ===")
        
        # 检查是否有预处理数据
        if 'x_processed_path' not in self.data_cache:
            print("错误：第2步未完成，无法进行数据集分离")
            return None
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 数据集分离
        from utils.SampleManager import SampleManager
        
        print("开始数据集分离...")
        # 加载数据（正确处理索引列）
        x_data = pd.read_csv(self.data_cache['x_processed_path'], index_col=0)  # 第一列作为索引
        y_data = pd.read_csv(self.data_cache['y_path'], header=None)  # Y文件没有列名
        
        # 使用SampleManager进行数据分割
        sample_manager = SampleManager()
        
        # 使用前7列的浓度数据生成样本ID（排除最后一列的样本ID字符串）
        if y_data.shape[1] > 7:
            y_concentrations = y_data.iloc[:, :7].values  # 只使用前7列浓度数据
        else:
            y_concentrations = y_data.values
        sample_ids, measurement_counts = sample_manager.generate_sample_ids(y_concentrations)
        
        # 按样本ID分割数据
        x_train, y_train, x_test, y_test, x_val, y_val = sample_manager.split_by_sample_id(
            X=x_data.values,  # 现在x_data.values不包含索引列
            y=y_data.iloc[:, self.target_element_index].values.reshape(-1, 1),
            sample_ids=sample_ids,
            train_ratio=0.7,
            test_ratio=0.2,
            val_ratio=0.1
        )
        
        # 同时分割样本ID
        unique_sample_ids = np.unique(sample_ids)
        np.random.seed(42)
        np.random.shuffle(unique_sample_ids)
        
        n_train = int(len(unique_sample_ids) * 0.7)
        n_test = int(len(unique_sample_ids) * 0.2)
        
        train_sample_ids_set = set(unique_sample_ids[:n_train])
        test_sample_ids_set = set(unique_sample_ids[n_train:n_train + n_test])
        val_sample_ids_set = set(unique_sample_ids[n_train + n_test:])
        
        train_sample_ids = sample_ids[np.isin(sample_ids, list(train_sample_ids_set))]
        test_sample_ids = sample_ids[np.isin(sample_ids, list(test_sample_ids_set))]
        val_sample_ids = sample_ids[np.isin(sample_ids, list(val_sample_ids_set))]
        
        # 保存分割后的数据（包含样本标签）
        # 为X数据添加样本标签列
        train_x_df = pd.DataFrame(x_train)
        train_x_df['sample_id'] = train_sample_ids
        train_x_df.to_csv(os.path.join(self.tools['output_dir2'], 'train_X.csv'), index=False)
        
        train_y_df = pd.DataFrame(y_train)
        train_y_df['sample_id'] = train_sample_ids
        train_y_df.to_csv(os.path.join(self.tools['output_dir2'], 'train_Y.csv'), index=False)
        
        val_x_df = pd.DataFrame(x_val)
        val_x_df['sample_id'] = val_sample_ids
        val_x_df.to_csv(os.path.join(self.tools['output_dir2'], 'val_X.csv'), index=False)
        
        val_y_df = pd.DataFrame(y_val)
        val_y_df['sample_id'] = val_sample_ids
        val_y_df.to_csv(os.path.join(self.tools['output_dir2'], 'val_Y.csv'), index=False)
        
        test_x_df = pd.DataFrame(x_test)
        test_x_df['sample_id'] = test_sample_ids
        test_x_df.to_csv(os.path.join(self.tools['output_dir2'], 'test_X.csv'), index=False)
        
        test_y_df = pd.DataFrame(y_test)
        test_y_df['sample_id'] = test_sample_ids
        test_y_df.to_csv(os.path.join(self.tools['output_dir2'], 'test_Y.csv'), index=False)
        
        split_result = (x_train, y_train, x_val, y_val, x_test, y_test)
        
        if split_result is None:
            print("错误：数据集分离失败")
            return None
        
        print(f"数据集分离完成！")
        print(f"训练集形状: X: {x_train.shape}, Y: {y_train.shape}")
        print(f"验证集形状: X: {x_val.shape}, Y: {y_val.shape}")
        print(f"测试集形状: X: {x_test.shape}, Y: {y_test.shape}")
        
        # 缓存分割后的数据和样本ID
        self.data_cache['x_train'] = x_train
        self.data_cache['y_train'] = y_train
        self.data_cache['x_val'] = x_val
        self.data_cache['y_val'] = y_val
        self.data_cache['x_test'] = x_test
        self.data_cache['y_test'] = y_test
        self.data_cache['train_sample_ids'] = train_sample_ids
        self.data_cache['val_sample_ids'] = val_sample_ids
        self.data_cache['test_sample_ids'] = test_sample_ids
        self.data_cache['sample_manager'] = sample_manager
        self.data_cache['instance'] = self.substance_name
        
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    def step4_data_visualization(self):
        """第4步：训练前总画图"""
        print("\n=== 第4步：训练前总画图 ===")
        
        # 检查是否有分离后的数据
        if not all(key in self.data_cache for key in ['x_train', 'y_train', 'x_test', 'y_test']):
            print("错误：第3步未完成，无法进行数据可视化")
            return None
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 数据可视化
        plot_manager = self.tools['EnhancedPlotManager']()
        sample_manager = self.tools['SampleManager']()
        
        print("开始数据可视化...")
        
        # 1. 真正的光谱图 - 按样本分组显示
        # 绘制光谱图 - 使用真实波数
        selected_wl_info = self.data_cache.get('selected_wavelengths')
        if selected_wl_info and 'wavelengths' in selected_wl_info:
            x_wavelengths = selected_wl_info['wavelengths']
            print(f"使用真实波数，范围: {x_wavelengths[0]:.1f} - {x_wavelengths[-1]:.1f} cm⁻¹")
        else:
            # 如果没有波数信息，尝试从x_processed.csv的列名获取
            try:
                processed_df = pd.read_csv(self.data_cache.get('x_processed_path', './raman_row/x_processed.csv'))
                x_wavelengths = np.array([float(col) for col in processed_df.columns])
                print(f"从处理后文件获取波数，范围: {x_wavelengths[0]:.1f} - {x_wavelengths[-1]:.1f} cm⁻¹")
            except:
                x_wavelengths = np.arange(self.data_cache['x_train'].shape[1])
                print(f"警告：无法获取真实波数，使用索引作为x轴: 0 - {len(x_wavelengths)-1}")
        
        # 获取样本ID和样本管理器
        if 'train_sample_ids' in self.data_cache and 'sample_manager' in self.data_cache:
            train_sample_ids = self.data_cache['train_sample_ids']
            sample_manager = self.data_cache['sample_manager']
            
            # 随机选择多个样本进行展示
            unique_samples = np.unique(train_sample_ids)
            selected_samples = np.random.choice(unique_samples, min(5, len(unique_samples)), replace=False)
            
            # 收集选中样本的所有光谱数据，并为每个样本分配颜色
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            # 为每个样本ID分配固定颜色
            colors = list(mcolors.TABLEAU_COLORS.values())
            sample_colors = {sample_id: colors[i % len(colors)] for i, sample_id in enumerate(selected_samples)}
            
            plt.figure(figsize=(12, 8))
            for sample_id in selected_samples:
                sample_mask = train_sample_ids == sample_id
                sample_spectra = self.data_cache['x_train'][sample_mask]
                color = sample_colors[sample_id]
                
                # 绘制该样本的所有光谱，使用相同颜色
                for i, spectrum in enumerate(sample_spectra):
                    label = f'样本ID {sample_id}' if i == 0 else None  # 只为第一条光谱添加标签
                    plt.plot(x_wavelengths, spectrum, color=color, alpha=0.7, linewidth=0.8, label=label)
            
            plt.title(f"{self.substance_name} - 多样本所有光谱图")
            plt.xlabel("波数 (cm⁻¹)")
            plt.ylabel("吸光度")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图片
            plot_path = f"./plots/{self.substance_name}_多样本光谱图.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"光谱图已保存到: {plot_path}")
        else:
            # 备用方案：显示随机光谱
            random_indices = np.random.choice(len(self.data_cache['x_train']), min(10, len(self.data_cache['x_train'])), replace=False)
            random_spectra = self.data_cache['x_train'][random_indices]
            
            plot_manager.plot_multi_line(
                x=x_wavelengths,
                Y=random_spectra,
                title=f"{self.substance_name} - 光谱图",
                xlabel="波数 (cm⁻¹)",
                ylabel="吸光度"
            )
        
        print("数据可视化完成！")
        
        return True
    
    def step4_5_load_preprocessed_data(self):
        """中断点：读取预处理数据"""
        print("\n=== 中断点：读取预处理数据 ===")
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 从文件加载预处理数据
        data_dir = self.tools['output_dir2']
        
        try:
            # 加载训练集数据（正确处理header和样本标签列）
            train_x_df = pd.read_csv(os.path.join(data_dir, 'train_X.csv'))
            x_train = train_x_df.iloc[:, :-1].values  # 排除最后一列样本标签
            train_sample_ids = train_x_df['sample_id'].values
            
            # 获取波数信息（从列名）
            wavelength_columns = [col for col in train_x_df.columns if col != 'sample_id']
            try:
                wavelengths = np.array([float(col) for col in wavelength_columns])
                print(f"从CSV文件获取波数信息，范围: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} cm⁻¹")
                # 保存波数信息到data_cache
                self.data_cache['selected_wavelengths'] = {
                    'wavelengths': wavelengths,
                    'start': wavelengths[0],
                    'end': wavelengths[-1]
                }
            except ValueError:
                print("警告：无法从CSV列名解析波数信息")
            
            train_y_df = pd.read_csv(os.path.join(data_dir, 'train_Y.csv'))
            y_train = train_y_df.iloc[:, 0].values  # 分割后的Y文件第0列是目标值
            
            # 加载验证集数据（排除样本标签列）
            val_x_df = pd.read_csv(os.path.join(data_dir, 'val_X.csv'))
            x_val = val_x_df.iloc[:, :-1].values  # 排除最后一列样本标签
            val_sample_ids = val_x_df['sample_id'].values
            
            val_y_df = pd.read_csv(os.path.join(data_dir, 'val_Y.csv'))
            y_val = val_y_df.iloc[:, 0].values  # 分割后的Y文件第0列是目标值
            
            # 加载测试集数据（排除样本标签列）
            test_x_df = pd.read_csv(os.path.join(data_dir, 'test_X.csv'))
            x_test = test_x_df.iloc[:, :-1].values  # 排除最后一列样本标签
            test_sample_ids = test_x_df['sample_id'].values
            
            test_y_df = pd.read_csv(os.path.join(data_dir, 'test_Y.csv'))
            y_test = test_y_df.iloc[:, 0].values  # 分割后的Y文件第0列是目标值
            
            print(f"数据加载完成！")
            print(f"训练集形状: X: {x_train.shape}, Y: {y_train.shape}")
            print(f"验证集形状: X: {x_val.shape}, Y: {y_val.shape}")
            print(f"测试集形状: X: {x_test.shape}, Y: {y_test.shape}")
            
            # 缓存加载的数据
            self.data_cache.update({
                'x_train': x_train,
                'y_train': y_train,
                'x_val': x_val,
                'y_val': y_val,
                'x_test': x_test,
                'y_test': y_test,
                'train_sample_ids': train_sample_ids,
                'val_sample_ids': val_sample_ids,
                'test_sample_ids': test_sample_ids,
                'instance': self.substance_name
            })
            
            return x_train, y_train, x_val, y_val, x_test, y_test
            
        except Exception as e:
            print(f"错误：无法加载预处理数据 - {e}")
            return None
    
    def step5_pls_analysis(self):
        """第5步：PLS回归分析"""
        print("\n=== 第5步：PLS回归分析 ===")
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 确保数据已加载
        if not all(key in self.data_cache for key in ['x_train', 'y_train', 'x_test', 'y_test']):
            print("数据未加载，先执行中断点加载数据...")
            self.step4_5_load_preprocessed_data()
        
        # PLS回归分析
        pls_processor = self.tools['PLSProcessor']()
        
        print("开始PLS回归分析...")
        
        # 第一部分：PLS训练（仅使用训练集，输出训练损失图）
        print("第一部分：PLS训练...")
        model, Y_calib_pred = pls_processor.pls_training(
            X_calib=self.data_cache['x_train'], 
            Y_calib=self.data_cache['y_train'], 
            loss='huber', 
            item=self.substance_name
        )
        print('第一部分训练结束!!!\n\n\n\n\n\n\n\n')
        print(f"PLS最优组件数: {pls_processor.best_components}")
        
        # 训练后进行简单评估
        print("\n--- PLS训练后评估 ---")
        pls_processor._evaluate_model(
            y_train_true=self.data_cache['y_train'],
            y_train_pred=Y_calib_pred,
            y_test_true=self.data_cache['y_train'],  # 用训练集作为测试集进行简单评估
            y_test_pred=Y_calib_pred,
            train_sample_ids=self.data_cache['train_sample_ids'],
            test_sample_ids=self.data_cache['train_sample_ids'],
            item=self.substance_name
        )
        print("第一部分后面的简单评估完成!!!!!\n\n\n\n\n\n\n\n\n\n")
        
        # 第二部分：PLS预测（传入所有数据，进行样本点预测，输出预测图像和样本方差图）
        print("第二部分：PLS预测...")
        predictions = pls_processor.pls_prediction(
            x_train=self.data_cache['x_train'],
            y_train=self.data_cache['y_train'],
            x_val=self.data_cache['x_val'],
            y_val=self.data_cache['y_val'],
            x_test=self.data_cache['x_test'],
            y_test=self.data_cache['y_test'],
            train_sample_ids=self.data_cache['train_sample_ids'],
            val_sample_ids=self.data_cache['val_sample_ids'],
            test_sample_ids=self.data_cache['test_sample_ids'],
            loss='huber',
            item=self.substance_name
        )
        print("第二部分的模型预测完成！\n\n\n\n\n\n\n\n")
        # 获取预测结果
        Y_valid_pred = predictions['y_test_pred']
        
        # 计算并保存PLS指标
        from sklearn.metrics import r2_score, mean_squared_error
        
        r2 = r2_score(self.data_cache['y_test'], Y_valid_pred)
        rmse = np.sqrt(mean_squared_error(self.data_cache['y_test'], Y_valid_pred))
        
        # 保存PLS指标到缓存
        self.data_cache['pls_metrics'] = {
            'r2': f"{r2:.4f}",
            'rmse': f"{rmse:.4f}",
            'best_components': pls_processor.best_components
        }
        
        # 缓存PLS处理器和预测结果
        self.data_cache.update({
            'pls_processor': pls_processor,
            'y_train_pred': predictions['y_train_pred'],
            'y_val_pred': predictions['y_val_pred'],
            'y_test_pred': predictions['y_test_pred'],
            'predictions': predictions
        })
        
        print(f"PLS回归分析完成！R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        return Y_calib_pred, Y_valid_pred
    
    def step5_5_pls_evaluation(self):
        """第5.5步：PLS模型评估基于样本点预测"""
        print("\n=== 第5.5步：PLS模型评估基于样本点预测 ===")
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 确保PLS模型已训练
        if 'pls_processor' not in self.data_cache:
            print("PLS模型未训练，先执行PLS分析...")
            self.step5_pls_analysis()
        
        # 获取PLS预测结果
        train_predictions = self.data_cache['y_train_pred']
        test_predictions = self.data_cache['y_test_pred']
        val_predictions = self.data_cache['y_val_pred']
        
        # 使用ErrorDeviationAnalyzer进行样本级别的误差和方差分析
        from utils.ErrorDeviationAnalyzer import ErrorDeviationAnalyzer
        from utils.EnhancedPlotManager import EnhancedPlotManager
        
        enhanced_plot_manager = EnhancedPlotManager()
        error_analyzer = ErrorDeviationAnalyzer(plot_manager=enhanced_plot_manager)
        
        # 准备数据集字典
        datasets = {
            'train': {
                'y_true': self.data_cache['y_train'].flatten(),
                'y_pred': train_predictions,
                'sample_ids': self.data_cache['train_sample_ids']
            },
            'test': {
                'y_true': self.data_cache['y_test'].flatten(),
                'y_pred': test_predictions,
                'sample_ids': self.data_cache['test_sample_ids']
            },
            'val': {
                'y_true': self.data_cache['y_val'].flatten(),
                'y_pred': val_predictions,
                'sample_ids': self.data_cache['val_sample_ids']
            }
        }
        
        # 进行多数据集误差分析
        print("开始PLS样本级别误差和方差分析...")
        error_analyzer.analyze_multi_dataset_errors(
            datasets=datasets,
            title_prefix=f"{self.substance_name}_PLS"
        )
        
        # 生成样本偏差和方差分析图
        print("生成PLS样本偏差和方差分析图...")
        # 对训练集进行偏差方差分析
        error_analyzer.plot_bias_variance_analysis(
            y_true=self.data_cache['y_train'].flatten(),
            y_pred=train_predictions,
            sample_ids=self.data_cache['train_sample_ids'],
            config={'substance_name': self.substance_name},
            model_name=f"{self.substance_name}_PLS_Train"
        )
        # 对测试集进行偏差方差分析
        error_analyzer.plot_bias_variance_analysis(
            y_true=self.data_cache['y_test'].flatten(),
            y_pred=test_predictions,
            sample_ids=self.data_cache['test_sample_ids'],
            config={'substance_name': self.substance_name},
            model_name=f"{self.substance_name}_PLS_Test"
        )
        print("第三步,样本级别的误差和方差分析已经完成!!!!!!!")
        
        print("已生成PLS样本级别的误差和方差分析图")
        print("PLS模型评估完成！")
        
        # 缓存评估结果
        pls_evaluation_results = {
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'val_predictions': val_predictions
        }
        self.data_cache['pls_evaluation_results'] = pls_evaluation_results
        
        return pls_evaluation_results
    
    def step6_bayesian_optimization(self):
        """第6步：超参数优化（贝叶斯优化）"""
        print("\n=== 第6步：超参数优化（贝叶斯优化） ===")
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 确保数据已加载
        if not all(key in self.data_cache for key in ['x_train', 'y_train', 'x_test', 'y_test']):
            print("数据未加载，先执行中断点加载数据...")
            self.step4_5_load_preprocessed_data()
        
        # 贝叶斯优化
        optimizer = self.tools['NeuralNetwork']()
        
        print("开始贝叶斯优化...")
        optimization_result = optimizer.bayesian_optimization(
            X_train=self.data_cache['x_train'],
            y_train=self.data_cache['y_train'],
            X_test=self.data_cache['x_test'],
            y_test=self.data_cache['y_test'],
            n_trials=10,
            epochs=200,
            batch_size=32,
            item=self.substance_name,
            loss_function=self.loss_function
        )
        
        best_params = optimization_result['best_params']
        
        print(f"贝叶斯优化完成！最佳参数: {best_params}")
        
        # 缓存优化结果
        self.data_cache['best_params'] = best_params
        
        return best_params
    
    def step7_neural_network_training(self):
        """第7步：神经网络基于测量数据点训练"""
        print("\n=== 第7步：神经网络基于测量数据点训练 ===")
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 确保数据和优化参数已准备
        if not all(key in self.data_cache for key in ['x_train', 'y_train', 'x_test', 'y_test']):
            print("数据未加载，先执行中断点加载数据...")
            self.step4_5_load_preprocessed_data()
        
        if 'best_params' not in self.data_cache:
            print("未找到优化参数，使用默认参数...")
            self.data_cache['best_params'] = {
                'DenseN': 128,
                'DropoutR': 0.04,
                'C1_K': 92,
                'C1_S': 56,
                'C2_K': 108,
                'C2_S': 72,
                'learning_rate': 0.001
            }
        
        # 神经网络训练
        trainer = self.tools['NeuralNetwork']()
        
        print("开始神经网络训练...")
        training_results = trainer.train_model(
            X_train=self.data_cache['x_train'],
            y_train=self.data_cache['y_train'],
            X_test=self.data_cache['x_test'],
            y_test=self.data_cache['y_test'],
            model_params=self.data_cache['best_params'],
            epochs=200,
            batch_size=32,
            item=self.substance_name,
            save_path='./models/best_model.h5',
            loss_function=self.loss_function
        )
        
        # 训练后进行简单预测评估
        print("\n--- 训练后评估 ---")
        evaluation_results = trainer.evaluate_model(
            X_train=self.data_cache['x_train'],
            y_train=self.data_cache['y_train'],
            X_test=self.data_cache['x_test'],
            y_test=self.data_cache['y_test'],
            item=self.substance_name,
            train_sample_ids=self.data_cache['train_sample_ids'],
            test_sample_ids=self.data_cache['test_sample_ids']
        )
        
        print(f"神经网络训练完成！")
        print(f"训练集R²: {training_results['train_metrics']['r2']:.4f}, 测试集R²: {training_results['test_metrics']['r2']:.4f}")
        print(f"训练集RMSE: {training_results['train_metrics']['rmse']:.4f}, 测试集RMSE: {training_results['test_metrics']['rmse']:.4f}")
        
        # 缓存训练结果和评估结果
        self.data_cache['training_results'] = training_results
        self.data_cache['evaluation_results'] = evaluation_results
        
        return training_results
    
    def step8_model_evaluation(self):
        """第8步：模型评估基于样本点预测"""
        print("\n=== 第8步：模型评估基于样本点预测 ===")
        
        # 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 确保模型已训练
        if 'training_results' not in self.data_cache:
            print("模型未训练，先执行神经网络训练...")
            self.step7_neural_network_training()
        
        # 模型评估 - 使用evaluate_model获取三个数据集的预测结果
        evaluator = self.tools['NeuralNetwork']()
        model_path = self.data_cache['training_results']['model_path']
        
        print("开始三数据集模型评估...")
        
        # 分别对三个数据集进行评估（不绘图）
        train_eval = evaluator.evaluate_model(
            X_train=self.data_cache['x_train'],
            y_train=self.data_cache['y_train'],
            X_test=self.data_cache['x_train'],  # 用训练集作为测试集进行评估
            y_test=self.data_cache['y_train'],
            model_path=model_path,
            item=self.substance_name,
            train_sample_ids=self.data_cache['train_sample_ids'],
            test_sample_ids=self.data_cache['train_sample_ids'],
            enable_plotting=False
        )
        
        test_eval = evaluator.evaluate_model(
            X_train=self.data_cache['x_test'],
            y_train=self.data_cache['y_test'],
            X_test=self.data_cache['x_test'],  # 用测试集作为测试集进行评估
            y_test=self.data_cache['y_test'],
            model_path=model_path,
            item=self.substance_name,
            train_sample_ids=self.data_cache['test_sample_ids'],
            test_sample_ids=self.data_cache['test_sample_ids'],
            enable_plotting=False
        )
        
        val_eval = evaluator.evaluate_model(
            X_train=self.data_cache['x_val'],
            y_train=self.data_cache['y_val'],
            X_test=self.data_cache['x_val'],  # 用验证集作为测试集进行评估
            y_test=self.data_cache['y_val'],
            model_path=model_path,
            item=self.substance_name,
            train_sample_ids=self.data_cache['val_sample_ids'],
            test_sample_ids=self.data_cache['val_sample_ids'],
            enable_plotting=False
        )
        
        # 提取预测结果
        train_predictions = train_eval['train_predictions']
        test_predictions = test_eval['train_predictions']
        val_predictions = val_eval['train_predictions']
        
        # 使用ErrorDeviationAnalyzer进行样本级别的误差和方差分析
        from utils.ErrorDeviationAnalyzer import ErrorDeviationAnalyzer
        from utils.EnhancedPlotManager import EnhancedPlotManager
        
        enhanced_plot_manager = EnhancedPlotManager()
        error_analyzer = ErrorDeviationAnalyzer(plot_manager=enhanced_plot_manager)
        
        # 准备数据集字典
        datasets = {
            'train': {
                'y_true': self.data_cache['y_train'].flatten(),
                'y_pred': train_predictions,
                'sample_ids': self.data_cache['train_sample_ids']
            },
            'test': {
                'y_true': self.data_cache['y_test'].flatten(),
                'y_pred': test_predictions,
                'sample_ids': self.data_cache['test_sample_ids']
            },
            'val': {
                'y_true': self.data_cache['y_val'].flatten(),
                'y_pred': val_predictions,
                'sample_ids': self.data_cache['val_sample_ids']
            }
        }
        
        # 进行多数据集误差分析
        print("开始样本级别误差和方差分析...")
        error_analyzer.analyze_multi_dataset_errors(
            datasets=datasets,
            title_prefix=f"{self.substance_name}"
        )
        
        # 生成样本偏差和方差分析图
        print("生成神经网络样本偏差和方差分析图...")
        # 对训练集进行偏差方差分析
        error_analyzer.plot_bias_variance_analysis(
            y_true=self.data_cache['y_train'].flatten(),
            y_pred=train_predictions,
            sample_ids=self.data_cache['train_sample_ids'],
            config={'substance_name': self.substance_name},
            model_name=f"{self.substance_name}_NN_Train"
        )
        # 对测试集进行偏差方差分析
        error_analyzer.plot_bias_variance_analysis(
            y_true=self.data_cache['y_test'].flatten(),
            y_pred=test_predictions,
            sample_ids=self.data_cache['test_sample_ids'],
            config={'substance_name': self.substance_name},
            model_name=f"{self.substance_name}_NN_Test"
        )
        
        print("已生成样本级别的误差和方差分析图")
        print("模型评估完成！")
        
        # 缓存评估结果
        evaluation_results = {
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'val_predictions': val_predictions
        }
        self.data_cache['evaluation_results'] = evaluation_results
        
        return evaluation_results
    
    def step9_new_sample_prediction(self, spectrum_data=None):
        """第9步：新样本预测"""
        print("\n=== 第9步：新样本预测 ===,pass掉了")
        
        '''# 确保工具类已初始化
        if self.tools is None:
            self.setup_environment()
        
        # 确保模型已评估
        if 'evaluation_results' not in self.data_cache:
            print("模型未评估，先执行模型评估...")
            self.step8_model_evaluation()
        
        # 新样本预测
        if spectrum_data is None:
            print("未提供光谱数据，使用测试集第一个样本作为示例")
            spectrum_data = self.data_cache['x_val'][0:1]
        
        # 使用训练好的模型进行预测
        predictor = self.tools['NeuralNetwork']()
        model_path = self.data_cache['training_results']['model_path']
        
        print("开始新样本预测...")
        prediction = predictor.predict_new_samples(spectrum_data, model_path)
        
        # 由于现在只有一个模型，直接使用其预测结果
        mean_prediction = prediction
        std_prediction = np.zeros_like(prediction)  # 单个模型没有方差
        
        print(f"新样本预测完成！")
        print(f"预测值: {float(mean_prediction[0]):.4f}")
        print(f"预测标准差: {float(std_prediction[0]):.4f}")
        
        return mean_prediction, std_prediction'''
    
    def get_data_cache(self):
        """获取数据缓存"""
        return self.data_cache
    
    def get_tools(self):
        """获取工具类字典"""
        return self.tools