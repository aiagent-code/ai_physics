#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### ProcessCenter.py - 主控制器接口和交互模块
**主要功能：作为九步标准流程的主控制器接口，包含交互式菜单**

九步标准流程：
1. 数据读取和整合
2. 数据预处理（波段选择、归一化等）
3. 数据集分离（训练/验证/测试）
4. 数据可视化（训练前总画图）
5. PLS回归分析
5.5. PLS样本级别评估
6. 超参数优化（贝叶斯优化）
7. 神经网络训练
8. 模型评估
9. 新样本预测

附加功能：
- 从预处理数据重新开始（中断点恢复）
- 独立异常值清洗
- 交互式菜单系统

类功能说明：
- ProcessCenter: 主控制器接口，委托具体实现给PreCenter和DataCleaner

主要方法：
- setup_environment(): 环境配置和包导入
- step1_read_and_integrate_data(): 数据读取和整合
- step2_data_preprocessing(): 数据预处理
- step3_data_splitting(): 数据集分离
- step4_data_visualization(): 数据可视化
- step5_pls_analysis(): PLS回归分析
- step5_5_pls_evaluation(): PLS样本级别评估
- step6_hyperparameter_optimization(): 超参数优化
- step7_neural_network_training(): 神经网络训练
- step8_model_evaluation(): 模型评估
- step9_new_sample_prediction(): 新样本预测
- load_preprocessed_data(): 从预处理数据重新开始
- clean_data(): 独立异常值清洗
- run_step(): 执行指定步骤
- show_menu(): 显示处理步骤菜单
- run_interactive(): 交互式执行处理步骤
- run_complete_workflow(): 执行完整工作流程
"""

import os
import sys
from precenter import PreCenter
from DataCleaner import DataCleaner

class ProcessCenter:
    """
    LIBS光谱数据处理中心接口类
    
    功能说明:
    - 提供标准九步处理流程的统一接口
    - 委托PreCenter实现具体的处理逻辑
    - 支持独立的异常值清洗功能
    - 符合pro.md规范的流程设计
    
    标准九步方法:
    1. step1_read_and_integrate_data(): 读取文件内容，生成整合数据文件
    2. step2_data_preprocessing(): 数据预处理（选取特定波段，进行归一化等）
    3. step3_data_splitting(): 数据集分离
    4. step4_data_visualization(): 训练前总画图
    5. step5_pls_analysis(): PLS回归分析
    5.5. step5_5_pls_evaluation(): PLS样本级别评估
    6. step6_bayesian_optimization(): 超参数优化（贝叶斯优化）
    7. step7_neural_network_training(): 神经网络基于测量数据点训练
    8. step8_model_evaluation(): 模型评估基于样本点预测
    9. step9_new_sample_prediction(): 新样本预测
    
    附加方法:
    - load_preprocessed_data(): 中断点，从预处理数据重新开始
    - clean_data(): 独立异常值清洗
    - run_step(): 运行指定步骤
    
    属性:
    - precenter: PreCenter实例，负责具体实现
    - data_cleaner: DataCleaner实例，负责异常值清洗
    - selected_substance: 选择的物质字母
    """
    
    def __init__(self, selected_substance='b'):
        """初始化处理中心接口
        
        参数:
        selected_substance: 选择的物质字母 (a-g)，默认为'b'(丙三醇)
                          a: 异丙醇, b: 丙三醇, c: 乙二醇, d: 聚乙二醇, 
                          e: 乙酸, f: 二甲基乙枫, g: 三乙醇胺
        """
        self.selected_substance = selected_substance
        
        # 初始化具体实现类
        self.precenter = PreCenter(selected_substance=selected_substance)
        self.data_cleaner = None  # 按需初始化
        
        print(f"ProcessCenter接口初始化完成，选择物质: {selected_substance}")
    
    # ==================== 标准九步流程接口 ====================
    
    def setup_environment(self):
        """环境配置和包导入（委托给PreCenter）"""
        return self.precenter.setup_environment()
    
    def step1_read_and_integrate_data(self):
        """第1步：读取文件内容，生成整合数据文件"""
        return self.precenter.step1_read_and_integrate_data()
    
    def step2_data_preprocessing(self):
        """第2步：数据预处理（选取特定波段，进行归一化等）"""
        return self.precenter.step2_data_preprocessing()
    
    def step3_data_splitting(self):
        """第3步：数据集分离"""
        return self.precenter.step3_data_splitting()
    
    def step4_data_visualization(self):
        """第4步：训练前总画图"""
        return self.precenter.step4_data_visualization()
    
    def step5_pls_analysis(self):
        """第5步：PLS回归分析"""
        return self.precenter.step5_pls_analysis()
    
    def step5_5_pls_evaluation(self):
        """第5.5步：PLS样本级别评估"""
        return self.precenter.step5_5_pls_evaluation()
    
    def step6_bayesian_optimization(self):
        """第6步：超参数优化（贝叶斯优化）"""
        return self.precenter.step6_bayesian_optimization()
    
    def step7_neural_network_training(self):
        """第7步：神经网络基于测量数据点训练"""
        return self.precenter.step7_neural_network_training()
    
    def step8_model_evaluation(self):
        """第8步：模型评估基于样本点预测"""
        return self.precenter.step8_model_evaluation()
    
    def step9_new_sample_prediction(self, spectrum_data=None):
        """第9步：新样本预测"""
        return self.precenter.step9_new_sample_prediction(spectrum_data)
    
    # ==================== 附加功能接口 ====================
    
    def load_preprocessed_data(self):
        """中断点：从预处理数据重新开始"""
        return self.precenter.step4_5_load_preprocessed_data()
    
    def clean_data(self, data_path='./data_row', prediction_method='pls', 
                   detection_method='3sigma', threshold_factor=3.0):
        """独立异常值清洗功能
        
        参数:
        data_path: 数据文件路径
        prediction_method: 预测方法 ('pls' 或 'neural_network')
        detection_method: 检测方法 ('3sigma', 'iqr', 'prediction_error')
        threshold_factor: 阈值因子
        
        返回:
        dict: 清洗结果
        """
        print(f"\n=== 启动独立异常值清洗 ===")
        print(f"数据路径: {data_path}")
        print(f"预测方法: {prediction_method.upper()}")
        print(f"检测方法: {detection_method.upper()}")
        
        # 初始化数据清洗器
        self.data_cleaner = DataCleaner(
            data_path=data_path,
            prediction_method=prediction_method
        )
        
        # 执行清洗
        result = self.data_cleaner.clean_data(
            method=detection_method,
            threshold_factor=threshold_factor,
            save_cleaned_data=True
        )
        
        return result
    
    # ==================== 统一调用接口 ====================
    
    def run_step(self, step_number, **kwargs):
        """运行指定步骤
        
        参数:
        step_number: 步骤编号 (0-9)
        **kwargs: 传递给各步骤的参数
        
        返回:
        步骤执行结果
        """
        if step_number == 0:
            return self.setup_environment()
        elif step_number == 1:
            return self.step1_read_and_integrate_data()
        elif step_number == 2:
            return self.step2_data_preprocessing()
        elif step_number == 3:
            return self.step3_data_splitting()
        elif step_number == 4:
            return self.step4_data_visualization()
        elif step_number == 4.5:
            return self.load_preprocessed_data()
        elif step_number == 5:
            return self.step5_pls_analysis()
        elif step_number == 5.5:
            return self.step5_5_pls_evaluation()
        elif step_number == 6:
            return self.step6_bayesian_optimization()
        elif step_number == 7:
            return self.step7_neural_network_training()
        elif step_number == 8:
            return self.step8_model_evaluation()
        elif step_number == 9:
            return self.step9_new_sample_prediction(kwargs.get('spectrum_data', None))
        else:
            print(f"无效的步骤号: {step_number}")
            print("有效步骤号: 0-9, 4.5(中断点), 5.5(PLS评估)")
            return None
    
    def show_menu(self):
        """显示处理步骤菜单"""
        print("\n" + "="*50)
        print("    LIBS光谱数据处理系统 (九步标准流程)")
        print("="*50)
        print("请选择要执行的步骤:")
        print("0. 环境配置和包导入")
        print("1. 读取文件内容，生成整合数据文件")
        print("2. 数据预处理（选取特定波段，进行归一化等）")
        print("3. 数据集分离")
        print("4. 训练前总画图")
        print("4.5. 从预处理数据重新开始（中断点）")
        print("5. PLS回归分析")
        print("5.5. PLS样本级别评估")
        print("6. 超参数优化（贝叶斯优化）")
        print("7. 神经网络基于测量数据点训练")
        print("8. 模型评估基于样本点预测")
        print("9. 新样本预测")
        print("-"*30)
        print("10. 执行完整九步流程 (1-9步)")
        print("11. 独立异常值清洗")
        print("12. 退出")
        print("="*50)

    def run_interactive(self):
        """交互式运行"""
        while True:
            self.show_menu()
            try:
                choice = input("\n请输入步骤号 (0-12): ").strip()
                
                if choice == '12':
                    print("退出程序")
                    break
                elif choice == '10':
                    a=int(input("请输入选择"))
                    if a==0:
                        print("\n=== 执行完整九步流程 ===")
                        self.run_complete_workflow()
                    else:
                        print("执行4.5-8步")
                        self.run_step5_8()
                elif choice == '11':
                    print("\n=== 独立异常值清洗 ===")
                    self.run_data_cleaning()
                elif choice.replace('.', '').isdigit():
                    step_num = float(choice)
                    if step_num in [0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 9]:
                        print(f"\n=== 执行第{step_num}步 ===")
                        result = self.run_step(step_num)
                        if result:
                            print(f"第{step_num}步执行完成！")
                        input("\n按回车键继续...")
                    else:
                        print("无效的步骤号，请输入0-9或10-12")
                else:
                    print("请输入有效的数字")
                    
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"执行过程中出现错误: {e}")
                import traceback
                traceback.print_exc()
                input("\n按回车键继续...")
    

    def run_step5_8(self):
        """运行步骤4.5-8"""

        self.run_step(4.5)
        # 第5步：PLS回归分析
        self.run_step(5)
        self.run_step(5.5)
        # 第6步：超参数优化（贝叶斯优化）
        self.run_step(6)
            
        # 第7步：神经网络训练
        self.run_step(7)
            
        # 第8步：模型评估
        self.run_step(8)

    def run_complete_workflow(self):
        """运行完整九步流程"""
        print("=== LIBS光谱数据处理中心 - 九步标准流程开始 ===\n")
        
        try:
            # 第1步：读取和整合数据
            self.run_step(1)
            
            # 第2步：数据预处理
            self.run_step(2)
            
            # 第3步：数据集分离
            self.run_step(3)
            
            # 第4步：数据可视化
            self.run_step(4)
            
            # 第5步：PLS回归分析
            self.run_step(5)
            
            # 第6步：超参数优化（贝叶斯优化）
            self.run_step(6)
            
            # 第7步：神经网络训练
            self.run_step(7)
            
            # 第8步：模型评估
            self.run_step(8)
            
            # 第9步：新样本预测
            self.run_step(9)
            
            print("\n=== 九步标准流程执行完毕 ===")
            print("所有步骤已完成，包括:")
            print("1. 文件读取和整合 ✓")
            print("2. 数据预处理和归一化 ✓")
            print("3. 数据集分离 ✓")
            print("4. 训练前总画图 ✓")
            print("5. PLS回归分析 ✓")
            print("6. 超参数优化（贝叶斯优化）✓")
            print("7. 神经网络基于测量数据点训练 ✓")
            print("8. 模型评估基于样本点预测 ✓")
            print("9. 新样本预测 ✓")
            return True
            
        except Exception as e:
            print(f"流程执行过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_data_cleaning(self):
         """独立异常值清洗功能"""
         print("=== 独立异常值清洗开始 ===\n")
         
         try:
             # 获取用户输入
             data_path = input("请输入数据路径 (默认: ./data_row): ").strip() or './data_row'
             
             print("\n请选择预测方法:")
             print("1. PLS")
             print("2. 神经网络")
             method_choice = input("请输入选择 (1-2): ").strip()
             prediction_method = 'pls' if method_choice == '1' else 'neural_network'
             
             print("\n请选择检测方法:")
             print("1. 3-sigma")
             print("2. IQR")
             print("3. 预测误差")
             detection_choice = input("请输入选择 (1-3): ").strip()
             detection_methods = {'1': '3sigma', '2': 'iqr', '3': 'prediction_error'}
             detection_method = detection_methods.get(detection_choice, '3sigma')
             
             threshold_factor = float(input("请输入阈值因子 (默认: 3.0): ").strip() or '3.0')
             
             # 执行异常值清洗
             result = self.clean_data(
                 data_path=data_path,
                 prediction_method=prediction_method,
                 detection_method=detection_method,
                 threshold_factor=threshold_factor
             )
             
             if result:
                 print("\n异常值清洗完成！")
                 print(f"清洗结果: {result}")
             else:
                 print("异常值清洗失败！")
                 
         except Exception as e:
             print(f"异常值清洗过程中出现错误: {e}")
             import traceback
             traceback.print_exc()
         
         input("\n按回车键继续...")
    
    # ==================== 数据访问接口 ====================
    
    def get_data_cache(self):
        """获取数据缓存"""
        return self.precenter.get_data_cache()
    
    def get_tools(self):
        """获取工具类字典"""
        return self.precenter.get_tools()
    
    def get_cleaning_results(self):
        """获取清洗结果"""
        if self.data_cleaner is not None:
            return self.data_cleaner.get_cleaning_results()
        else:
            return None
    
    # ==================== 兼容性方法 ====================
    
    def step4_5_load_preprocessed_data(self):
        """兼容性方法：中断点加载数据"""
        return self.load_preprocessed_data()
    
    def step10_prediction_and_sample_analysis(self, spectrum_data=None):
        """兼容性方法：样本预测分析（映射到第9步）"""
        print("注意：step10已合并到step9，正在调用step9_new_sample_prediction")
        return self.step9_new_sample_prediction(spectrum_data)


def main():
    """主函数"""
    print("=== LIBS光谱数据处理中心 - 启动 ===\n")
    
    # 创建ProcessCenter实例
    process_center = ProcessCenter()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        try:
            step_number = float(sys.argv[1])
            print(f"从命令行参数执行第{step_number}步...")
            result = process_center.run_step(step_number)
            if result is not None:
                print(f"第{step_number}步执行完成！")
        except ValueError:
            print("无效的命令行参数，请输入数字")
            return
    else:
        # 交互式运行
        process_center.run_interactive()


if __name__ == "__main__":
    main()