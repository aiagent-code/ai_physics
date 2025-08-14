# -*- coding: utf-8 -*-
"""
模型端口面板 - 独立的模型端口模式UI组件
Function: 提供模型端口模式的用户界面
Purpose:
- 模型路径配置界面
- 溶液类型名称配置
- 多次测量功能
- 结果查看和可视化
- 独立于数据集模式的完整功能
UI Components:
- 模型配置区域
- 溶液类型配置
- 测量控制按钮
- 结果显示区域
- 状态信息显示
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from experiment.model_processor import ModelProcessor
from experiment.model_port_manager import ModelPortManager
from .message_panel import message_panel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModelPortPanel(ttk.LabelFrame):
    """模型端口面板 - 独立的模型端口模式UI"""
    
    def __init__(self, parent, spectrometer_panel):
        super().__init__(parent, text="模型端口模式", padding=10)
        self.spectrometer_panel = spectrometer_panel
        self.data_save_function = None  # 将由外部设置
        
        # 初始化组件
        self.model_processor = ModelProcessor()
        self.model_port_manager = ModelPortManager(self.model_processor)
        
        self._build_panel()
        self._refresh_components()  # 初始化时显示虚拟模式状态
        
    def _config_panel(self, data_save_function):
        """配置数据保存功能引用"""
        self.data_save_function = data_save_function
        
    def _build_panel(self):
        """构建UI面板"""
        # 模型配置区域
        self._build_model_config()
        
        # 溶液类型配置
        self._build_solution_config()
        
        # 测量控制区域
        self._build_measurement_control()
        
        # 结果显示区域
        self._build_results_display()
        
        # 状态信息
        self._build_status_display()
        
    def _build_model_config(self):
        """构建模型配置区域"""
        model_frame = ttk.LabelFrame(self, text="模型配置", padding=5)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 模型路径选择
        path_frame = ttk.Frame(model_frame)
        path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame, text="模型路径:").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar()
        model_path_entry = ttk.Entry(path_frame, textvariable=self.model_path_var, width=30)
        model_path_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        ttk.Button(path_frame, text="浏览", command=self.browse_model_path).pack(side=tk.RIGHT)
        
        # 模型控制按钮
        control_frame = ttk.Frame(model_frame)
        control_frame.pack(fill=tk.X, pady=2)
        self.load_model_btn = ttk.Button(control_frame, text="加载模型", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.unload_model_btn = ttk.Button(control_frame, text="卸载模型", command=self.unload_model, state='disabled')
        self.unload_model_btn.pack(side=tk.LEFT)
        
        # 模型状态显示
        self.model_status_var = tk.StringVar(value="未加载模型")
        ttk.Label(model_frame, textvariable=self.model_status_var).pack(anchor=tk.W, pady=2)
        
    def _build_solution_config(self):
        """构建溶液类型配置区域"""
        solution_frame = ttk.LabelFrame(self, text="溶液类型配置", padding=5)
        solution_frame.pack(fill=tk.X, pady=(0, 10))
        
        config_frame = ttk.Frame(solution_frame)
        config_frame.pack(fill=tk.X, pady=2)
        ttk.Label(config_frame, text="溶液类型:").pack(side=tk.LEFT)
        self.solution_type_var = tk.StringVar(value="丙三醇")
        solution_entry = ttk.Entry(config_frame, textvariable=self.solution_type_var, width=15)
        solution_entry.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(config_frame, text="设置", command=self.set_solution_type).pack(side=tk.RIGHT)
        
    def _build_measurement_control(self):
        """构建测量控制区域"""
        measure_frame = ttk.LabelFrame(self, text="测量控制", padding=5)
        measure_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 测量按钮行
        button_frame = ttk.Frame(measure_frame)
        button_frame.pack(fill=tk.X, pady=2)
        
        self.measure_btn = ttk.Button(button_frame, text="测量", command=self.perform_measurement, state='normal')
        self.measure_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_btn = ttk.Button(button_frame, text="清空记录", command=self.clear_measurements)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.export_btn = ttk.Button(button_frame, text="导出结果", command=self.export_results)
        self.export_btn.pack(side=tk.LEFT)
        
        # 查看结果按钮
        self.view_results_btn = ttk.Button(measure_frame, text="查看结果图表", command=self.view_results, state='disabled')
        self.view_results_btn.pack(fill=tk.X, pady=(5, 0))
        
    def _build_results_display(self):
        """构建结果显示区域"""
        results_frame = ttk.LabelFrame(self, text="测量结果", padding=5)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 最新结果显示
        latest_frame = ttk.Frame(results_frame)
        latest_frame.pack(fill=tk.X, pady=2)
        ttk.Label(latest_frame, text="最新结果:").pack(side=tk.LEFT)
        self.latest_result_var = tk.StringVar(value="--")
        result_entry = ttk.Entry(latest_frame, textvariable=self.latest_result_var, state='readonly', width=20)
        result_entry.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Label(latest_frame, text="%").pack(side=tk.LEFT)
        
        # 测量统计
        stats_frame = ttk.Frame(results_frame)
        stats_frame.pack(fill=tk.X, pady=2)
        self.measurement_stats_var = tk.StringVar(value="测量次数: 0")
        ttk.Label(stats_frame, textvariable=self.measurement_stats_var).pack(anchor=tk.W)
        
    def _build_status_display(self):
        """构建状态显示区域"""
        status_frame = ttk.LabelFrame(self, text="状态信息", padding=5)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
    def browse_model_path(self):
        """浏览模型文件路径"""
        filetypes = [
            ('所有支持的模型', '*.pkl;*.h5;*.hdf5;*.joblib'),
            ('Pickle文件', '*.pkl'),
            ('Keras模型', '*.h5;*.hdf5'),
            ('Joblib文件', '*.joblib'),
            ('所有文件', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=filetypes
        )
        
        if filename:
            self.model_path_var.set(filename)
            
    def load_model(self):
        """加载神经网络模型"""
        model_path = self.model_path_var.get().strip()
        if not model_path:
            message_panel.show_auto_close_message(
                self, "请先选择模型文件路径", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        try:
            self.status_var.set("正在加载模型...")
            success = self.model_processor.load_model(model_path)
            
            if success:
                self.model_status_var.set(f"已加载: {os.path.basename(model_path)}")
                self.load_model_btn.config(state='disabled')
                self.unload_model_btn.config(state='normal')
                self.status_var.set("模型加载成功")
                message_panel.show_auto_close_message(
                    self, "模型加载成功", "success",
                    refresh_callback=self._refresh_components
                )
            else:
                self.status_var.set("模型加载失败")
                message_panel.show_auto_close_message(
                    self, "模型加载失败，请检查文件格式", "error",
                    refresh_callback=self._refresh_components
                )
                
        except Exception as e:
            self.status_var.set(f"模型加载失败: {str(e)}")
            message_panel.show_auto_close_message(
                self, f"模型加载失败: {str(e)}", "error",
                refresh_callback=self._refresh_components
            )
            
    def unload_model(self):
        """卸载模型"""
        self.model_processor.unload_model()
        self.model_status_var.set("未加载模型")
        self.latest_result_var.set("--")
        self.load_model_btn.config(state='normal')
        self.unload_model_btn.config(state='disabled')
        self.status_var.set("模型已卸载")
        
    def set_solution_type(self):
        """设置溶液类型"""
        solution_type = self.solution_type_var.get().strip()
        if not solution_type:
            message_panel.show_auto_close_message(
                self, "溶液类型不能为空", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        self.model_port_manager.set_solution_type(solution_type)
        self.status_var.set(f"溶液类型已设置为: {solution_type}")
        message_panel.show_auto_close_message(
            self, f"溶液类型已设置为: {solution_type}", "success",
            refresh_callback=self._refresh_components
        )
        
    def perform_measurement(self):
        """执行测量（需要加载模型）"""
            
        if not self.spectrometer_panel or not hasattr(self.spectrometer_panel, 'spectrometer') or not self.spectrometer_panel.spectrometer:
            message_panel.show_auto_close_message(
                self, "请先连接光谱仪设备", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        if not self.model_processor.is_loaded:
            message_panel.show_auto_close_message(
                self, "请先加载模型文件", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        if not self.data_save_function:
            message_panel.show_auto_close_message(
                self, "数据保存功能未配置", "error",
                refresh_callback=self._refresh_components
            )
            return
            
        try:
            self.status_var.set("正在测量...")
            
            # 获取处理后的光谱数据
            x_data, spectrum, display_mode = self.data_save_function.get_processed_spectrum_data()
            
            if spectrum is None or x_data is None:
                raise ValueError("无法获取光谱数据")
                
            # 添加测量记录
            measurement_record = self.model_port_manager.add_measurement(spectrum, x_data, display_mode)
            
            # 更新UI显示
            self.latest_result_var.set(f"{measurement_record['predicted_concentration']:.1f}")
            self.measurement_stats_var.set(f"测量次数: {self.model_port_manager.get_measurement_count()}")
            
            # 启用查看结果按钮
            self.view_results_btn.config(state='normal')
            
            self.status_var.set(f"测量完成: {measurement_record['predicted_concentration']:.1f}%")
            message_panel.show_auto_close_message(
                self, f"测量完成: {measurement_record['solution_type']} {measurement_record['predicted_concentration']:.1f}%", "success",
                refresh_callback=self._refresh_components
            )
            
        except Exception as e:
            self.status_var.set(f"测量失败: {str(e)}")
            message_panel.show_auto_close_message(
                self, f"测量失败: {str(e)}", "error",
                refresh_callback=self._refresh_components
            )
            
    def clear_measurements(self):
        """清空测量记录"""
        if self.model_port_manager.get_measurement_count() == 0:
            message_panel.show_auto_close_message(
                self, "没有测量记录可清空", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        result = messagebox.askyesno("确认清空", "确定要清空所有测量记录吗？")
        if result:
            self.model_port_manager.clear_measurements()
            self.latest_result_var.set("--")
            self.measurement_stats_var.set("测量次数: 0")
            self.view_results_btn.config(state='disabled')
            self.status_var.set("测量记录已清空")
            message_panel.show_auto_close_message(
                self, "测量记录已清空", "success",
                refresh_callback=self._refresh_components
            )
            
    def view_results(self):
        """查看结果图表"""
        if self.model_port_manager.get_measurement_count() == 0:
            message_panel.show_auto_close_message(
                self, "没有测量数据可查看", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        try:
            # 创建结果图表窗口
            self._show_results_window()
            
        except Exception as e:
            message_panel.show_auto_close_message(
                self, f"显示结果图表失败: {str(e)}", "error",
                refresh_callback=self._refresh_components
            )
            
    def _show_results_window(self):
        """显示结果图表窗口"""
        # 创建新窗口
        results_window = tk.Toplevel(self)
        results_window.title(f"{self.model_port_manager.get_solution_type()}溶液测量结果")
        results_window.geometry("1000x800")
        
        # 创建图表
        fig = self.model_port_manager.create_results_plot()
        
        # 嵌入到窗口中
        canvas = FigureCanvasTkAgg(fig, results_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加关闭按钮
        button_frame = ttk.Frame(results_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(button_frame, text="关闭", command=results_window.destroy).pack(side=tk.RIGHT)
        
    def export_results(self):
        """导出结果"""
        if self.model_port_manager.get_measurement_count() == 0:
            message_panel.show_auto_close_message(
                self, "没有测量数据可导出", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        filename = filedialog.asksaveasfilename(
            title="导出测量结果",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                success = self.model_port_manager.export_results(filename)
                if success:
                    message_panel.show_auto_close_message(
                        self, f"结果已导出到: {filename}", "success",
                        refresh_callback=self._refresh_components
                    )
                else:
                    message_panel.show_auto_close_message(
                        self, "导出失败", "error",
                        refresh_callback=self._refresh_components
                    )
            except Exception as e:
                message_panel.show_auto_close_message(
                    self, f"导出失败: {str(e)}", "error",
                    refresh_callback=self._refresh_components
                )
                
    def _refresh_components(self):
        """刷新组件状态"""
        # 更新模型状态显示
        if self.model_processor.is_loaded:
            model_name = os.path.basename(self.model_processor.model_path) if self.model_processor.model_path else "未知模型"
            self.model_status_var.set(f"已加载: {model_name}")
        else:
            self.model_status_var.set("未加载模型")
            
        # 更新溶液类型显示
        self.solution_type_var.set(self.model_port_manager.get_solution_type())
        
        # 更新测量统计
        count = self.model_port_manager.get_measurement_count()
        self.measurement_stats_var.set(f"测量次数: {count}")
        
        # 更新按钮状态
        if count > 0:
            self.view_results_btn.config(state='normal')
        else:
            self.view_results_btn.config(state='disabled')
            
        # 更新状态提示
        if not self.model_processor.is_loaded and self.model_port_manager.get_measurement_count() == 0:
            self.status_var.set("请先加载模型文件")
            
    def get_manager_status(self) -> dict:
        """获取管理器状态信息"""
        return {
            'model_loaded': self.model_processor.is_loaded,
            'solution_type': self.model_port_manager.get_solution_type(),
            'measurement_count': self.model_port_manager.get_measurement_count(),
            'summary': self.model_port_manager.get_measurement_summary()
        }