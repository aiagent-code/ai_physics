# -*- coding: utf-8 -*-
"""
实验面板 - 统一的实验模式管理面板
Function: 提供实验模式的统一管理界面，支持数据集模式和模型端口模式的切换
Purpose:
- 模式切换管理：在数据集模式和模型端口模式之间切换
- 独立模式管理：每种模式都有独立的UI和功能
- 状态同步：管理两种模式的状态信息
- 数据隔离：确保两种模式的数据和配置相互独立
UI Components:
- 模式选择区域
- 数据集模式面板（DatasetPanel）
- 模型端口模式面板（ModelPortPanel）
- 状态信息显示
"""

import tkinter as tk
from tkinter import ttk
from .dataset_panel import DatasetPanel
from .model_port_panel import ModelPortPanel
from .message_panel import message_panel

class ExperimentPanel(ttk.LabelFrame):
    """实验面板 - 统一的实验模式管理面板"""
    
    def __init__(self, parent, spectrometer_panel):
        super().__init__(parent, text="实验模式", padding=10)
        self.spectrometer_panel = spectrometer_panel
        self.data_save_function = None  # 将由外部设置
        
        # 当前模式
        self.current_mode = "dataset"  # 当前模式：dataset 或 model
        
        # 子面板
        self.dataset_panel = None
        self.model_port_panel = None
        
        self._build_panel()
        
    def _config_panel(self, data_save_function):
        """配置数据保存功能引用"""
        self.data_save_function = data_save_function
        
        # 配置子面板
        if self.dataset_panel:
            self.dataset_panel._config_panel(data_save_function)
        if self.model_port_panel:
            self.model_port_panel._config_panel(data_save_function)
        
    def _build_panel(self):
        """构建UI面板"""
        # 模式选择
        self._build_mode_selection()
        
        # 子面板容器
        self.panel_container = ttk.Frame(self)
        self.panel_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # 创建子面板
        self._create_sub_panels()
        
        # 状态显示
        self._build_status_display()
        
        # 初始显示数据集模式
        self.on_mode_change()
        
    def _build_mode_selection(self):
        """构建模式选择区域"""
        mode_frame = ttk.LabelFrame(self, text="模式选择", padding=5)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="dataset")
        ttk.Radiobutton(mode_frame, text="数据集模式", variable=self.mode_var, value="dataset", command=self.on_mode_change).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(mode_frame, text="模型端口模式", variable=self.mode_var, value="model", command=self.on_mode_change).pack(side=tk.LEFT)
        
    def _create_sub_panels(self):
        """创建子面板"""
        # 创建数据集面板
        self.dataset_panel = DatasetPanel(self.panel_container, self.spectrometer_panel)
        if self.data_save_function:
            self.dataset_panel._config_panel(self.data_save_function)
            
        # 创建模型端口面板
        self.model_port_panel = ModelPortPanel(self.panel_container, self.spectrometer_panel)
        if self.data_save_function:
            self.model_port_panel._config_panel(self.data_save_function)
    
    def _build_status_display(self):
        """构建状态显示区域"""
        self.status_frame = ttk.LabelFrame(self, text="状态信息", padding=5)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        self.mode_status_var = tk.StringVar(value="当前模式: 数据集模式")
        ttk.Label(self.status_frame, textvariable=self.mode_status_var).pack(anchor=tk.W)
        
    def on_mode_change(self):
        """模式切换处理"""
        self.current_mode = self.mode_var.get()
        
        if self.current_mode == "dataset":
            self.dataset_panel.pack(fill=tk.BOTH, expand=True)
            self.model_port_panel.pack_forget()
        else:
            self.model_port_panel.pack(fill=tk.BOTH, expand=True)
            self.dataset_panel.pack_forget()
            
        self._update_status_display()
    
    def _update_status_display(self):
        """更新状态显示"""
        mode_text = "数据集模式" if self.current_mode == "dataset" else "模型端口模式"
        self.mode_status_var.set(f"当前模式: {mode_text}")
        
        # 获取当前模式的状态信息
        if self.current_mode == "dataset" and self.dataset_panel:
            status = self.dataset_panel.get_manager_status()
            if status['experiment_running']:
                self.status_var.set("数据集实验进行中")
            elif status['dataset_configured']:
                self.status_var.set(f"数据集已配置 - {len(status['variable_names'])}个变量")
            else:
                self.status_var.set("数据集模式就绪")
        elif self.current_mode == "model" and self.model_port_panel:
            status = self.model_port_panel.get_manager_status()
            if status['model_loaded']:
                self.status_var.set(f"模型已加载 - {status['measurement_count']}次测量")
            else:
                self.status_var.set("模型端口模式就绪")
        else:
            self.status_var.set("就绪")
    def _refresh_components(self):
        """刷新特定组件"""
        # 更新状态显示
        self._update_status_display()
        
        # 刷新当前活动面板
        if self.current_mode == "dataset" and self.dataset_panel:
            if hasattr(self.dataset_panel, '_refresh_components'):
                self.dataset_panel._refresh_components()
        elif self.current_mode == "model" and self.model_port_panel:
            if hasattr(self.model_port_panel, '_refresh_components'):
                self.model_port_panel._refresh_components()
    
    def get_current_mode_status(self) -> dict:
        """获取当前模式的状态信息"""
        if self.current_mode == "dataset" and self.dataset_panel:
            return {
                'mode': 'dataset',
                'status': self.dataset_panel.get_manager_status()
            }
        elif self.current_mode == "model" and self.model_port_panel:
            return {
                'mode': 'model',
                'status': self.model_port_panel.get_manager_status()
            }
        else:
            return {
                'mode': self.current_mode,
                'status': {}
            }