# -*- coding: utf-8 -*-
"""
数据集面板 - 独立的数据集模式UI组件
Function: 提供数据集模式的用户界面
Purpose:
- 数据集配置和管理
- 实验控制功能
- 测量和保存功能
- 独立于模型端口模式的完整功能
UI Components:
- 数据集配置区域
- 文件命名设置
- 实验控制按钮
- 溶液浓度输入
- 测量控制区域
- 状态信息显示
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from experiment.experiment_manager import ExperimentManager
from experiment.dataset_manager import DatasetManager
from experiment.measurement_session import MeasurementSession
from .message_panel import message_panel

class DatasetPanel(ttk.LabelFrame):
    """数据集面板 - 独立的数据集模式UI"""
    
    def __init__(self, parent, spectrometer_panel):
        super().__init__(parent, text="数据集模式", padding=10)
        self.spectrometer_panel = spectrometer_panel
        self.data_save_function = None  # 将由外部设置
        
        # 初始化组件
        self.experiment_manager = ExperimentManager()
        self.dataset_manager = DatasetManager()
        self.measurement_session = MeasurementSession(self.dataset_manager, spectrometer_panel)
        
        # 数据变量
        self.variable_entries = []
        self.variable_names = []
        self.dataset_path = None
        
        self._build_panel()
        self.experiment_manager.load_config()
        
    def _config_panel(self, data_save_function):
        """配置数据保存功能引用"""
        self.data_save_function = data_save_function
        
    def _build_panel(self):
        """构建UI面板"""
        # 数据集配置
        self._build_dataset_config()
        
        # 文件前缀设置
        self._build_file_prefix_config()
        
        # 实验控制
        self._build_experiment_control()
        
        # 溶液浓度输入
        self._build_concentration_inputs()
        
        # 测量控制
        self._build_measurement_control()
        
        # 状态显示
        self._build_status_display()
        
    def _build_dataset_config(self):
        """构建数据集配置区域"""
        dataset_frame = ttk.LabelFrame(self, text="数据集配置", padding=5)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(dataset_frame, text="配置数据集", command=self.configure_dataset).pack(fill=tk.X, pady=2)
        
    def _build_file_prefix_config(self):
        """构建文件前缀配置区域"""
        prefix_frame = ttk.LabelFrame(self, text="文件命名设置", padding=5)
        prefix_frame.pack(fill=tk.X, pady=(0, 10))
        
        prefix_input_frame = ttk.Frame(prefix_frame)
        prefix_input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prefix_input_frame, text="文件前缀:").pack(side=tk.LEFT)
        self.prefix_var = tk.StringVar(value="spectral")
        prefix_entry = ttk.Entry(prefix_input_frame, textvariable=self.prefix_var, width=15)
        prefix_entry.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(prefix_input_frame, text="设置", command=self.set_file_prefix).pack(side=tk.RIGHT)
        
    def _build_experiment_control(self):
        """构建实验控制区域"""
        experiment_frame = ttk.LabelFrame(self, text="实验控制", padding=5)
        experiment_frame.pack(fill=tk.X, pady=(0, 10))
        
        control_frame = ttk.Frame(experiment_frame)
        control_frame.pack(fill=tk.X, pady=2)
        self.start_btn = ttk.Button(control_frame, text="开始实验", command=self.start_experiment)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn = ttk.Button(control_frame, text="停止实验", command=self.stop_experiment, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
    def _build_concentration_inputs(self):
        """构建溶液浓度输入区域"""
        self.concentration_frame = ttk.LabelFrame(self, text="溶液浓度", padding=5)
        self.concentration_frame.pack(fill=tk.X, pady=(0, 10))
        self.concentration_entries = []
        
    def _build_measurement_control(self):
        """构建测量控制区域"""
        measure_frame = ttk.LabelFrame(self, text="测量控制", padding=5)
        measure_frame.pack(fill=tk.X, pady=(0, 10))
        
        measure_control_frame = ttk.Frame(measure_frame)
        measure_control_frame.pack(fill=tk.X, pady=2)
        self.measure_btn = ttk.Button(measure_control_frame, text="测量并保存", command=self.measure_and_save, state='disabled')
        self.measure_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.delete_btn = ttk.Button(measure_control_frame, text="删除最后测量", command=self.delete_last_measurement, state='disabled')
        self.delete_btn.pack(side=tk.LEFT)
        
    def _build_status_display(self):
        """构建状态显示区域"""
        status_frame = ttk.LabelFrame(self, text="状态信息", padding=5)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        self.measurement_count_var = tk.StringVar(value="测量次数: 0")
        ttk.Label(status_frame, textvariable=self.measurement_count_var).pack(anchor=tk.W)
        
    def configure_dataset(self):
        """配置数据集"""
        dialog = DatasetConfigDialog(self)
        self.wait_window(dialog)
        if dialog.result:
            self.variable_names = dialog.variable_names
            self.dataset_path = dialog.dataset_path
            self.dataset_manager.dataset_path = self.dataset_path
            self.measurement_session.set_variables(self.variable_names)
            self._rebuild_concentration_inputs()
            self.status_var.set(f"数据集已配置: {len(self.variable_names)}个变量")
            self._update_ui_state()
            
    def _rebuild_concentration_inputs(self):
        """重建溶液浓度输入界面"""
        for widget in self.concentration_frame.winfo_children():
            widget.destroy()
        self.concentration_entries = []
        
        for i, name in enumerate(self.variable_names):
            frame = ttk.Frame(self.concentration_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{name}:").pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=15)
            entry.pack(side=tk.RIGHT, padx=(5, 0))
            self.concentration_entries.append(entry)
            
    def set_file_prefix(self):
        """设置文件前缀"""
        prefix = self.prefix_var.get().strip()
        if not prefix:
            message_panel.show_auto_close_message(
                self, "文件前缀不能为空", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        self.dataset_manager.set_file_prefix(prefix)
        self.status_var.set(f"文件前缀已设置为: {prefix}")
        
    def start_experiment(self):
        """开始实验"""
        if not self.variable_names:
            message_panel.show_auto_close_message(
                self, "请先配置数据集", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        try:
            self.experiment_manager.start_experiment()
            self.status_var.set("实验进行中")
            self._update_ui_state()
            message_panel.show_auto_close_message(
                self, "实验已开始", "success",
                refresh_callback=self._refresh_components
            )
        except Exception as e:
            message_panel.show_auto_close_message(
                self, f"开始实验失败: {str(e)}", "error",
                refresh_callback=self._refresh_components
            )
    
    def stop_experiment(self):
        """停止实验"""
        try:
            self.experiment_manager.stop_experiment()
            self.status_var.set("实验已停止")
            self._update_ui_state()
            message_panel.show_auto_close_message(
                self, "实验已停止", "success",
                refresh_callback=self._refresh_components
            )
        except Exception as e:
            message_panel.show_auto_close_message(
                self, f"停止实验失败: {str(e)}", "error",
                refresh_callback=self._refresh_components
            )
            
    def measure_and_save(self):
        """测量并保存数据"""
        if not self.variable_names:
            message_panel.show_auto_close_message(
                self, "请先配置数据集", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        # 获取浓度值
        concentrations = []
        for i, entry in enumerate(self.concentration_entries):
            try:
                value = float(entry.get())
                concentrations.append(value)
            except ValueError:
                message_panel.show_auto_close_message(
                    self, f"请输入有效的{self.variable_names[i]}浓度值", "error",
                    refresh_callback=self._refresh_components
                )
                return
        
        try:
            # 设置变量值到measurement_session
            self.measurement_session.set_variable_values(concentrations)
            
            # 执行测量
            result = self.measurement_session.measure_and_save(self.data_save_function)
            if result:
                self.status_var.set(f"测量完成，已保存第{self.dataset_manager.measurement_count}个样本")
                self.measurement_count_var.set(f"测量次数: {self.dataset_manager.measurement_count}")
                self._update_ui_state()
                message_panel.show_auto_close_message(
                    self, "测量并保存成功", "success",
                    refresh_callback=self._refresh_components
                )
            else:
                message_panel.show_auto_close_message(
                    self, "测量失败", "error",
                    refresh_callback=self._refresh_components
                )
        except Exception as e:
            message_panel.show_auto_close_message(
                self, f"测量失败: {str(e)}", "error",
                refresh_callback=self._refresh_components
            )
            
    def delete_last_measurement(self):
        """删除最后一个测量"""
        if self.dataset_manager.measurement_count == 0:
            message_panel.show_auto_close_message(
                self, "没有可删除的测量数据", "warning",
                refresh_callback=self._refresh_components
            )
            return
            
        try:
            if self.dataset_manager.delete_last_measurement():
                self.status_var.set(f"已删除最后一个测量，当前测量次数: {self.dataset_manager.measurement_count}")
                self.measurement_count_var.set(f"测量次数: {self.dataset_manager.measurement_count}")
                self._update_ui_state()
                message_panel.show_auto_close_message(
                    self, "已删除最后一个测量文件", "success",
                    refresh_callback=self._refresh_components
                )
            else:
                message_panel.show_auto_close_message(
                    self, "删除文件失败", "error",
                    refresh_callback=self._refresh_components
                )
        except Exception as e:
            message_panel.show_auto_close_message(
                self, f"删除失败: {str(e)}", "error",
                refresh_callback=self._refresh_components
            )
            
    def _update_ui_state(self):
        """更新UI状态"""
        is_running = self.experiment_manager.is_running
        has_dataset = bool(self.variable_names)
        has_measurements = self.dataset_manager.measurement_count > 0
        
        # 更新按钮状态
        if is_running:
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.measure_btn.config(state='normal' if has_dataset else 'disabled')
            self.delete_btn.config(state='normal' if has_measurements else 'disabled')
        else:
            self.start_btn.config(state='normal' if has_dataset else 'disabled')
            self.stop_btn.config(state='disabled')
            self.measure_btn.config(state='disabled')
            self.delete_btn.config(state='normal' if has_measurements else 'disabled')
            
    def _refresh_components(self):
        """刷新特定组件"""
        # 更新UI状态
        if hasattr(self, '_update_ui_state'):
            try:
                self._update_ui_state()
            except:
                pass
        
        # 更新状态显示
        if hasattr(self, 'experiment_manager'):
            try:
                status = self.experiment_manager.get_experiment_status()
                if hasattr(self, 'measurement_count_var'):
                    self.measurement_count_var.set(f"测量次数: {status['measurement_count']}")
            except:
                pass
                
    def get_manager_status(self) -> dict:
        """获取管理器状态信息"""
        return {
            'experiment_running': self.experiment_manager.is_running,
            'dataset_configured': bool(self.variable_names),
            'measurement_count': self.dataset_manager.measurement_count,
            'variable_names': self.variable_names.copy(),
            'dataset_path': self.dataset_path
        }


class DatasetConfigDialog(tk.Toplevel):
    """数据集配置对话框"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("数据集配置")
        self.geometry("500x400")
        self.transient(parent)
        self.grab_set()
        
        self.result = None
        self.variable_names = []
        self.dataset_path = ""
        
        self._build_dialog()
        
        # 居中显示
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
        
    def _build_dialog(self):
        """构建对话框界面"""
        # 数据集路径选择
        path_frame = ttk.LabelFrame(self, text="数据集路径", padding=10)
        path_frame.pack(fill=tk.X, padx=10, pady=5)
        
        path_input_frame = ttk.Frame(path_frame)
        path_input_frame.pack(fill=tk.X)
        
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(path_input_frame, textvariable=self.path_var, width=40)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(path_input_frame, text="浏览", command=self.browse_path).pack(side=tk.RIGHT)
        
        # 变量配置
        var_frame = ttk.LabelFrame(self, text="变量配置", padding=10)
        var_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 变量数量设置
        count_frame = ttk.Frame(var_frame)
        count_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(count_frame, text="变量数量:").pack(side=tk.LEFT)
        self.var_count_var = tk.StringVar(value="1")
        count_spinbox = ttk.Spinbox(count_frame, from_=1, to=10, textvariable=self.var_count_var, width=5)
        count_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        ttk.Button(count_frame, text="生成", command=self._generate_variable_inputs).pack(side=tk.LEFT)
        
        # 变量输入区域
        self.var_input_frame = ttk.Frame(var_frame)
        self.var_input_frame.pack(fill=tk.BOTH, expand=True)
        
        # 按钮区域
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="确定", command=self.ok).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="取消", command=self.cancel).pack(side=tk.RIGHT)
        
        # 初始生成一个变量输入
        self._generate_variable_inputs()
        
    def _generate_variable_inputs(self):
        """生成变量输入框"""
        # 清空现有输入框
        for widget in self.var_input_frame.winfo_children():
            widget.destroy()
            
        self.variable_entries = []
        
        try:
            count = int(self.var_count_var.get())
        except ValueError:
            count = 1
            
        for i in range(count):
            frame = ttk.Frame(self.var_input_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"变量{i+1}:").pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=20)
            entry.pack(side=tk.LEFT, padx=(5, 5))
            entry.insert(0, f"变量{i+1}")
            
            ttk.Button(frame, text="删除", command=lambda f=frame, e=entry: self._delete_variable(f, e)).pack(side=tk.RIGHT)
            
            self.variable_entries.append(entry)
            
    def _delete_variable(self, frame, entry):
        """删除变量"""
        if len(self.variable_entries) <= 1:
            messagebox.showwarning("警告", "至少需要保留一个变量")
            return
            
        self.variable_entries.remove(entry)
        frame.destroy()
        self._renumber_variables()
        
    def _renumber_variables(self):
        """重新编号变量"""
        for i, entry in enumerate(self.variable_entries):
            current_text = entry.get()
            if current_text.startswith("变量"):
                entry.delete(0, tk.END)
                entry.insert(0, f"变量{i+1}")
                
    def browse_path(self):
        """浏览数据集路径"""
        path = filedialog.askdirectory(title="选择数据集保存路径")
        if path:
            self.path_var.set(path)
            
    def ok(self):
        """确定按钮处理"""
        # 验证路径
        path = self.path_var.get().strip()
        if not path:
            messagebox.showerror("错误", "请选择数据集路径")
            return
            
        # 验证变量名
        variable_names = []
        for entry in self.variable_entries:
            name = entry.get().strip()
            if not name:
                messagebox.showerror("错误", "变量名不能为空")
                return
            if name in variable_names:
                messagebox.showerror("错误", f"变量名'{name}'重复")
                return
            variable_names.append(name)
            
        self.variable_names = variable_names
        self.dataset_path = path
        self.result = True
        self.destroy()
        
    def cancel(self):
        """取消按钮处理"""
        self.result = False
        self.destroy()