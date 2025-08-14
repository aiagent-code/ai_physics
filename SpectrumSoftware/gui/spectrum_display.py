"""
SpectrumDisplay Class Description:
Function: Spectrum data display and visualization component
Purpose: 
- Receives spectrum data from spectrometer_panel for real-time display
- Processes and displays spectra based on config_panel settings (display mode, excitation wavelength, raman direction)
- Provides zoom control, baseline removal and other interactive features
- Passes processed data to data_save_panel for storage
Stored Data:
- Figure and axis objects (fig, ax, canvas)
- Display control variables (show_raw_var, show_sub_dark_var, show_baseline_var)
- Zoom and scaling states (auto_scale_var, lock_axis, fixed_xlim, fixed_ylim, saved_xlim, saved_ylim)
- Component references (config_panel, spectrometer_panel, data_save_panel)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from processor.spectrum_processor import SpectrumProcessor
import threading
import numpy as np

# Display mode and raman direction mapping - unified to English


class SpectrumDisplay(ttk.LabelFrame):
    def __init__(self, parent, config_panel=None):
        super().__init__(parent, text="光谱显示", padding=10)
        from config import GUI_SETTINGS, DISPLAY_SETTINGS
        self.fig = Figure(figsize=GUI_SETTINGS['plot_figsize'], dpi=GUI_SETTINGS['plot_dpi'])
        # 显示模式相关变量 - 从config_panel获取
        self.config_panel = config_panel
        # 显示模式相关变量 - 从config_panel获取
        self.config_panel = config_panel
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        # 绑定缩放事件
        self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        
        self.show_raw_var = tk.BooleanVar(value=True)  # 显示原始光谱
        self.show_sub_dark_var = tk.BooleanVar(value=True)  # 显示减暗光谱
        self.show_baseline_var = tk.BooleanVar(value=False)  # 显示基线
        self.auto_scale_var = tk.StringVar(value="auto")  # 默认自动调节
        self.lock_axis = False  # 固定缩放锁
        self.fixed_xlim = None
        self.fixed_ylim = None
        self.saved_xlim = None  # 保存模式的坐标范围
        self.saved_ylim = None
        self.initial_xlim = None
        self.initial_ylim = None
        # 数据源引用
        self.spectrometer_panel = None
        self.data_save_panel = None
        
        self._build_toolbar()
        
    def _config_panel(self,data_save_panel,spectrometer_panel):
        self.data_save_panel = data_save_panel
        self.spectrometer_panel = spectrometer_panel
    
    def get_display_mode(self):
        """获取当前显示模式 (wavelength/raman)"""
        return self.config_panel.display_mode_var.get() if self.config_panel else 'wavelength'
    
    def get_excitation_wavelength(self):
        """获取当前激发光波长"""
        return float(self.config_panel.excitation_wavelength_var.get()) if self.config_panel else 785.0
    
    def get_raman_direction(self):
        """获取当前拉曼方向 (positive/negative)"""
        return self.config_panel.raman_direction_var.get() if self.config_panel else 'positive'
        
    def _build_toolbar(self):
        from config import DISPLAY_SETTINGS
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill=tk.X, pady=(0, 10))
        zoom_frame = ttk.LabelFrame(toolbar_frame, text="缩放控制", padding=5)
        zoom_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(zoom_frame, text="自动缩放", command=self.auto_scale).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="重置缩放", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="返回初始", command=self.restore_initial_zoom).pack(side=tk.LEFT, padx=2)
        axis_frame = ttk.LabelFrame(toolbar_frame, text="坐标轴模式", padding=5)
        axis_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        ttk.Radiobutton(axis_frame, text="自动模式", variable=self.auto_scale_var, value="auto", command=self.toggle_axis_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(axis_frame, text="固定模式", variable=self.auto_scale_var, value="fixed", command=self.toggle_axis_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(axis_frame, text="保存模式", variable=self.auto_scale_var, value="saved", command=self.toggle_axis_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(axis_frame, text="重置模式", variable=self.auto_scale_var, value="reset", command=self.toggle_axis_mode).pack(side=tk.LEFT, padx=2)
        display_options_frame = ttk.LabelFrame(toolbar_frame, text="显示选项", padding=5)
        display_options_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        ttk.Checkbutton(display_options_frame, text="显示原始光谱", variable=self.show_raw_var, command=self.update_plot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(display_options_frame, text="显示减暗光谱", variable=self.show_sub_dark_var, command=self.update_plot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(display_options_frame, text="显示基线", variable=self.show_baseline_var, command=self.update_plot).pack(side=tk.LEFT, padx=2)
        ttk.Button(display_options_frame, text="基线去除", command=self.remove_baseline).pack(side=tk.LEFT, padx=2)

    def toggle_axis_mode(self):
        mode = self.auto_scale_var.get()
        if mode == "auto":
            self.lock_axis = False
            self.auto_scale()
        elif mode == "fixed":
            self.lock_axis = True
            # 保存当前范围作为固定模式的参考点
            if self.ax.get_xlim() != (0, 1):
                self.fixed_xlim = self.ax.get_xlim()
                self.fixed_ylim = self.ax.get_ylim()
        elif mode == "saved":
            # 保存当前视图范围
            if self.ax.get_xlim() != (0, 1):
                self.saved_xlim = self.ax.get_xlim()
                self.saved_ylim = self.ax.get_ylim()
            self.lock_axis = True
        elif mode == "reset":
            self.lock_axis = False
            # 优先使用保存的范围，如果没有则使用初始范围
            if self.saved_xlim and self.saved_ylim:
                self.ax.set_xlim(self.saved_xlim)
                self.ax.set_ylim(self.saved_ylim)
            elif self.initial_xlim and self.initial_ylim:
                self.ax.set_xlim(self.initial_xlim)
                self.ax.set_ylim(self.initial_ylim)
            self.canvas.draw()

    def auto_scale(self):
        self.lock_axis = False
        self.ax.autoscale()
        self.canvas.draw()

    def reset_zoom(self):
        self.lock_axis = False
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)
        self.canvas.draw()

    def restore_initial_zoom(self):
        self.lock_axis = False
        if self.initial_xlim and self.initial_ylim:
            self.ax.set_xlim(self.initial_xlim)
            self.ax.set_ylim(self.initial_ylim)
            self.canvas.draw()

    def _on_xlim_changed(self, event_ax):
        """X轴范围变化事件处理"""
        if self.auto_scale_var.get() == "fixed" and self.ax.get_xlim() != (0, 1):
            self.fixed_xlim = self.ax.get_xlim()

    def _on_ylim_changed(self, event_ax):
        """Y轴范围变化事件处理"""
        if self.auto_scale_var.get() == "fixed" and self.ax.get_ylim() != (0, 1):
            self.fixed_ylim = self.ax.get_ylim()

    def save_current_view(self):
        """保存当前视图范围"""
        if self.ax.get_xlim() != (0, 1):
            self.fixed_xlim = self.ax.get_xlim()
            self.fixed_ylim = self.ax.get_ylim()

    def remove_baseline(self):
        # 对当前显示的减暗光谱做基线去除
        if self.spectrometer_panel and hasattr(self.spectrometer_panel, 'sub_dark_spectrum'):
            sub_dark_spectrum = self.spectrometer_panel.sub_dark_spectrum
            if sub_dark_spectrum is not None:
                y, baseline = SpectrumProcessor.baseline_remove_asls(sub_dark_spectrum)
                self.spectrometer_panel.sub_dark_spectrum = y
                self.spectrometer_panel.baseline = baseline
                self.show_baseline_var.set(True)
                self.update_plot()
        else:
            print("[remove_baseline] 没有减暗光谱可进行基线去除")



    def update_plot(self, spectrum=None, wave_lengths=None):
        """线程安全的更新绘图方法"""
            # # 检查是否在主线程中
            # if threading.current_thread() != threading.main_thread():
            #     # 如果不在主线程，使用after方法调度到主线程
            #     self.after(0, lambda: self.update_plot(spectrum, wave_lengths))
            #     return
        
        # 原有的update_plot逻辑
        from config import DISPLAY_SETTINGS
        try:
            # 保存当前视图范围（如果在固定模式下）
            mode = self.auto_scale_var.get()
            if self.lock_axis and mode in ["fixed", "saved"]:
                current_xlim = self.ax.get_xlim()
                current_ylim = self.ax.get_ylim()
            
            self.ax.clear()
            if spectrum is None:
                raise ValueError("No spectrum data provided")
            
            # 使用getter方法获取当前显示配置
            display_mode = self.get_display_mode()
            excitation_wavelength = self.get_excitation_wavelength()
            raman_direction = self.get_raman_direction()

            # 数据处理 - 使用最新配置处理数据
            x_data, processed_spectrum, sub_dark_spectrum, baseline = self._process_spectrum_data(spectrum,wave_lengths,display_mode,excitation_wavelength,raman_direction)
            '''print("当前数据:",x_data[:5])'''
            # 更新数据到DataSavePanel
            if self.data_save_panel:
                self.data_save_panel.update_processed_data(
                    x_data, processed_spectrum, None,  # dark_spectrum 在 _process_spectrum_data 中处理
                    sub_dark_spectrum, baseline, display_mode
                )
            
            # 绘制
            if self.show_raw_var.get() and processed_spectrum is not None and x_data is not None and len(processed_spectrum) == len(x_data):
                self.ax.plot(x_data, processed_spectrum, color=DISPLAY_SETTINGS['colors']['raw_data'], label='原始光谱', linewidth=DISPLAY_SETTINGS['line_width'], alpha=DISPLAY_SETTINGS['alpha'])
            if self.show_sub_dark_var.get() and sub_dark_spectrum is not None and x_data is not None and len(sub_dark_spectrum) == len(x_data):
                self.ax.plot(x_data, sub_dark_spectrum, color=DISPLAY_SETTINGS['colors']['current_spectrum'], label='减暗光谱', linewidth=DISPLAY_SETTINGS['line_width'], alpha=DISPLAY_SETTINGS['alpha'])
            if self.show_baseline_var.get() and baseline is not None and x_data is not None and len(baseline) == len(x_data):
                self.ax.plot(x_data, baseline, color='orange', label='基线', linewidth=1, alpha=0.8)
            
            # 设置标签 - 根据当前显示模式设置
            x_label = '拉曼位移 (cm^-1)' if display_mode == 'raman' else '波长 (nm)'
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel('absorbance')  # 修改：从'强度'改为'absorbance'
            self.ax.legend()
            self.ax.grid(DISPLAY_SETTINGS['show_grid'], alpha=0.3)
            
            # 保存初始范围（第一次绘图时）
            if self.initial_xlim is None and self.ax.get_xlim() != (0, 1):
                self.initial_xlim = self.ax.get_xlim()
                self.initial_ylim = self.ax.get_ylim()
            
            # 坐标轴模式控制
            if self.lock_axis:
                if mode == "fixed" and 'current_xlim' in locals():
                    # 固定模式：恢复到之前保存的视图范围
                    self.ax.set_xlim(current_xlim)
                    self.ax.set_ylim(current_ylim)
                elif mode == "saved" and self.saved_xlim and self.saved_ylim:
                    # 保存模式：使用保存的范围
                    self.ax.set_xlim(self.saved_xlim)
                    self.ax.set_ylim(self.saved_ylim)
            elif mode == "auto":
                # 自动模式：启用自动缩放
                self.ax.autoscale()
            self.canvas.draw()
        except Exception as e:
            print(f"[update_plot] Exception: {e}")
            import traceback
            traceback.print_exc()

    def _process_spectrum_data(self, spectrum,wave_lengths,display_mode,excitation_wavelength,raman_direction):
        """处理光谱数据 - 使用统一的英文配置"""
        # 获取原始数据
        current_spectrum = np.array(spectrum) if spectrum is not None else None  # 确保转换为numpy数组
        if current_spectrum is None or wave_lengths is None:
            return None, None, None, None
        #print("有数据")

        if display_mode == 'raman':
            # 使用激发光波长计算拉曼位移
            raman_shift = SpectrumProcessor.wavelength_to_raman_shift(wave_lengths, excitation_wavelength)
            if raman_shift is not None:
                # 使用拉曼方向设置掩码
                mask = raman_shift > 0 if raman_direction == 'positive' else raman_shift < 0
                filtered_x = raman_shift[mask]
                filtered_spectrum = current_spectrum[mask]  # 现在可以正常使用布尔掩码索引
                
                # 获取暗光谱处理
                dark_spectrum = None
                sub_dark_spectrum = filtered_spectrum
                baseline = None
                
                if self.data_save_panel:
                    dark_data = self.data_save_panel.get_selected_dark(filtered_x, 'raman', raman_direction)
                    if dark_data is not None and len(dark_data) == len(filtered_spectrum):
                        dark_spectrum = dark_data
                        sub_dark_spectrum = SpectrumProcessor.dark_correction(filtered_spectrum, dark_spectrum)
                
                return filtered_x, filtered_spectrum, sub_dark_spectrum, baseline
            else:
                return None, None, None, None
        else:
            # 波长模式处理
            x_data = np.array(wave_lengths)
            
            # 获取暗光谱处理
            dark_spectrum = None
            sub_dark_spectrum = current_spectrum
            baseline = None
            
            if self.data_save_panel:
                dark_data = self.data_save_panel.get_selected_dark(x_data, 'wavelength', None)
                if dark_data is not None and len(dark_data) == len(current_spectrum):
                    dark_spectrum = dark_data
                    sub_dark_spectrum = SpectrumProcessor.dark_correction(current_spectrum, dark_spectrum)
            
            return x_data, current_spectrum, sub_dark_spectrum, baseline
