"""
ConfigPanel Class Description:
Function: Configuration panel for spectrum display settings
Purpose:
- Provides UI controls for display mode selection (wavelength/raman shift)
- Manages excitation wavelength input for raman calculations
- Controls raman direction setting (positive/negative)
- Triggers spectrum display updates when configuration changes
Stored Data:
- Display mode variable (display_mode_var)
- Excitation wavelength variable (excitation_wavelength_var) 
- Raman direction variable (raman_direction_var)
- Reference to spectrum_display component
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from .message_panel import message_panel

class ConfigPanel(ttk.LabelFrame):
    def __init__(self, parent, spectrum_display=None, on_mode_change=None, on_excitation_change=None, on_raman_direction_change=None):
        super().__init__(parent, text="配置面板")
        self.spectrum_display = spectrum_display
        self.display_mode_var = tk.StringVar(value="wavelenth")
        self.excitation_wavelength_var = tk.StringVar(value="785")
        self.raman_direction_var = tk.StringVar(value="positive")
        self._build_panel()

    def _build_panel(self):
        # 显示模式选择
        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill="x", pady=2)
        ttk.Label(mode_frame, text="显示模式:").pack(side="left")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.display_mode_var, 
                                 values=["wavelength", "raman"], state="readonly", width=10)
        mode_combo.pack(side="right")
        mode_combo.bind('<<ComboboxSelected>>', lambda e: self._mode_changed())
        
        # 激发光波长设置
        excitation_frame = ttk.Frame(self)
        excitation_frame.pack(fill="x", pady=2)
        ttk.Label(excitation_frame, text="激发光波长(nm):").pack(side="left")
        excitation_entry = ttk.Entry(excitation_frame, textvariable=self.excitation_wavelength_var, width=10)
        excitation_entry.pack(side="right")
        excitation_entry.bind('<FocusOut>', self._excitation_changed)
        excitation_entry.bind('<Return>', self._excitation_changed)
        
        # 拉曼位移方向
        raman_frame = ttk.Frame(self)
        raman_frame.pack(fill="x", pady=2)
        ttk.Label(raman_frame, text="拉曼方向:").pack(side="left")
        raman_combo = ttk.Combobox(raman_frame, textvariable=self.raman_direction_var,
                                  values=["positive", "negetive"], state="readonly", width=10)
        raman_combo.pack(side="right")
        raman_combo.bind('<<ComboboxSelected>>', lambda e: self._raman_direction_changed())

    def _mode_changed(self):
        """显示模式改变时更新光谱显示"""
        print(f"显示模式已切换为: {self.display_mode_var.get()}")
        self._update_spectrum_display()

    def _excitation_changed(self, event=None):
        try:
            value = float(self.excitation_wavelength_var.get())
            print(f"激发光波长已设置为: {value} nm")
            self._update_spectrum_display()
        except ValueError:
            message_panel.show_auto_close_message(
                self, "输入错误: 请输入有效的数字", "error",
                refresh_callback=self._refresh_specific_components
            )

    def _raman_direction_changed(self):
        """拉曼方向改变时更新光谱显示"""
        print(f"拉曼方向已切换为: {self.raman_direction_var.get()}")
        self._update_spectrum_display()
        
    def _update_spectrum_display(self):
        """更新光谱显示"""
        if self.spectrum_display and hasattr(self.spectrum_display, 'spectrometer_panel'):
            spectrometer_panel = self.spectrum_display.spectrometer_panel
            if (spectrometer_panel and 
                hasattr(spectrometer_panel, 'spectrometer') and 
                spectrometer_panel.spectrometer):
                try:
                    # 获取当前光谱数据
                    spectrum = spectrometer_panel.spectrometer.get_formatted_spectrum()
                    wavelengths = spectrometer_panel.spectrometer.get_wavelengths()
                    if spectrum is not None and wavelengths is not None:
                        self.spectrum_display.update_plot(spectrum, wavelengths)
                except Exception as e:
                    print(f"更新光谱显示失败: {e}")

    def _refresh_specific_components(self):
        """刷新特定组件"""
        # 重置输入框为默认值
        try:
            float(self.excitation_wavelength_var.get())
        except ValueError:
            self.excitation_wavelength_var.set("785")