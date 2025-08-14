"""
SpectrometerPanel Class Description:
Function: Spectrometer device control and data acquisition
Purpose:
- Manages spectrometer device connection and disconnection
- Controls acquisition parameters (integration time, scan count)
- Runs real-time spectrum acquisition loop
- Provides mock device functionality for testing
Stored Data:
- API and spectrometer device references (api, spectrometer)
- Acquisition state flags (is_acquiring, _auto_refresh)
- Wavelength data (wavelengths)
- Acquisition parameters (integration_time_var, scans_var)
- Callback functions for status and data updates
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from oceandirect.OceanDirectAPI import OceanDirectAPI, OceanDirectError, MockSpectrometer
from .message_panel import message_panel
import traceback
import logging

class SpectrometerPanel(ttk.LabelFrame):
    def __init__(self, parent, spectrum_display=None, on_status=None, on_device_info=None):
        super().__init__(parent, text="设备与采集控制", padding=10)
        # 使用test_device.py中验证有效的初始化方式
        self.api = OceanDirectAPI()
        self.spectrometer = None
        self.device_ids = []
        self.spectrum_display = spectrum_display
        self.is_acquiring = False
        self._auto_refresh = False
        self.on_status = on_status
        self.on_device_info = on_device_info
        
        # 只存储波长数据，不存储光谱数据
        self.wavelengths = None
        
        self._build_panel()
        self.i=0

    def _build_panel(self):
        # 设备信息
        device_frame = ttk.LabelFrame(self, text="设备信息", padding=5)
        device_frame.pack(fill=tk.X, pady=(0, 10))
        self.device_info_text = tk.Text(device_frame, height=6, width=30)
        self.device_info_text.pack(fill=tk.X)

        # 设备控制
        device_control_frame = ttk.LabelFrame(self, text="设备控制", padding=5)
        device_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 设备控制按钮
        button_frame = ttk.Frame(device_control_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="查找设备", command=self.find_devices).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="连接设备", command=self.connect_device).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="断开设备", command=self.disconnect_device).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="模拟设备", command=self.enable_mock_device).pack(side=tk.LEFT)
        
        # 采集参数
        params_frame = ttk.LabelFrame(self, text="采集参数", padding=5)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 积分时间
        ttk.Label(params_frame, text="积分时间(ms):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.integration_time_var = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.integration_time_var, width=10).grid(row=0, column=1, padx=(0, 10))
        
        # 扫描次数
        ttk.Label(params_frame, text="扫描次数:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.scans_var = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.scans_var, width=10).grid(row=0, column=3, padx=(0, 10))
        
        ttk.Button(params_frame, text="应用参数", command=self.apply_parameters).grid(row=0, column=4)
        
        # 实时采集控制
        acquisition_frame = ttk.LabelFrame(self, text="实时采集", padding=5)
        acquisition_frame.pack(fill=tk.X)
        
        ttk.Button(acquisition_frame, text="开始采集", command=lambda: self.start_real_time_acquisition()).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(acquisition_frame, text="停止采集", command=self.stop_acquisition).pack(side=tk.LEFT)

    def update_status(self, msg):
        if self.on_status:
            self.on_status(msg)

    def update_device_info(self, info):
        self.device_info_text.delete(1.0, tk.END)
        self.device_info_text.insert(tk.END, info)
        if self.on_device_info:
            self.on_device_info(info)

    def find_devices(self):
        """使用test_device.py中验证有效的设备查找方法"""
        try:
            self.update_status("正在查找设备...")
            # 使用test_device.py中验证有效的方法
            num_devices = self.api.find_devices()
            self.update_device_info(f"找到 {num_devices} 个设备")
            
            if num_devices > 0:
                # 获取设备ID列表
                self.device_ids = self.api.get_device_ids()
                device_info = f"找到 {num_devices} 个设备\n设备ID: {self.device_ids}"
                self.update_device_info(device_info)
                self.update_status("设备查找完成")
            else:
                self.update_device_info("未找到设备")
                self.update_status("未找到设备")
                
        except OceanDirectError as e:
            error_msg = f"查找设备失败: {str(e)}"
            self.update_status(error_msg)
            self.update_device_info(error_msg)
            logger = logging.getLogger(__name__)
            logger.error(error_msg)
            traceback.print_exc()
            message_panel.show_auto_close_message(
                self, error_msg, "error",
                refresh_callback=self._refresh_specific_components
            )
        except Exception as e:
            error_msg = f"查找设备失败: {str(e)}"
            self.update_status(error_msg)
            self.update_device_info(error_msg)
            traceback.print_exc()
            message_panel.show_auto_close_message(
                self, error_msg, "error",
                refresh_callback=self._refresh_specific_components
            )
    
    def _refresh_specific_components(self):
        """刷新特定组件"""
        # 更新设备信息显示
        if hasattr(self, 'device_info_text'):
            try:
                # 触发设备信息更新
                pass
            except:
                pass
    
    def connect_device(self):
        """使用test_device.py中验证有效的设备连接方法"""
        try:
            if self.spectrometer is not None:
                self._show_auto_close_message("设备已连接", "info")
                return
                
            # 如果没有设备ID，先查找设备
            if not self.device_ids:
                self.find_devices()
                
            if not self.device_ids:
                self._show_auto_close_message("警告: 没有可用的设备", "warning")
                return
                
            # 使用test_device.py中验证有效的连接方法
            first_dev_id = self.device_ids[0]
            self.spectrometer = self.api.open_device(first_dev_id)
            
            # 获取设备信息
            serial_number = self.api.get_serial_number(first_dev_id)
            model = self.spectrometer.get_model()
            
            device_info = f"设备已连接\n设备ID: {first_dev_id}\n序列号: {serial_number}\n型号: {model}"
            self.update_device_info(device_info)
            self.update_status("设备连接成功")
            
            # 获取波长数据
            self.wavelengths = self.spectrometer.get_wavelengths()
            
            # 连接成功后自动开始实时采集
            if self.spectrum_display:
                self.start_real_time_acquisition(self.spectrum_display.update_plot)
                
        except OceanDirectError as e:
            error_msg = f"连接设备失败: {str(e)}"
            self.update_status(error_msg)
            self.update_device_info(error_msg)
            logger = logging.getLogger(__name__)
            logger.error(error_msg)
            traceback.print_exc()
            self._show_auto_close_message(error_msg, "error")
        except Exception as e:
            error_msg = f"连接设备失败: {str(e)}"
            self.update_status(error_msg)
            self.update_device_info(error_msg)
            traceback.print_exc()
            self._show_auto_close_message(error_msg, "error")

    def disconnect_device(self):
        """使用test_device.py中验证有效的设备断开方法"""
        try:
            if self.spectrometer:
                self.stop_acquisition()
                # 使用test_device.py中验证有效的断开方法
                self.api.close_all_devices()
                self.spectrometer = None
                self.device_ids = []
                self.update_status("设备已断开")
                self.update_device_info("设备已断开")
            else:
                self.update_status("没有连接的设备")
        except Exception as e:
            error_msg = f"断开设备失败: {str(e)}"
            self.update_status(error_msg)
            traceback.print_exc()

    def apply_parameters(self):
        if not self.spectrometer:
            self._show_auto_close_message("警告: 请先连接设备", "warning")
            return
        try:
            integration_time = int(self.integration_time_var.get())
            scans = int(self.scans_var.get())
            self.spectrometer.set_integration_time_micros(integration_time * 1000)
            self.spectrometer.set_scans_to_average(scans)
            self.update_status(f"参数已应用: 积分时间={integration_time}ms, 扫描次数={scans}")
        except ValueError:
            self._show_auto_close_message("参数格式错误", "error")
        except Exception as e:
            self._show_auto_close_message(f"应用参数失败: {str(e)}", "error")

    def start_real_time_acquisition(self, on_update_plot=None):
        if not self.spectrometer:
            self._show_auto_close_message("请先连接设备", "warning")
            return
        if self.is_acquiring:
            return
        self.is_acquiring = True
        self._auto_refresh = True
        self.update_status("开始实时采集")
        # 在新线程中运行采集循环
        threading.Thread(target=self._acquisition_loop, args=(on_update_plot,), daemon=True).start()

    def stop_acquisition(self):
        self._auto_refresh = False
        self.is_acquiring = False
        self.update_status("停止采集")

    def _acquisition_loop(self, on_update_plot):
        while self._auto_refresh and self.spectrometer:
            try:
                spectrum = self.spectrometer.get_formatted_spectrum()
                if spectrum is not None and on_update_plot:  # 修复数组真值判断歧义
                    on_update_plot(spectrum,self.wavelengths)
            except Exception as e:
                self.update_status(f"采集错误: {str(e)}")
            threading.Event().wait(0.1)  # 100ms间隔

    def get_current_spectrum(self):
        if self.spectrometer:
            return self.spectrometer.get_formatted_spectrum()
        return None

    def enable_mock_device(self):
        """启用模拟设备"""
        try:
            if self.spectrometer:
                self.disconnect_device()
            
            self.spectrometer = MockSpectrometer()
            self.wavelengths = self.spectrometer.get_wavelengths()
            
            device_info = "模拟设备已启用\n型号: MockSpectrometer\n序列号: MOCK001"
            self.update_device_info(device_info)
            self.update_status("模拟设备已启用")
            
            # 启用模拟设备后自动开始实时采集
            if self.spectrum_display:
                self.start_real_time_acquisition(self.spectrum_display.update_plot)
                
        except Exception as e:
            error_msg = f"启用模拟设备失败: {str(e)}"
            self.update_status(error_msg)
            self._show_auto_close_message(error_msg, "error")

    def _show_auto_close_message(self, message, msg_type="info", duration=2000):
        """显示自动关闭的消息"""
        message_panel.show_auto_close_message(
            self, message, msg_type,
            refresh_callback=self._refresh_specific_components
        )

    def get_formatted_spectrum(self):
        """获取格式化的光谱数据"""
        if self.spectrometer:
            return self.spectrometer.get_formatted_spectrum()
        return None

    def get_device_status(self):
        """获取设备状态信息"""
        return {
            'connected': self.spectrometer is not None,
            'acquiring': self.is_acquiring,
            'device_count': len(self.device_ids),
            'integration_time': self.integration_time_var.get(),
            'scans': self.scans_var.get()
        }

    def _refresh_interface(self):
        """刷新界面"""
        try:
            self.update_idletasks()
        except:
            pass
