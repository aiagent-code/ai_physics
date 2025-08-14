from gui.base_gui import BaseGUI
from gui.spectrometer_panel import SpectrometerPanel
from gui.spectrum_display import SpectrumDisplay
from gui.data_save_panel import DataSavePanel
from gui.config_panel import ConfigPanel
from gui.experiment_panel import ExperimentPanel
from processor.spectrum_processor import SpectrumProcessor
from oceandirect.OceanDirectAPI import OceanDirectAPI
from config import GUI_SETTINGS
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
import sys
import argparse
import socket
import threading
import json

# 在MainApp类中添加以下方法

class MainApp(BaseGUI):
    def __init__(self):
        super().__init__(title="OceanDirect 光谱仪软件", size=f"{GUI_SETTINGS['window_width']}x{GUI_SETTINGS['window_height']}")
        self.api = OceanDirectAPI()
        self.i=0
        # 创建左侧滚动面板容器
        left_frame = ttk.Frame(self.main_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # 创建画布和滚动条
        canvas = tk.Canvas(left_frame, width=350, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        left_panel = ttk.Frame(canvas)
        
        # 配置画布
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 创建窗口在画布中
        canvas_window = canvas.create_window((0, 0), window=left_panel, anchor="nw", width=350)
        
        self.config_panel = ConfigPanel(
            left_panel
        )

        # 先创建所有面板，然后再设置回调
        self.spectrum_display = SpectrumDisplay(self.main_frame,self.config_panel)
        self.spectrum_display.pack(side="right", fill="both", expand=True)
        
        # 左侧面板按顺序排列
        self.spectrometer_panel = SpectrometerPanel(
            left_panel, 
            spectrum_display=self.spectrum_display,
            on_status=self.update_status,
            on_device_info=self.update_device_info
        )
        self.spectrometer_panel.pack(fill="x", pady=(0, 5))
        
        self.experiment_panel = ExperimentPanel(
            left_panel, self.spectrometer_panel
        )
        self.experiment_panel.pack(fill="x", pady=(0, 5))
        
        self.config_panel.pack(fill="x", pady=(0, 5))
        
        self.data_save_panel = DataSavePanel(
            left_panel,
            config_panel=self.config_panel,
            spectrometer=self.spectrometer_panel
        )
        self.data_save_panel.pack(fill="x", pady=(0, 5))
        
        self.spectrum_display._config_panel(
            self.data_save_panel,
            self.spectrometer_panel
        )
        self.experiment_panel._config_panel(
            self.data_save_panel
        )
        # 配置画布滚动 - 修复滚动条绑定
        def update_scroll_region(event=None):
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        left_panel.bind('<Configure>', on_frame_configure)
        canvas.bind('<Configure>', on_canvas_configure)
        
        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 设置最小宽度，避免抖动
        left_panel.update_idletasks()
        min_width = 350
        canvas.config(width=min_width)
        
        # 初始更新滚动区域
        self.root.after(100, update_scroll_region)
        
        self._start_ipc_server()

    def _start_ipc_server(self, host='127.0.0.1', port=9001):  # 改为9001端口
        def ipc_server():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            print(f"[UI] 本地IPC监听: {host}:{port}")
            while True:
                conn, addr = s.accept()
                with conn:
                    data = b''
                    while True:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            data += chunk
                            if b'\n' in chunk:
                                break
                    try:
                        msg = json.loads(data.decode())
                        result = self._handle_ipc_message(msg)
                        conn.sendall((json.dumps(result, ensure_ascii=False)+'\n').encode())
                    except Exception as e:
                        conn.sendall((json.dumps({'success': False, 'error': str(e)})+'\n').encode())
        t = threading.Thread(target=ipc_server, daemon=True)
        t.start()

    def _handle_ipc_message(self,msg):
        # 复用MCPServer的handle_message逻辑
        try:
            from mcp.server import MCPServer
            # 构造临时MCPServer对象以利用其handlers
            dummy = MCPServer(None, None, None, main_app=self)
            dummy.handlers = self._get_mcp_handlers()
            return dummy.handle_message(msg)
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _get_mcp_handlers(self):
        # 只暴露UI可用的handlers
        from mcp.server import MCPServer
        dummy = MCPServer(None, None, None, main_app=self)
        return dummy.handlers
    # 删除MainApp中的update_display方法
    

    def get_params(self):
        return {
            'integration_time_ms': float(self.spectrometer_panel.integration_time_var.get()),
            'scans_to_average': int(self.spectrometer_panel.scans_var.get()),
            'boxcar_width': int(self.spectrometer_panel.boxcar_var.get()),
        }

    def get_device_info(self):
        s = self.spectrometer_panel.spectrometer
        return {
            'device_serial': s.get_serial_number() if s else None,
            'device_model': s.get_model() if s else None
        }

    def update_status(self, msg):
        # 可扩展为状态栏显示
        print(msg)

    def update_device_info(self, info):
        # 可扩展为设备信息显示
        print(info)

    def setup_mcp_communication(self, ui_command_queue, notification_queue):
        """设置MCP通信队列"""
        self.ui_command_queue = ui_command_queue
        self.notification_queue = notification_queue
        
        # 创建通知标签
        self.mcp_notice_label = tk.Label(
            self.root, 
            text="", 
            bg="yellow", 
            fg="red", 
            font=("Arial", 12, "bold"),
            relief="raised",
            borderwidth=2
        )
        
        # 启动队列处理
        self.process_mcp_commands()
        self.process_notifications()
    
    def process_mcp_commands(self):
        """处理MCP命令队列"""
        try:
            while not self.ui_command_queue.empty():
                command = self.ui_command_queue.get_nowait()
                self.execute_mcp_command(command)
        except:
            pass
        
        # 每100ms检查一次
        self.root.after(100, self.process_mcp_commands)
    "fuck:step3,UI端队列处理.操作: 从队列中取出命令并调用 execute_mcp_command"
    
    def process_notifications(self):
        """处理通知队列"""
        try:
            while not self.notification_queue.empty():
                message = self.notification_queue.get_nowait()
                self.show_mcp_notice(message)
        except:
            pass
        
        # 每100ms检查一次
        self.root.after(100, self.process_notifications)
    
    # 在execute_mcp_command方法中添加实验功能处理
    def execute_mcp_command(self, command):
        """执行MCP命令"""
        action = command.get('action')
        value = command.get('value')
        
        # 1. 设备连接管理
        if action == 'connect_device' and hasattr(self, 'spectrometer_panel'):
            self.spectrometer_panel.connect_device()
        elif action == 'disconnect_device' and hasattr(self, 'spectrometer_panel'):
            self.spectrometer_panel.disconnect_device()
        elif action == 'start_mock_device' and hasattr(self, 'spectrometer_panel'):
            self.spectrometer_panel.enable_mock_device()
        
        # 2. 批量采集参数配置
        elif action == 'set_integration_time' and hasattr(self, 'spectrometer_panel'):
            self.spectrometer_panel.integration_time_var.set(value)
            self.spectrometer_panel.update_integration_time()
        elif action == 'set_scans_to_average' and hasattr(self, 'spectrometer_panel'):
            self.spectrometer_panel.scans_var.set(value)
            self.spectrometer_panel.update_scans()
        elif action == 'set_boxcar_width' and hasattr(self, 'spectrometer_panel'):
            self.spectrometer_panel.boxcar_var.set(value)
            self.spectrometer_panel.update_boxcar()
        
        # 3. 显示面板配置
        elif action == 'set_display_mode' and hasattr(self, 'config_panel'):
            mode_map = ['wavelength', 'raman']
            if value in mode_map:
                self.config_panel.display_mode_var.set(value)
                self.config_panel._mode_changed()
        elif action == 'set_raman_mode' and hasattr(self, 'config_panel'):
            direction_map = ['positive', 'negative']
            if value in direction_map:
                self.config_panel.raman_direction_var.set(value)
                self.config_panel._raman_direction_changed()
        elif action == 'set_baseline_correction' and hasattr(self, 'config_panel'):
            # 基线校正功能需要在config_panel中实现
            if hasattr(self.config_panel, 'baseline_correction_var'):
                self.config_panel.baseline_correction_var.set(value)
        
        # 4. 暗光谱配置
        elif action == 'capture_dark_spectrum' and hasattr(self, 'data_save_panel'):
            self.data_save_panel.capture_dark_spectrum()
        elif action == 'load_dark_spectrum' and hasattr(self, 'data_save_panel'):
            if value and 'filename' in value:
                self.data_save_panel.load_dark_spectrum(value['filename'])
            else:
                self.data_save_panel.load_dark_spectrum()
        elif action == 'clear_dark_spectrum' and hasattr(self, 'data_save_panel'):
            self.data_save_panel.clear_dark_spectrum()
        
        # 5. 通用数据保存 - 修复与UI功能一致
        elif action == 'save_data' and hasattr(self, 'data_save_panel'):
            if value:
                data_type = value.get('data_type', 'current_spectrum')
                filename = value.get('filename', 'spectrum')
                format_type = value.get('format', 'csv')
                apply_baseline = value.get('apply_baseline', False)
                include_metadata = value.get('include_metadata', True)
                
                if data_type == 'current_spectrum':
                    # 使用与UI界面相同的保存方法
                    self.data_save_panel.save_current_spectrum(filename, apply_baseline)
                elif data_type == 'experiment_data':
                    # 保存实验数据
                    metadata = value.get('metadata', {})
                    self.data_save_panel.save_experiment_data(format_type, metadata)
        elif action == 'save_experiment_data' and hasattr(self, 'data_save_panel'):
            if value and 'format' in value:
                # 根据格式保存实验数据
                format_type = value['format']
                metadata = value.get('metadata', {})
                self.data_save_panel.save_experiment_data(format_type, metadata)
        
        # 6. 实验模式管理 - 完善功能
        # 实验数据集配置
        elif action == 'configure_experiment_dataset' and hasattr(self, 'experiment_panel'):
            if value and hasattr(self.experiment_panel, 'dataset_panel'):
                variable_names = value.get('variable_names', [])
                dataset_path = value.get('dataset_path')
                file_prefix = value.get('file_prefix', 'spectral')
                
                dataset_panel = self.experiment_panel.dataset_panel
                # 设置变量名和路径
                dataset_panel.variable_names = variable_names
                if dataset_path:
                    dataset_panel.dataset_path = dataset_path
                    dataset_panel.dataset_manager.dataset_path = dataset_path
                
                # 设置文件前缀
                dataset_panel.prefix_var.set(file_prefix)
                dataset_panel.dataset_manager.set_file_prefix(file_prefix)
                
                # 更新UI
                dataset_panel.measurement_session.set_variables(variable_names)
                dataset_panel._rebuild_concentration_inputs()
                dataset_panel._update_ui_state()
        
        # 实验测量（带浓度参数）
        elif action == 'experiment_measure_and_save' and hasattr(self, 'experiment_panel'):
            if value and 'concentrations' in value and hasattr(self.experiment_panel, 'dataset_panel'):
                concentrations = value['concentrations']
                dataset_panel = self.experiment_panel.dataset_panel
                # 设置浓度值到输入框
                for i, concentration in enumerate(concentrations):
                    if i < len(dataset_panel.concentration_entries):
                        dataset_panel.concentration_entries[i].delete(0, 'end')
                        dataset_panel.concentration_entries[i].insert(0, str(concentration))
                # 执行测量并自动保存
                dataset_panel.measure_and_save()
        
        # 实验模式启动/停止
        elif action == 'start_experiment_mode' and hasattr(self, 'experiment_panel'):
            if hasattr(self.experiment_panel, 'dataset_panel'):
                self.experiment_panel.dataset_panel.start_experiment()
        elif action == 'stop_experiment_mode' and hasattr(self, 'experiment_panel'):
            if hasattr(self.experiment_panel, 'dataset_panel'):
                self.experiment_panel.dataset_panel.stop_experiment()
        elif action == 'delete_last_measurement' and hasattr(self, 'experiment_panel'):
            if hasattr(self.experiment_panel, 'dataset_panel'):
                self.experiment_panel.dataset_panel.delete_last_measurement()
        elif action == 'measure_and_save' and hasattr(self, 'experiment_panel'):
            if hasattr(self.experiment_panel, 'dataset_panel'):
                self.experiment_panel.dataset_panel.measure_and_save()
        elif action == 'save_experiment_results' and hasattr(self, 'experiment_panel'):
            if value and 'filename' in value:
                filename = value['filename']
                if hasattr(self.experiment_panel, 'dataset_panel') and hasattr(self.experiment_panel.dataset_panel, 'save_experiment_results'):
                    self.experiment_panel.dataset_panel.save_experiment_results(filename)
        "fuck:step4,UI端命令映射与执行"
    
    def show_mcp_notice(self, message="AI正在远程控制，请勿手动操作", duration=3000):
        """显示MCP通知"""
        if self.mcp_notice_label:
            self.mcp_notice_label.config(text=message)
            self.mcp_notice_label.pack(side="top", fill="x", padx=5, pady=2)
            
            # 取消之前的定时器
            if self.notice_timer:
                self.root.after_cancel(self.notice_timer)
            
            # 设置自动隐藏
            self.notice_timer = self.root.after(duration, self.hide_mcp_notice)
    
    def hide_mcp_notice(self):
        """隐藏MCP通知"""
        if self.mcp_notice_label:
            self.mcp_notice_label.pack_forget()
        if self.notice_timer:
            self.root.after_cancel(self.notice_timer)
            self.notice_timer = None

if __name__ == "__main__":
    app = MainApp()
    # 添加中文支持
    from matplotlib import pyplot as plt
    plt.rcParams['font.family'] = 'SimHei' # 或者其他支持中文的字体，如 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    parser = argparse.ArgumentParser()
    parser.add_argument('--mcp-mode', default='sse')
    parser.add_argument('--mcp-port', type=int, default=8080)
    args = parser.parse_args()
    
    app.run()
