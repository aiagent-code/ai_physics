"""  
MCP服务器主模块
功能：提供MCP工具定义、配置管理和服务器运行逻辑
作用：处理AI客户端的远程控制请求，与光谱仪UI进行通信
存储数据：
- mcp: FastMCP服务器实例
- config: MCP配置信息
- ui_command_queue: UI命令队列
- notification_queue: 通知队列
- main_app: 主应用实例引用
"""

import json
import threading
import queue
import time
from mcp.server.fastmcp import FastMCP
from mcp_my.utils.decorators import with_notice

class MCPServer:
    def __init__(self):
        # 读取配置文件
        self.config = {}
        try:
            with open('mcp_config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print("配置文件mcp_config.json未找到，使用默认配置")
            self.config = {"transport": "stdio"}
        
        # 解析host和port
        self.host = "127.0.0.1"
        self.port = 8000
        if "host_port" in self.config:
            host_port = self.config["host_port"]
            if ":" in host_port:
                host_part, port_str = host_port.split(":")
                # 将localhost转换为127.0.0.1
                if host_part.lower() == "localhost":
                    self.host = "127.0.0.1"
                else:
                    self.host = host_part
                self.port = int(port_str)
        
        # 初始化FastMCP服务器
        self.mcp = FastMCP(
            "ocean_direct", 
            host=self.host, 
            port=self.port,
            sse_path="/sse",
        )
        
        # 通信相关
        self.ui_command_queue = None
        self.notification_queue = None
        self.main_app = None
        self.ui_ready = None
        
        # 注册工具
        self._register_tools()
    
    def setup_communication(self, ui_command_queue, notification_queue, main_app, ui_ready):
        """设置与UI的通信"""
        self.ui_command_queue = ui_command_queue
        self.notification_queue = notification_queue
        self.main_app = main_app
        self.ui_ready = ui_ready
    
    def _register_tools(self):
        """注册所有MCP工具"""

        # 启用模拟设备（保留）
        @self.mcp.tool()
        @with_notice("启动模拟设备")
        def start_mock_device() -> dict:
            """启动模拟光谱仪设备"""
            if self.main_app:
                self.ui_command_queue.put({
                    'action': 'start_mock_device',
                    'value': None
                })
                return {"status": "success", "message": "模拟设备已启动，开始实时采集模拟光谱数据"}
            return {"error": "UI未初始化"}
        
        # 1. 设备连接管理（合并连接/断开/模拟设备功能）
        @self.mcp.tool()
        @with_notice("设备连接管理")
        def manage_device_connection(action: str, device_type: str = "real") -> dict:
            """管理设备连接状态
            
            Args:
                action: 操作类型 - "connect", "disconnect", "mock"
                device_type: 设备类型 - "real" 或 "mock"
            """
            if self.main_app:
                if action == "connect":
                    self.ui_command_queue.put({
                        'action': 'connect_device',
                        'value': device_type
                    })
                    return {"status": "success", "message": f"正在连接{device_type}设备"}
                elif action == "disconnect":
                    self.ui_command_queue.put({
                        'action': 'disconnect_device',
                        'value': None
                    })
                    return {"status": "success", "message": "设备已断开连接"}
                elif action == "mock":
                    self.ui_command_queue.put({
                        'action': 'start_mock_device',
                        'value': None
                    })
                    return {"status": "success", "message": "模拟设备已启动"}
                else:
                    return {"error": "无效的操作类型，请使用 connect, disconnect 或 mock"}
            return {"error": "UI未初始化"}
        
        # 2. 批量参数配置
        @self.mcp.tool()
        @with_notice("批量配置采集参数")
        def configure_acquisition_parameters(integration_time: int = None, scans_to_average: int = None, boxcar_width: int = None) -> dict:
            """批量配置采集参数
            
            Args:
                integration_time: 积分时间(毫秒)
                scans_to_average: 平均扫描次数
                boxcar_width: Boxcar平滑宽度
            """
            if self.main_app:
                config_results = []
                
                if integration_time is not None:
                    self.ui_command_queue.put({
                        'action': 'set_integration_time',
                        'value': integration_time
                    })
                    config_results.append(f"积分时间: {integration_time}ms")
                
                if scans_to_average is not None:
                    self.ui_command_queue.put({
                        'action': 'set_scans_to_average',
                        'value': scans_to_average
                    })
                    config_results.append(f"扫描次数: {scans_to_average}")
                
                if boxcar_width is not None:
                    self.ui_command_queue.put({
                        'action': 'set_boxcar_width',
                        'value': boxcar_width
                    })
                    config_results.append(f"Boxcar宽度: {boxcar_width}")
                
                if config_results:
                    return {"status": "success", "message": f"参数配置完成: {', '.join(config_results)}"}
                else:
                    return {"status": "info", "message": "未提供任何参数进行配置"}
            return {"error": "UI未初始化"}
        
        # 3. 配置显示面板设置
        @self.mcp.tool()
        @with_notice("配置显示面板")
        def configure_display_panel(display_mode: str = None, raman_mode: bool = None, baseline_correction: bool = None) -> dict:
            """配置显示面板设置
            
            Args:
                display_mode: 显示模式 - "spectrum", "absorbance", "transmission"
                raman_mode: 是否启用拉曼模式
                baseline_correction: 是否启用基线校正
            """
            if self.main_app:
                config_results = []
                
                if display_mode is not None:
                    self.ui_command_queue.put({
                        'action': 'set_display_mode',
                        'value': display_mode
                    })
                    config_results.append(f"显示模式: {display_mode}")
                
                if raman_mode is not None:
                    self.ui_command_queue.put({
                        'action': 'set_raman_mode',
                        'value': raman_mode
                    })
                    config_results.append(f"拉曼模式: {'启用' if raman_mode else '禁用'}")
                
                if baseline_correction is not None:
                    self.ui_command_queue.put({
                        'action': 'set_baseline_correction',
                        'value': baseline_correction
                    })
                    config_results.append(f"基线校正: {'启用' if baseline_correction else '禁用'}")
                
                if config_results:
                    return {"status": "success", "message": f"显示面板配置完成: {', '.join(config_results)}"}
                else:
                    return {"status": "info", "message": "未提供任何显示配置参数"}
            return {"error": "UI未初始化"}
        
        # 4. 暗光谱配置
        @self.mcp.tool()
        @with_notice("暗光谱配置")
        def configure_dark_spectrum(action: str, filename: str = None) -> dict:
            """暗光谱配置管理
            
            Args:
                action: 操作类型 - "capture", "load", "clear"
                filename: 文件名（用于load操作）
            """
            if self.main_app:
                if action == "capture":
                    self.ui_command_queue.put({
                        'action': 'capture_dark_spectrum',
                        'value': None
                    })
                    return {"status": "success", "message": "暗光谱采集完成"}
                elif action == "load":
                    if filename:
                        self.ui_command_queue.put({
                            'action': 'load_dark_spectrum',
                            'value': filename
                        })
                        return {"status": "success", "message": f"暗光谱已从 {filename} 加载"}
                    else:
                        return {"error": "加载暗光谱需要指定文件名"}
                elif action == "clear":
                    self.ui_command_queue.put({
                        'action': 'clear_dark_spectrum',
                        'value': None
                    })
                    return {"status": "success", "message": "暗光谱已清除"}
                else:
                    return {"error": "无效的操作类型，请使用 capture, load 或 clear"}
            return {"error": "UI未初始化"}
        
        # 5. 通用数据保存功能
        # 修改save_data工具函数，使其与UI功能完全一致
        @self.mcp.tool()
        @with_notice("通用数据保存")
        def save_data(data_type: str = "current_spectrum", filename: str = None, format: str = "csv", apply_baseline: bool = False, include_metadata: bool = True) -> dict:
            """通用数据保存功能 - 与UI界面功能完全一致
            
            Args:
                data_type: 数据类型 - "current_spectrum", "experiment_data"
                filename: 文件名（不含扩展名）
                format: 保存格式 - "csv", "txt", "json"
                apply_baseline: 是否应用基线校正
                include_metadata: 是否包含元数据
            """
            if self.main_app:
                if not filename:
                    filename = f"{data_type}_data"
                
                self.ui_command_queue.put({
                    'action': 'save_data',
                    'value': {
                        'data_type': data_type,
                        'filename': filename,
                        'format': format,
                        'apply_baseline': apply_baseline,
                        'include_metadata': include_metadata
                    }
                })
                return {"status": "success", "message": f"{data_type}数据已保存为 {filename}.{format}"}
            return {"error": "UI未初始化"}
        
        # 6. 实验数据集配置工具
        @self.mcp.tool()
        @with_notice("配置实验数据集")
        def configure_experiment_dataset(variable_names: list, dataset_path: str = None, file_prefix: str = "spectral") -> dict:
            """配置实验数据集
            
            Args:
                variable_names: 变量名称列表，如["浓度1", "浓度2"]
                dataset_path: 数据集存储路径
                file_prefix: 文件前缀
            """
            if self.main_app:
                if not variable_names:
                    return {"error": "变量名称列表不能为空"}
                
                self.ui_command_queue.put({
                    'action': 'configure_experiment_dataset',
                    'value': {
                        'variable_names': variable_names,
                        'dataset_path': dataset_path,
                        'file_prefix': file_prefix
                    }
                })
                return {"status": "success", "message": f"数据集已配置: {len(variable_names)}个变量, 前缀: {file_prefix}"}
            return {"error": "UI未初始化"}
        
        # 7. 实验模式管理（简化版）
        @self.mcp.tool()
        @with_notice("实验模式管理")
        def manage_experiment_mode(action: str, experiment_name: str = None) -> dict:
            """实验模式管理
            
            Args:
                action: 操作类型 - "start", "stop"
                experiment_name: 实验名称（可选）
            """
            if self.main_app:
                if action == "start":
                    self.ui_command_queue.put({
                        'action': 'start_experiment_mode',
                        'value': experiment_name or "default"
                    })
                    return {"status": "success", "message": f"实验模式已启动: {experiment_name or 'default'}"}
                elif action == "stop":
                    self.ui_command_queue.put({
                        'action': 'stop_experiment_mode',
                        'value': None
                    })
                    return {"status": "success", "message": "实验模式已停止"}
                else:
                    return {"error": "无效的操作类型，请使用 start 或 stop"}
            return {"error": "UI未初始化"}
        
        # 8. 实验测量工具（支持浓度参数）
        @self.mcp.tool()
        @with_notice("实验测量")
        def experiment_measure(concentrations: list) -> dict:
            """执行实验测量并自动保存
            
            Args:
                concentrations: 浓度值列表，如[23.5, 43.2]
            """
            if self.main_app:
                if not concentrations:
                    return {"error": "浓度值列表不能为空"}
                
                self.ui_command_queue.put({
                    'action': 'experiment_measure_and_save',
                    'value': {
                        'concentrations': concentrations
                    }
                })
                return {"status": "success", "message": f"测量完成，浓度: {concentrations}"}
            return {"error": "UI未初始化"}
        
        # 9. 获取系统状态信息
        @self.mcp.tool()
        @with_notice("获取系统状态")
        def get_system_status() -> dict:
            """获取系统状态信息"""
            if self.main_app:
                try:
                    # 获取设备状态
                    device_info = self.main_app.get_device_info()
                    params = self.main_app.get_params()
                    
                    status = {
                        "device_connected": device_info.get("connected", False),
                        "device_type": device_info.get("type", "unknown"),
                        "integration_time": params.get("integration_time_ms", 0),
                        "scans_to_average": params.get("scans_to_average", 1),
                        "boxcar_width": params.get("boxcar_width", 0),
                        "experiment_running": hasattr(self.main_app, 'experiment_panel') and 
                                            getattr(self.main_app.experiment_panel, 'is_running', False),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    return {"status": "success", "data": status}
                except Exception as e:
                    return {"error": f"获取系统状态失败: {str(e)}"}
            return {"error": "UI未初始化"}
    
    def run_server(self):
        """运行MCP服务器"""
        try:
            # 等待UI初始化完成
            if self.ui_ready:
                self.ui_ready.wait()
            
            # 确保main_app已初始化
            while self.main_app is None:
                time.sleep(0.1)
            
            print(f"MCP服务器启动配置:")
            print(f"  - 传输方式: {self.config.get('transport', 'stdio')}")
            if self.config.get('transport') == 'sse':
                print(f"  - 服务器地址: http://{self.host}:{self.port}")
                print(f"  - SSE端点: http://{self.host}:{self.port}/sse")
            
            # 运行服务器
            self.mcp.run(transport=self.config.get('transport', 'stdio'))
            
        except KeyboardInterrupt:
            print("\nMCP服务器收到中断信号，正在关闭...")
        except Exception as e:
            print(f"MCP服务器运行错误: {e}")
        finally:
            print("MCP服务器线程已退出")

