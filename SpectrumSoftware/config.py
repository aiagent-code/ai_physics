# -*- coding: utf-8 -*-
"""
配置文件 - OceanDirect 光谱仪数据采集系统
"""

# 默认采集参数
DEFAULT_PARAMS = {
    'integration_time_ms': 100,  # 积分时间 (毫秒)
    'scans_to_average': 1,       # 扫描次数
    'boxcar_width': 0,           # Boxcar宽度
    'acquisition_interval': 0.1, # 实时采集间隔 (秒)
}

# GUI设置
GUI_SETTINGS = {
    'window_width': 1400,
    'window_height': 900,
    'plot_figsize': (12, 6),  # 增加图形尺寸
    'plot_dpi': 100,
    'control_panel_width': 100,  # 减少控制面板宽度
}

# 文件保存设置
SAVE_SETTINGS = {
    'default_directory': './spectrum_data',
    'file_formats': {
        'csv': {
            'extension': '.csv',
            'description': 'CSV files (*.csv)',
            'delimiter': ',',
        },
        'txt': {
            'extension': '.txt',
            'description': 'Text files (*.txt)',
            'delimiter': '\t',
        },
        'json': {
            'extension': '.json',
            'description': 'JSON files (*.json)',
        }
    },
    'auto_save': False,
    'auto_save_interval': 60,  # 自动保存间隔 (秒)
}

# 光谱处理设置
SPECTRUM_PROCESSING = {
    'dark_correction': True,      # 启用暗光谱校正
    'nonlinearity_correction': False,  # 启用非线性校正
    'boxcar_smoothing': False,    # 启用Boxcar平滑
    'min_intensity': 0,           # 最小强度阈值
    'max_intensity': None,        # 最大强度阈值 (None表示自动)
}

# 显示设置
DISPLAY_SETTINGS = {
    'show_dark_spectrum': True,   # 显示暗光谱
    'show_raw_data': False,       # 显示原始数据
    'show_grid': True,            # 显示网格
    'auto_scale': True,           # 自动缩放
    'line_width': 1,              # 线条宽度
    'alpha': 0.7,                 # 透明度
    'colors': {
        'current_spectrum': 'blue',
        'dark_spectrum': 'red',
        'raw_data': 'green',
    }
}

# 设备设置
DEVICE_SETTINGS = {
    'auto_connect': True,         # 自动连接设备
    'connection_timeout': 10,     # 连接超时时间 (秒)
    'retry_attempts': 3,          # 重试次数
    'supported_features': [
        'SPECTROMETER',
        'THERMOELECTRIC',
        'LIGHT_SOURCE',
        'SHUTTER',
        'WAVELENGTH_CAL',
        'NONLINEARITY_CAL',
        'STRAYLIGHT_CAL',
    ]
}

# 日志设置
LOGGING_SETTINGS = {
    'level': 'INFO',              # 日志级别
    'file': 'spectrometer.log',   # 日志文件
    'max_size': 10 * 1024 * 1024,  # 最大文件大小 (10MB)
    'backup_count': 5,            # 备份文件数量
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# 错误处理设置
ERROR_HANDLING = {
    'show_error_dialogs': True,   # 显示错误对话框
    'log_errors': True,           # 记录错误日志
    'retry_on_error': True,       # 错误时重试
    'max_retries': 3,             # 最大重试次数
}

# 性能设置
PERFORMANCE_SETTINGS = {
    'update_rate': 10,            # 更新频率 (Hz)
    'buffer_size': 1000,          # 数据缓冲区大小
    'thread_timeout': 5,          # 线程超时时间 (秒)
    'memory_limit': 100 * 1024 * 1024,  # 内存限制 (100MB)
}

# 校准设置
CALIBRATION_SETTINGS = {
    'wavelength_calibration': True,    # 波长校准
    'intensity_calibration': False,    # 强度校准
    'dark_calibration': True,          # 暗电流校准
    'reference_spectrum': None,        # 参考光谱文件
}

# 网络设置 (如果使用网络设备)
NETWORK_SETTINGS = {
    'timeout': 5,                 # 网络超时时间 (秒)
    'retry_interval': 1,          # 重试间隔 (秒)
    'max_connections': 5,         # 最大连接数
    'multicast_enabled': False,   # 启用多播
}

# 数据格式设置
DATA_FORMAT_SETTINGS = {
    'wavelength_precision': 3,    # 波长精度 (小数位数)
    'intensity_precision': 6,     # 强度精度 (小数位数)
    'timestamp_format': '%Y-%m-%d %H:%M:%S',  # 时间戳格式
    'include_metadata': True,     # 包含元数据
}

# 导出设置
EXPORT_SETTINGS = {
    'include_headers': True,      # 包含表头
    'include_timestamp': True,    # 包含时间戳
    'include_device_info': True,  # 包含设备信息
    'include_parameters': True,   # 包含采集参数
    'compression': False,         # 启用压缩
}

# 主题设置
THEME_SETTINGS = {
    'style': 'default',           # GUI样式
    'color_scheme': 'light',      # 颜色方案 (light/dark)
    'font_size': 10,              # 字体大小
    'font_family': 'Arial',       # 字体族
}

# 快捷键设置
SHORTCUTS = {
    'start_acquisition': 'Ctrl+S',
    'stop_acquisition': 'Ctrl+X',
    'capture_dark': 'Ctrl+D',
    'save_spectrum': 'Ctrl+Shift+S',
    'auto_scale': 'Ctrl+A',
    'reset_zoom': 'Ctrl+R',
    'connect_device': 'Ctrl+C',
    'disconnect_device': 'Ctrl+Shift+C',
}

# 帮助信息
HELP_INFO = {
    'integration_time': '积分时间决定了光谱仪收集光子的时间长度，影响信噪比和动态范围',
    'scans_to_average': '多次扫描的平均可以提高信噪比，但会增加采集时间',
    'boxcar_width': 'Boxcar平滑可以减少噪声，但会降低光谱分辨率',
    'dark_correction': '暗光谱校正可以消除暗电流和电子噪声的影响',
    'real_time_acquisition': '实时采集模式会持续获取光谱数据并更新显示',
    'single_acquisition': '单次采集模式获取一次光谱数据',
}

# 版本信息
VERSION_INFO = {
    'version': '1.0.0',
    'build_date': '2024-01-01',
    'author': 'OceanDirect SDK User',
    'description': 'OceanDirect 光谱仪数据采集系统',
    'requirements': [
        'oceandirect',
        'numpy',
        'matplotlib',
        'tkinter',
    ]
} 