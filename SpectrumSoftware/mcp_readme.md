# MCP 光谱仪控制服务器详细指南

## 项目简介

本项目提供了一个基于 MCP (Model Context Protocol) 的光谱仪控制服务器，支持远程控制 Ocean Optics 光谱仪设备。项目包含图形界面和 MCP 服务器两种运行模式，可以通过多种 MCP 客户端进行远程控制。

## MCP 架构设计

### 双线程统筹设计

本系统巧妙地统筹了 MCP 远程控制和 UI 界面两种请求方式，实现**互不冲突**的双模式操作：

#### 1. 线程架构
```
主程序 (main.py)
├── UI 线程 (spectrum_main.py)
│   ├── 图形界面运行
│   ├── 用户交互处理
│   └── 命令队列监听
└── MCP 线程 (mcp_main.py)
    ├── MCP 服务器运行
    ├── AI 客户端请求处理
    └── 命令队列发送
```

#### 2. 通信机制
- **ui_command_queue**：MCP → UI 命令传递
- **notification_queue**：UI → MCP 状态同步
- **线程安全**：使用 `queue.Queue` 确保数据安全传递
- **异步处理**：命令异步执行，不阻塞任何一方

#### 3. 冲突避免策略
- **命令队列化**：所有 MCP 命令进入队列，由 UI 线程统一执行
- **状态提示**：MCP 操作时 UI 显示"AI正在远程控制"提示
- **操作锁定**：关键操作期间暂时锁定用户界面
- **实时同步**：UI 状态变化实时反馈给 MCP 客户端

## 功能特性

### 🔬 设备控制功能
- **设备管理**：查找、连接、断开光谱仪设备
- **参数配置**：积分时间、扫描次数、Boxcar宽度设置
- **模拟模式**：支持无硬件的模拟设备测试
- **状态监控**：实时设备状态和连接信息获取

### 📊 数据采集功能
- **实时采集**：启动/停止实时光谱数据采集
- **单次测量**：获取单次光谱测量数据
- **暗光谱管理**：采集、加载、清除暗光谱
- **数据处理**：光谱平滑、基线校正、拉曼转换

### 🧪 实验管理功能
- **实验配置**：创建和配置实验参数
- **批量测量**：自动化批量数据采集
- **数据集管理**：数据集创建、变量配置
- **结果保存**：多格式数据导出和保存

### 🖥️ 显示控制功能
- **显示模式**：波长、波数、拉曼位移模式切换
- **可视化控制**：缩放、平移、基线显示控制
- **界面同步**：MCP 操作与 UI 界面实时同步

## 安装与配置

### 环境要求
```bash
Python 3.8+
Windows 操作系统
Ocean Optics 光谱仪驱动（可选）
```

### 依赖安装
```bash
pip install -r requirements.txt
```

主要依赖包括：
- `mcp` - MCP 协议支持
- `fastmcp` - 快速 MCP 服务器框架
- `tkinter` - 图形界面
- `numpy` - 数据处理
- `matplotlib` - 光谱显示

### 配置文件

#### mcp_config.json
```json
{
    "transport": "sse",
    "host_port": "127.0.0.1:8000",
    "sse_path": "/sse",
    "description": "Ocean Optics 光谱仪 MCP 控制服务器"
}
```

## 运行方式

### 1. 图形界面模式
```bash
python spectrum_main.py
```
仅启动 GUI 界面，适合本地操作。

### 2. MCP 服务器模式
```bash
python mcp_main.py
```
仅启动 MCP 服务器，适合纯远程控制。

### 3. 混合模式（推荐）
```bash
python main.py
```
同时启动 UI 和 MCP 服务器，支持：
- 本地 GUI 操作
- 远程 MCP 控制
- 实时状态同步
- 操作冲突提示

## MCP 工具函数

### 设备管理工具
- `manage_device_connection`：统一设备连接管理
- `get_system_status`：获取系统状态信息

### 参数配置工具
- `configure_acquisition_parameters`：批量配置采集参数
- `configure_display_panel`：配置显示面板设置
- `configure_dark_spectrum`：暗光谱配置管理

### 数据操作工具
- `save_data`：通用数据保存功能
- `manage_experiment_mode`：完整实验模式管理

## 客户端连接

### Claude Desktop 配置
在 Claude Desktop 配置文件中添加：
```json
{
    "mcpServers": {
        "ocean_direct": {
            "command": "python",
            "args": ["path/to/mcp_main.py"],
            "env": {
                "PYTHONPATH": "path/to/project"
            }
        }
    }
}
```

### 其他 MCP 客户端
- 服务器地址：`http://127.0.0.1:8000`
- SSE 端点：`http://127.0.0.1:8000/sse`
- 协议：MCP over SSE (Server-Sent Events)

## 使用示例

### 基本设备控制
```python
# 通过 MCP 客户端调用
# 1. 连接设备
manage_device_connection(action="connect")

# 2. 配置参数
configure_acquisition_parameters(
    integration_time=100,
    scans_to_average=3,
    boxcar_width=1
)

# 3. 开始采集
manage_device_connection(action="start_acquisition")

# 4. 保存数据
save_data(data_type="current_spectrum", filename="test.csv")
```

### 实验模式操作
```python
# 1. 配置实验
manage_experiment_mode(
    action="configure",
    dataset_path="./experiment_data",
    variables=["concentration", "temperature"]
)

# 2. 开始实验
manage_experiment_mode(action="start")

# 3. 进行测量
manage_experiment_mode(
    action="measure",
    variable_values=["10%", "25°C"]
)

# 4. 保存结果
manage_experiment_mode(action="save_results")
```

## 错误处理

### 常见错误及解决方案

1. **端口占用错误**
   ```
   错误：Address already in use
   解决：检查端口 8000 是否被占用，或修改配置文件中的端口
   ```

2. **设备连接失败**
   ```
   错误：No devices found
   解决：检查设备连接，或启用模拟设备模式
   ```

3. **MCP 通信错误**
   ```
   错误：Connection refused
   解决：确保 MCP 服务器正在运行，检查防火墙设置
   ```

## 开发指南

### 添加新的 MCP 工具

1. 在 `mcp_main.py` 中定义工具函数：
```python
@mcp.tool()
def my_new_tool(param1: str, param2: int) -> str:
    """工具描述"""
    # 实现工具逻辑
    return result
```

2. 注册工具到服务器：
```python
def _register_tools(self):
    # 现有工具注册...
    self.mcp.tool()(my_new_tool)
```

### 扩展通信协议

1. 修改命令队列处理逻辑
2. 添加新的通知类型
3. 更新状态同步机制

## 性能优化

### 建议配置
- **内存使用**：建议系统内存 ≥ 4GB
- **CPU 要求**：双核以上处理器
- **网络延迟**：本地连接延迟 < 10ms

### 优化策略
- 使用异步处理减少阻塞
- 合理设置队列大小避免内存溢出
- 定期清理临时数据和缓存

## 安全考虑

- MCP 服务器默认仅监听本地地址 (127.0.0.1)
- 生产环境建议配置访问控制和身份验证
- 敏感数据传输建议使用 HTTPS

## 故障排除

### 日志查看
程序运行时会输出详细日志，包括：
- 设备连接状态
- MCP 命令执行情况
- 错误信息和堆栈跟踪

### 调试模式
启动时添加调试参数：
```bash
python main.py --debug
```

## 版本更新

当前版本支持的主要功能：
- MCP 协议完整支持
- 双线程架构稳定运行
- 完整的光谱仪控制功能
- 实验管理和数据保存

## 技术支持

如遇到问题，请：
1. 查看程序输出的错误日志
2. 检查配置文件是否正确
3. 确认设备驱动是否安装
4. 通过 GitHub Issues 报告问题

---

**注意**：本 MCP 服务器设计为与 UI 界面协同工作，建议使用混合模式以获得最佳体验。