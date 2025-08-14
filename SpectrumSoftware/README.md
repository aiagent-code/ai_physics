# OceanDirect 光谱仪控制系统

## 项目简介

本项目是一个基于 Python 的 Ocean Optics 光谱仪控制系统，提供图形界面和 MCP (Model Context Protocol) 远程控制两种操作方式。系统采用模块化设计，支持实时光谱采集、数据处理、实验管理和多种数据导出格式。

## 核心特性

### 🔬 光谱仪控制
- 支持 Ocean Optics 光谱仪设备的连接和控制
- 实时光谱数据采集和显示
- 积分时间、扫描次数等参数配置
- 模拟设备模式，支持无硬件测试

### 📊 数据处理与显示
- 多种显示模式：波长、波数、拉曼位移
- 暗光谱采集和减除功能
- 基线校正和光谱平滑处理
- 实时光谱可视化和交互式缩放

### 🧪 实验管理
- 完整的实验流程管理
- 数据集配置和批量测量
- 自定义文件命名和存储路径
- 实验环境配置自动记录

### 🤖 双模式控制
- **图形界面模式**：直观的 GUI 操作界面
- **MCP 远程控制**：支持 AI 客户端远程操作
- **混合模式**：UI 和 MCP 可同时运行，互不冲突

## 系统架构设计

### 统筹设计思路

本系统采用**双线程架构**，巧妙地统筹了 MCP 远程控制和 UI 界面两种请求方式：

#### 1. 线程分离与通信
- **UI 线程**：运行图形界面，处理用户交互
- **MCP 线程**：运行 MCP 服务器，处理远程控制请求
- **通信机制**：通过 `queue.Queue` 实现线程间安全通信
  - `ui_command_queue`：MCP 向 UI 发送命令
  - `notification_queue`：UI 向 MCP 发送状态通知

#### 2. 互不冲突的设计
- **命令队列化**：所有 MCP 命令都通过队列传递给 UI 线程执行
- **状态同步**：UI 操作状态实时同步到 MCP 服务器
- **冲突提示**：MCP 操作时 UI 显示提醒，避免用户误操作
- **统一接口**：MCP 和 UI 调用相同的底层方法，保证一致性

#### 3. 核心大循环逻辑

系统的核心是 `spectrometer_panel` 中的**实时采集大循环**：

```
光谱仪连接状态检测
    ↓
定时调用光谱仪API获取数据
    ↓
调用 spectrum_display.py 处理和显示
    ↓
查询 config_panel 当前配置
    ↓
更新 data_save_panel 数据存储和格式
    ↓
循环继续...
```

**数据流向**：
- `spectrometer_panel`：设备控制和数据采集
- `spectrum_display`：数据处理和可视化
- `config_panel`：配置管理和参数控制
- `data_save_panel`：数据存储和导出（主要光谱数据存储接口）

## 快速开始

### 环境要求
- Python 3.8+
- Windows 操作系统
- Ocean Optics 光谱仪驱动（可选，支持模拟模式）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行方式

#### 1. 仅图形界面模式
```bash
python spectrum_main.py
```

#### 2. 仅 MCP 服务器模式
```bash
python mcp_main.py
```

#### 3. 混合模式（推荐）
```bash
python main.py
```

混合模式同时启动 UI 和 MCP 服务器，支持：
- 本地 GUI 操作
- 远程 MCP 控制
- 实时状态同步
- 操作冲突提示

## 项目结构

```
├── main.py                 # 主程序入口（混合模式）
├── spectrum_main.py        # 光谱仪 GUI 主程序
├── mcp_main.py            # MCP 服务器主程序
├── config.py              # 全局配置文件
├── gui/                   # GUI 模块
│   ├── spectrometer_panel.py    # 设备控制面板
│   ├── spectrum_display.py      # 光谱显示面板
│   ├── data_save_panel.py       # 数据保存面板
│   ├── config_panel.py          # 配置面板
│   ├── experiment_panel.py      # 实验管理面板
│   └── ...
├── experiment/            # 实验管理模块
├── processor/             # 数据处理模块
├── oceandirect/          # Ocean Optics SDK 封装
└── mcp_my/               # MCP 工具和装饰器
```

## 使用指南

### 基本操作流程
1. **设备连接**：启动程序后点击"查找设备"和"连接设备"
2. **参数配置**：设置积分时间、扫描次数等采集参数
3. **实时采集**：点击"开始采集"查看实时光谱
4. **数据保存**：配置保存路径和格式，保存光谱数据
5. **实验模式**：创建实验，进行批量测量和数据管理

### MCP 远程控制
- 启动混合模式后，MCP 服务器默认运行在 `http://127.0.0.1:8000`
- 支持 Claude Desktop、其他 MCP 客户端连接
- 提供完整的设备控制、数据采集、实验管理 API

## 配置文件

- `config.py`：GUI 界面配置
- `mcp_config.json`：MCP 服务器配置
- `experiment_config.json`：实验配置模板

## 开发说明

### 类与对象设计原则
- **功能分离**：每个类负责特定功能模块
- **数据封装**：各模块存储不同类型的数据
- **接口统一**：提供标准化的方法接口
- **松耦合**：通过回调和队列实现模块间通信

### 扩展开发
- 新增 MCP 工具：在 `mcp_main.py` 中注册新的工具函数
- 新增 GUI 面板：继承 `ttk.LabelFrame` 并实现标准接口
- 新增数据处理：在 `processor/` 目录下添加处理模块

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系我们。