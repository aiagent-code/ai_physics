# 光谱仪系统设计思路与架构说明

## 系统概述

本光谱仪控制系统采用**模块化面向对象设计**，通过类与对象的程序思路实现各功能模块的分离与协作。系统核心是一个**实时数据采集大循环**，配合多个专业化面板类，实现了光谱仪的完整控制和数据处理流程。

## 核心设计思路

### 1. 大循环逻辑架构

系统的核心是 `spectrometer_panel` 中的**实时采集大循环**，这是整个系统的数据流动引擎：

```
┌─────────────────────────────────────────────────────────────┐
│                    大循环逻辑流程                              │
├─────────────────────────────────────────────────────────────┤
│  1. 光谱仪连接状态检测                                        │
│  2. 定时调用光谱仪API获取原始数据                             │
│  3. 调用 spectrum_display.py 进行数据处理和显示              │
│  4. spectrum_display 查询 config_panel 获取当前配置          │
│  5. 根据配置处理数据（显示模式、拉曼转换、基线校正等）         │
│  6. 将处理后的数据传递给 data_save_panel 进行存储             │
│  7. data_save_panel 更新内部数据结构和格式                   │
│  8. 循环继续...                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 数据流向详解

```
spectrometer_panel (数据源)
    ↓ [原始光谱数据]
spectrum_display (数据处理中心)
    ↓ [查询配置]
config_panel (配置管理)
    ↓ [配置参数]
spectrum_display (应用配置处理数据)
    ↓ [处理后数据]
data_save_panel (数据存储中心)
    ↓ [存储和格式化]
[光谱数据持久化存储]
```

### 2. 类与对象设计原则

#### 功能分离与职责明确
每个类都有明确的功能定位和数据管理职责：

**优势体现**：
- **模块独立性**：各模块功能独立，便于维护和扩展
- **数据封装性**：每个模块管理特定类型的数据
- **接口标准化**：提供统一的方法接口供其他模块调用
- **松耦合设计**：通过回调函数和参数传递实现模块间通信

## 核心模块详细设计

### 1. SpectrometerPanel - 设备控制与数据采集

**功能定位**：硬件设备控制和原始数据获取

**存储数据**：
- `api, spectrometer`：设备API和光谱仪实例
- `is_acquiring, _auto_refresh`：采集状态标志
- `wavelengths`：波长数据（只存储波长，不存储光谱数据）
- `integration_time_var, scans_var`：采集参数

**核心方法**：
- `_acquisition_loop()`：**大循环核心函数**
- `get_current_spectrum()`：获取当前光谱数据
- `connect_device()`, `disconnect_device()`：设备连接管理
- `apply_parameters()`：参数应用

**大循环实现**：
```python
def _acquisition_loop(self, on_update_plot):
    """实时采集大循环 - 系统核心"""
    while self.is_acquiring:
        if self.spectrometer:
            # 1. 获取光谱数据
            spectrum = self.spectrometer.get_spectrum()
            # 2. 调用回调函数传递数据
            if on_update_plot:
                on_update_plot(spectrum, self.wavelengths)
        time.sleep(0.1)  # 控制采集频率
```

### 2. SpectrumDisplay - 数据处理与可视化中心

**功能定位**：数据处理、可视化显示和配置协调

**存储数据**：
- `fig, ax, canvas`：matplotlib图形对象
- `show_raw_var, show_sub_dark_var, show_baseline_var`：显示控制变量
- `auto_scale_var, lock_axis`：缩放控制状态
- `config_panel, data_save_panel`：关联面板引用

**核心方法**：
- `update_plot()`：**大循环数据接收点**
- `_process_spectrum_data()`：数据处理核心
- `get_display_mode()`, `get_excitation_wavelength()`：配置查询

**数据处理流程**：
```python
def update_plot(self, spectrum=None, wave_lengths=None):
    """大循环数据接收和处理"""
    # 1. 查询当前配置
    display_mode = self.get_display_mode()
    excitation_wavelength = self.get_excitation_wavelength()
    raman_direction = self.get_raman_direction()
    
    # 2. 处理光谱数据
    processed_data = self._process_spectrum_data(
        spectrum, wave_lengths, display_mode, 
        excitation_wavelength, raman_direction
    )
    
    # 3. 更新显示
    self._update_display(processed_data)
    
    # 4. 传递给数据保存面板
    self.data_save_panel.update_processed_data(**processed_data)
```

### 3. ConfigPanel - 配置管理中心

**功能定位**：系统配置参数管理和提供

**存储数据**：
- `display_mode_var`：显示模式（波长/波数/拉曼位移）
- `excitation_wavelength_var`：激发波长
- `raman_direction_var`：拉曼方向
- `boxcar_width_var`：平滑宽度

**核心方法**：
- `get_display_mode()`, `get_excitation_wavelength()`：配置查询接口
- `get_raman_direction()`, `get_boxcar_width()`：参数获取

**配置提供机制**：
```python
# spectrum_display 查询配置的典型调用
display_mode = self.config_panel.get_display_mode()
excitation_wavelength = self.config_panel.get_excitation_wavelength()
```

### 4. DataSavePanel - 数据存储中心

**功能定位**：处理后数据存储和文件管理

**存储数据**：
- `processed_data`：**主要光谱数据存储接口**
  ```python
  {
      'x_data': None,           # X轴数据（波长/波数/拉曼位移）
      'raw_spectrum': None,     # 原始光谱
      'dark_spectrum': None,    # 暗光谱
      'sub_dark_spectrum': None, # 减暗光谱
      'baseline': None,         # 基线数据
      'display_mode': 'wavelength' # 显示模式
  }
  ```
- `dark_spectra`：暗光谱集合
- `current_dark_index`：当前暗光谱索引

**核心方法**：
- `update_processed_data()`：**大循环数据接收终点**
- `get_processed_spectrum_data()`：数据获取接口
- `save_current_spectrum()`：数据保存功能
- `capture_dark_spectrum()`：暗光谱管理

**数据更新机制**：
```python
def update_processed_data(self, x_data, raw_spectrum, dark_spectrum, 
                         sub_dark_spectrum, baseline, display_mode):
    """接收来自 spectrum_display 的处理后数据"""
    self.processed_data.update({
        'x_data': x_data,
        'raw_spectrum': raw_spectrum,
        'dark_spectrum': dark_spectrum,
        'sub_dark_spectrum': sub_dark_spectrum,
        'baseline': baseline,
        'display_mode': display_mode
    })
```

### 5. ExperimentPanel - 实验流程管理

**功能定位**：实验模式控制和批量测量管理

**存储数据**：
- 实验状态和配置信息
- 数据集管理和变量配置
- 测量会话和结果记录

**核心方法**：
- 实验创建和配置
- 批量测量控制
- 数据集管理

## 系统交互机制

### 1. 回调函数机制

系统通过回调函数实现模块间的松耦合通信：

```python
# spectrometer_panel 启动采集时注册回调
self.start_real_time_acquisition(
    on_update_plot=self.spectrum_display.update_plot
)

# spectrum_display 处理完数据后调用 data_save_panel
self.data_save_panel.update_processed_data(**processed_data)
```

### 2. 配置查询机制

`spectrum_display` 作为数据处理中心，主动查询配置：

```python
# 实时查询当前配置
display_mode = self.config_panel.get_display_mode()
excitation_wavelength = self.config_panel.get_excitation_wavelength()
raman_direction = self.config_panel.get_raman_direction()
```

### 3. 数据传递链

```
原始数据 → 配置查询 → 数据处理 → 结果存储
    ↓         ↓         ↓         ↓
spectrometer → config → spectrum → data_save
   _panel     _panel   _display   _panel
```

## 设计优势

### 1. 模块化设计
- **独立性**：每个模块功能独立，便于测试和维护
- **可扩展性**：新增功能只需添加新模块或扩展现有模块
- **可重用性**：模块可在不同场景下重复使用

### 2. 数据流清晰
- **单向流动**：数据按固定路径流动，避免循环依赖
- **集中处理**：`spectrum_display` 作为数据处理中心
- **统一存储**：`data_save_panel` 作为数据存储中心

### 3. 配置灵活
- **实时配置**：配置变更立即生效
- **集中管理**：所有配置集中在 `config_panel`
- **按需查询**：处理时才查询配置，保证实时性

### 4. 状态管理
- **状态分离**：各模块管理自己的状态
- **状态同步**：通过回调保持状态一致性
- **错误隔离**：单个模块错误不影响其他模块

## 扩展指南

### 添加新的数据处理功能
1. 在 `spectrum_display` 中添加处理方法
2. 在 `config_panel` 中添加相关配置项
3. 更新 `data_save_panel` 的数据结构

### 添加新的设备类型
1. 扩展 `spectrometer_panel` 的设备支持
2. 保持现有接口不变
3. 添加设备特定的配置项

### 添加新的显示模式
1. 在 `config_panel` 中添加模式选项
2. 在 `spectrum_display` 中实现处理逻辑
3. 更新 `data_save_panel` 的数据格式支持

## 性能考虑

### 1. 大循环优化
- **适当延时**：避免CPU占用过高
- **异常处理**：确保循环稳定运行
- **资源管理**：及时释放不需要的资源

### 2. 内存管理
- **数据更新**：及时更新而非累积数据
- **缓存控制**：合理控制缓存大小
- **垃圾回收**：定期清理无用对象

### 3. 界面响应
- **异步处理**：耗时操作异步执行
- **进度反馈**：长时间操作提供进度提示
- **用户体验**：保持界面响应性

## 总结

本光谱仪系统通过**大循环驱动的模块化架构**，实现了：

1. **清晰的数据流**：从设备采集到数据存储的完整链路
2. **灵活的配置管理**：实时配置查询和应用
3. **稳定的实时处理**：持续的数据采集和处理
4. **良好的扩展性**：模块化设计便于功能扩展
5. **优秀的维护性**：职责分离便于代码维护

这种设计思路不仅适用于光谱仪控制，也可以推广到其他需要实时数据处理的科学仪器控制系统中。