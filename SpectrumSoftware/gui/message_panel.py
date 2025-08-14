"""
MessagePanel Class Description:
Function: Unified message display system
Purpose:
- Provides standardized popup message functionality
- Supports different message types (info, success, warning, error)
- Manages auto-closing messages with customizable duration
- Tracks active popups to prevent conflicts
Stored Data:
- Active popup windows list (active_popups)
- Message type color mappings
- Refresh callback functions
"""

import tkinter as tk
from tkinter import ttk

class MessagePanel:
    """统一的消息显示面板类"""
    
    def __init__(self):
        self.active_popups = []  # 跟踪活动的弹窗
    
    def show_auto_close_message(self, parent, message, msg_type="info", duration=2000, refresh_callback=None):
        """显示自动关闭的消息
        
        Args:
            parent: 父组件
            message: 消息内容
            msg_type: 消息类型 ("info", "success", "warning", "error")
            duration: 显示时长(毫秒)
            refresh_callback: 关闭后的刷新回调函数
        """
        # 创建顶层窗口
        popup = tk.Toplevel(parent)
        popup.title("提示")
        popup.geometry("350x120")
        popup.transient(parent)
        popup.grab_set()
        
        # 设置颜色
        colors = {
            "error": "#ffcccc",
            "success": "#ccffcc", 
            "warning": "#ffffcc",
            "info": "#e6f3ff"
        }
        bg_color = colors.get(msg_type, colors["info"])
        popup.configure(bg=bg_color)
        
        # 添加消息标签
        label = tk.Label(popup, text=message, bg=bg_color, wraplength=330, justify="center")
        label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # 添加到活动弹窗列表
        self.active_popups.append(popup)
        
        # 定义关闭时的处理函数
        def close_and_refresh():
            try:
                if popup in self.active_popups:
                    self.active_popups.remove(popup)
                popup.destroy()
                
                # 执行刷新回调
                if refresh_callback:
                    refresh_callback()
                    
                # 通用界面刷新
                self._refresh_interface(parent)
                
            except Exception as e:
                print(f"关闭消息弹窗异常: {e}")
        
        # 自动关闭
        popup.after(duration, close_and_refresh)
        
        # 立即刷新界面
        parent.update_idletasks()
        
        return popup
    
    def _refresh_interface(self, parent):
        """多层次刷新界面"""
        try:
            # 1. 更新当前组件
            parent.update_idletasks()
            
            # 2. 更新父组件
            if hasattr(parent, 'master') and parent.master:
                parent.master.update_idletasks()
            
            # 3. 获取根窗口并强制刷新
            root = parent.winfo_toplevel()
            if root:
                root.update_idletasks()
                root.update()  # 强制处理所有待处理事件
            
            # 4. 触发特定组件刷新
            self._trigger_specific_refresh(parent)
            
        except Exception as e:
            print(f"界面刷新异常: {e}")
    
    def _trigger_specific_refresh(self, parent):
        """触发特定组件的刷新"""
        try:
            # 如果父组件有特定的刷新方法，调用它们
            if hasattr(parent, '_refresh_specific_components'):
                parent._refresh_specific_components()
            
            # 查找并刷新光谱显示组件
            root = parent.winfo_toplevel()
            if root:
                self._refresh_spectrum_display(root)
                
        except Exception as e:
            print(f"特定组件刷新异常: {e}")
    
    def _refresh_spectrum_display(self, root):
        """刷新光谱显示组件"""
        try:
            # 递归查找SpectrumDisplay组件
            def find_spectrum_display(widget):
                if hasattr(widget, '__class__') and 'SpectrumDisplay' in str(widget.__class__):
                    return widget
                for child in widget.winfo_children():
                    result = find_spectrum_display(child)
                    if result:
                        return result
                return None
            
            spectrum_display = find_spectrum_display(root)
            if spectrum_display:
                # 触发光谱显示刷新
                if hasattr(spectrum_display, 'update_display'):
                    spectrum_display.update_display()
                elif hasattr(spectrum_display, 'refresh'):
                    spectrum_display.refresh()
                spectrum_display.update_idletasks()
                
        except Exception as e:
            print(f"光谱显示刷新异常: {e}")
    
    def close_all_messages(self):
        """关闭所有活动的消息弹窗"""
        for popup in self.active_popups[:]:
            try:
                popup.destroy()
            except:
                pass
        self.active_popups.clear()

# 全局消息面板实例
message_panel = MessagePanel()