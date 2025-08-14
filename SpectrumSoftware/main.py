"""
主程序入口
功能：协调光谱仪UI和MCP服务器的启动和通信
作用：管理多线程运行，处理程序退出信号
"""

import threading
import queue
import time
import signal
import sys
from spectrum_main import MainApp
from mcp_main import MCPServer
import matplotlib.pyplot as plt

# 全局变量
ui_command_queue = queue.Queue()
notification_queue = queue.Queue()
main_app = None
mcp_server = None
ui_ready = threading.Event()
shutdown_event = threading.Event()
ui_thread = None
mcp_thread = None

def run_ui():
    """运行UI线程"""
    global main_app
    try:
        main_app = MainApp()
        main_app.setup_mcp_communication(ui_command_queue, notification_queue)
        ui_ready.set()  # 标记UI已准备就绪
        main_app.run()
    except Exception as e:
        print(f"UI线程异常: {e}")
        import traceback
        traceback.print_exc()
        # 给用户时间查看错误信息
        print("UI线程将在5秒后退出...")
        time.sleep(5)
    finally:
        print("UI线程已退出")
        shutdown_event.set()

def run_mcp():
    """运行MCP服务器线程"""
    global mcp_server
    try:
        mcp_server = MCPServer()
        
        # 等待UI初始化完成
        print("等待UI初始化完成...")
        ui_ready.wait()  # 等待UI线程设置ui_ready事件
        
        # 确保main_app已经创建
        while main_app is None:
            time.sleep(0.1)
        
        print("UI已就绪，设置MCP通信...")
        mcp_server.setup_communication(ui_command_queue, notification_queue, main_app, ui_ready)
        mcp_server.run_server()
    except Exception as e:
        print(f"MCP服务器异常: {e}")
        import traceback
        traceback.print_exc()
        print("MCP服务器将在3秒后退出...")
        time.sleep(3)
    finally:
        print("MCP服务器线程已退出")
        shutdown_event.set()
def monitor_threads():
    """监控线程状态"""
    global ui_thread, mcp_thread
    
    while not shutdown_event.is_set():
        try:
            # 检查UI线程状态
            if ui_thread and not ui_thread.is_alive():
                print("检测到UI线程已退出，等待5秒后关闭MCP服务器...")
                time.sleep(5)  # 给用户时间查看错误
                shutdown_event.set()
                break
                
            # 检查MCP服务器线程状态
            if mcp_thread and not mcp_thread.is_alive():
                print("检测到MCP服务器线程已退出，等待3秒后关闭UI...")
                time.sleep(3)
                shutdown_event.set()
                break
                
            time.sleep(1)  # 每秒检查一次
        except Exception as e:
            print(f"线程监控异常: {e}")
            break

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n接收到信号 {signum}，正在关闭程序...")
    shutdown_event.set()
    # 给线程时间优雅关闭
    time.sleep(2)
    sys.exit(0)

def graceful_shutdown():
    """优雅关闭程序"""
    print("正在关闭所有线程...")
    
    # 设置关闭事件
    shutdown_event.set()
    
    # 强制关闭UI窗口
    if main_app and hasattr(main_app, 'root'):
        try:
            main_app.root.quit()
            main_app.root.destroy()
        except:
            pass
    
    # 等待线程结束，增加超时时间
    if ui_thread and ui_thread.is_alive():
        print("等待UI线程结束...")
        ui_thread.join(timeout=5)  # 减少超时时间
        if ui_thread.is_alive():
            print("警告：UI线程未能在5秒内结束")
    
    if mcp_thread and mcp_thread.is_alive():
        print("等待MCP服务器线程结束...")
        mcp_thread.join(timeout=5)  # 减少超时时间
        if mcp_thread.is_alive():
            print("警告：MCP服务器线程未能在5秒内结束")
    
    print("所有线程已关闭")
    
    # 强制退出程序
    import os
    os._exit(0)

if __name__ == "__main__":
    print("启动光谱仪MCP控制服务器...")
    print("提示：如果遇到端口占用错误，请检查是否有其他程序占用了相关端口")
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 启动UI线程
        ui_thread = threading.Thread(target=run_ui, daemon=False)
        ui_thread.start()
        
        # 启动MCP服务器线程
        mcp_thread = threading.Thread(target=run_mcp, daemon=False)
        mcp_thread.start()
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
        monitor_thread.start()
        
        # 等待关闭信号
        shutdown_event.wait()
        
        # 优雅关闭
        graceful_shutdown()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        graceful_shutdown()
    except Exception as e:
        print(f"主程序异常: {e}")
        import traceback
        traceback.print_exc()
        graceful_shutdown()
    finally:
        print("程序已完全退出")
        # 移除这个sleep，直接退出
        import os
        os._exit(0)
