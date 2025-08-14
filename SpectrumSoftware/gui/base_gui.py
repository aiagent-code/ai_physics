import tkinter as tk
from tkinter import ttk

class BaseGUI:
    def __init__(self, title="OceanDirect 光谱仪软件", size="1200x800"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(size)
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._running = True
    
    def on_closing(self):
        """窗口关闭时的处理函数"""
        self._running = False
        self.root.quit()  # 退出mainloop
        self.root.destroy()  # 销毁窗口

    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"GUI运行异常: {e}")
        finally:
            self._running = False
            # 确保窗口被销毁
            try:
                if self.root.winfo_exists():
                    self.root.destroy()
            except:
                pass