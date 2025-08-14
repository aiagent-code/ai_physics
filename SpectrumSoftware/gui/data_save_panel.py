"""
DataSavePanel Class Description:
Function: Data management and saving component
Purpose:
- Manages dark spectrum collection and storage
- Stores processed spectrum data received from spectrum_display
- Provides spectrum saving functionality with various formats
- Handles dark spectrum selection and application
Stored Data:
- Processed spectrum data dictionary (processed_data) containing x_data, raw_spectrum, dark_spectrum, sub_dark_spectrum, baseline
- Dark spectra collection (dark_spectra) with metadata
- Current dark spectrum selection index (current_dark_index)
- References to config_panel and spectrometer_panel
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import os
import json
from datetime import datetime
from processor.spectrum_processor import SpectrumProcessor
from .message_panel import message_panel

class DataSavePanel(ttk.LabelFrame):
    def __init__(self, parent,config_panel=None,spectrometer=None):
        super().__init__(parent, text="数据保存", padding=10)
        self.config_panel = config_panel
        self.spectrometer_panel = spectrometer
        
        # 数据存储 - 由SpectrumDisplay更新
        self.processed_data = {
            'x_data': None,
            'raw_spectrum': None,
            'dark_spectrum': None,
            'sub_dark_spectrum': None,
            'baseline': None,
            'display_mode': 'wavelength'
        }
        
        self.dark_spectra = []  # 暗光谱内存区 [{'name':..., 'data':..., 'mode':...}]
        self.current_dark_index = None  # 当前选中暗光谱索引
        self._build_panel()

    def _build_panel(self):
        # 暗光谱管理区
        dark_frame = ttk.LabelFrame(self, text="暗光谱管理", padding=5)
        dark_frame.pack(fill=tk.X, pady=2)
        self.dark_listbox = tk.Listbox(dark_frame, height=4)
        self.dark_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.dark_listbox.bind('<<ListboxSelect>>', self.on_select_dark)
        btn_frame = ttk.Frame(dark_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(btn_frame, text="采集暗光谱", command=self.capture_dark_spectrum).pack(fill=tk.X, pady=1)
#
        ttk.Button(btn_frame, text="删除", command=self.delete_dark_spectrum).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="重命名", command=self.rename_dark_spectrum).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="查看", command=self.view_dark_spectrum).pack(fill=tk.X, pady=1)
        # 保存按钮
        ttk.Button(self, text="保存光谱", command=self.save_current_spectrum).pack(fill=tk.X, pady=6)

    def update_processed_data(self, x_data, raw_spectrum, dark_spectrum, sub_dark_spectrum, baseline, display_mode):
        """由SpectrumDisplay调用，更新处理后的数据"""
        self.processed_data.update({
            'x_data': x_data,
            'raw_spectrum': raw_spectrum,
            'dark_spectrum': dark_spectrum,
            'sub_dark_spectrum': sub_dark_spectrum,
            'baseline': baseline,
            'display_mode': display_mode
        })

    def get_display_mode(self):
        """获取当前显示模式 - 从config_panel获取"""
        if self.config_panel:
            return self.config_panel.display_mode_var.get()
        return 'wavelength'

    def get_raman_direction(self):
        """获取拉曼方向 - 从config_panel获取"""
        if self.config_panel:
            return self.config_panel.raman_direction_var.get()
        return 'positive'

    def capture_dark_spectrum(self):
        """采集暗光谱"""
        # 修复：简化连接状态检查，适配MockSpectrometer
        if not self.spectrometer_panel or not hasattr(self.spectrometer_panel, 'spectrometer') or not self.spectrometer_panel.spectrometer:
            message_panel.show_auto_close_message(
                self, "请先连接光谱仪设备", "warning",
                refresh_callback=self._refresh_specific_components
            )
            return
        
        # 检查是否有可用的光谱数据
        if self.processed_data['raw_spectrum'] is None or self.processed_data['x_data'] is None:
            message_panel.show_auto_close_message(
                self, "请先采集光谱数据", "warning",
                refresh_callback=self._refresh_specific_components
            )
            return
            
        try:
            self.spectrometer_panel.update_status("正在采集暗光谱...")
            
            # 直接使用已处理的数据
            dark_spectrum = self.processed_data['raw_spectrum'].copy()
            wavelengths = self.processed_data['x_data'].copy()
            current_mode = self.processed_data['display_mode']
            print('current_mode:',current_mode)
            if current_mode == 'raman':
                current_mode = 'raman'
            if current_mode == 'wavelength':
                current_mode = 'wavelength'
            '''fuck:很重要'''
            # 获取拉曼方向（仅在拉曼模式下需要）
            raman_direction = self.get_raman_direction() if (current_mode == 'raman' or current_mode == "拉曼位移") else None
            
            name = f"暗光谱{len(self.dark_spectra)+1}"
            
            self.dark_spectra.append({
                'name': name,
                'data': dark_spectrum,
                'mode': current_mode,
                'raman_direction': raman_direction,
                'wavelengths': wavelengths
            })
            self.current_dark_index = len(self.dark_spectra)-1
            self.refresh_dark_list()
            
            self.spectrometer_panel.update_status("暗光谱采集完成")
            self._show_auto_close_message("暗光谱采集完成", "success")
        except Exception as e:
            self.spectrometer_panel.update_status(f"采集暗光谱失败: {str(e)}")
            self._show_auto_close_message(f"采集暗光谱失败: {str(e)}", "error")

    def get_processed_spectrum_data(self):
        """获取处理后的光谱数据 - 直接返回存储的数据"""
        return (
            self.processed_data['x_data'],
            self.processed_data['raw_spectrum'],
            self.processed_data['display_mode']
        )

    def refresh_dark_list(self):
        self.dark_listbox.delete(0, tk.END)
        for d in self.dark_spectra:
            mode = d.get('mode')
            if mode == 'raman':
                direction = d.get('raman_direction', '未知')
                # 将英文方向转换为中文显示
                direction_text = '正' if direction == 'positive' else '负' if direction == 'negative' else direction
                mode_text = f"拉曼({direction_text})"
            else:
                mode_text = "波长"
            self.dark_listbox.insert(tk.END, f"{d['name']} ({mode_text})")
        if self.current_dark_index is not None and 0 <= self.current_dark_index < len(self.dark_spectra):
            self.dark_listbox.select_set(self.current_dark_index)

    def on_select_dark(self, event=None):
        sel = self.dark_listbox.curselection()
        if sel:
            self.current_dark_index = sel[0]
        else:
            self.current_dark_index = None

    '''def add_dark_spectrum(self):
        """添加当前光谱为暗光谱"""
        # 直接使用当前处理后的数据
        if self.processed_data['raw_spectrum'] is None:
            messagebox.showwarning("警告", "当前无光谱可添加为暗光谱")
            return
        
        # 获取当前显示模式
        current_mode = self.get_display_mode() if self.get_display_mode else 'wavelength'
        name = f"暗光谱{len(self.dark_spectra)+1}"
        
        self.dark_spectra.append({
            'name': name, 
            'data': self.processed_data['raw_spectrum'].copy(),
            'mode': current_mode,
            'wavelengths': self.processed_data['x_data'].copy() if self.processed_data['x_data'] is not None else None
        })
        print(self.processed_data['x_data'].copy())
        self.current_dark_index = len(self.dark_spectra)-1
        self.refresh_dark_list()'''

    def delete_dark_spectrum(self):
        if self.current_dark_index is not None:
            del self.dark_spectra[self.current_dark_index]
            self.current_dark_index = None
            self.refresh_dark_list()

    def rename_dark_spectrum(self):
        if self.current_dark_index is not None:
            name = self.dark_spectra[self.current_dark_index]['name']
            new_name = tk.simpledialog.askstring("重命名", f"将 '{name}' 重命名为:")
            if new_name:
                self.dark_spectra[self.current_dark_index]['name'] = new_name
                self.refresh_dark_list()

    def view_dark_spectrum(self):
        if self.current_dark_index is not None:
            import matplotlib.pyplot as plt
            dark_data = self.dark_spectra[self.current_dark_index]
            data = dark_data['data']
            x = dark_data.get('wavelengths', self.get_wavelengths())
            if x is not None:
                plt.figure()
                plt.plot(x, data)
                mode_text = "raman" if dark_data.get('mode') == 'raman' else "wavelength"
                plt.title(f"{dark_data['name']} ({mode_text})")
                plt.xlabel(f'{mode_text} ({"cm⁻¹" if dark_data.get("mode") == "raman" else "nm"})')
                plt.ylabel('absorbence')
                plt.grid(True, alpha=0.3)
                plt.show()

    def get_selected_dark(self, x_data=None, mode=None, raman_direction=None):
        """
        根据当前显示模式、拉曼方向和x轴数据自动选择合适的暗光谱。
        如果x_data给出，则要求暗光谱的x轴与之完全匹配。
        """
        if not self.dark_spectra:
            #print("Warning: No dark spectra stored")
            return None
        if mode is None:
            mode = self.get_display_mode() if self.get_display_mode else 'wavelength'
        # 优先找模式、拉曼方向和x轴都匹配的
        for d in self.dark_spectra:
            if d.get('mode') == mode:
                # 对于拉曼模式，需要匹配方向
                if mode == 'raman' and raman_direction is not None:
                    if d.get('raman_direction') != raman_direction:
                        continue
                
                if x_data is not None and d.get('wavelengths') is not None:
                    if len(d['wavelengths']) == len(x_data) and all(abs(a-b)<1e-6 for a,b in zip(d['wavelengths'], x_data)):
                        return d['data']
                elif x_data is None:
                    return d['data']
                else:
                    print("Warning: x_data is not None but dark spectrum has no wavelengths")
        
        # 其次找模式和拉曼方向匹配的
        for d in self.dark_spectra:
            if d.get('mode') == mode:
                if mode == 'raman' and raman_direction is not None:
                    if d.get('raman_direction') == raman_direction:
                        return d['data']
                else:
                    print("Warning: storged data can not match gived data")
                    return d['data']
                    
        
        return None

    def save_current_spectrum(self, filename=None, apply_baseline=False):
        """保存当前光谱数据"""
        # 直接使用存储的处理后数据
        x_data = self.processed_data['x_data']
        spectrum = self.processed_data['sub_dark_spectrum']
        display_mode = self.processed_data['display_mode']
        
        if spectrum is None or x_data is None:
            # messagebox.showwarning("警告", "没有可保存的光谱数据")
            self._show_auto_close_message("警告: 没有可保存的光谱数据", "warning")
            return None
        
        # 基线去除
        if apply_baseline:
            spectrum, baseline = SpectrumProcessor.baseline_remove_asls(spectrum)
        
        # 文件名处理
        if filename is None:
            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if not filename:
            return None
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                import csv
                writer = csv.writer(file)
                x_label = '波长(nm)' if display_mode == 'wavelength' else 'raman_shift(cm⁻¹)'
                writer.writerow([x_label, '强度'])
                for w, i in zip(x_data, spectrum):
                    writer.writerow([w, i])
            # messagebox.showinfo("成功", f"光谱数据保存成功\n{filename}")
            self._show_auto_close_message(f"光谱数据保存成功\n{filename}", "success")
            return filename
        except Exception as e:
            # messagebox.showerror("错误", f"保存失败: {str(e)}")
            self._show_auto_close_message(f"保存失败: {str(e)}", "error")
            return None
            
    def _show_auto_close_message(self, message, msg_type="info", duration=2000):
        """显示自动关闭的消息"""
        import tkinter as tk
        
        # 创建顶层窗口
        popup = tk.Toplevel(self)
        popup.title("提示")
        popup.geometry("350x120")
        popup.transient(self)
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
        
        # 定义关闭时的刷新函数
        def close_and_refresh():
            popup.destroy()
            # 多层次刷新界面
            self._refresh_interface()
        
        # 自动关闭
        popup.after(duration, close_and_refresh)
        
        # 立即刷新界面
        self.update_idletasks()
        
    def _refresh_interface(self):
        """刷新界面的多层次方法"""
        try:
            # 1. 更新当前组件
            self.update_idletasks()
            
            # 2. 更新父组件
            if self.master:
                self.master.update_idletasks()
            
            # 3. 获取根窗口并强制刷新
            root = self.winfo_toplevel()
            if root:
                root.update_idletasks()
                root.update()  # 强制处理所有待处理事件
            
            # 4. 如果有特定的刷新需求，调用相关方法
            self._refresh_specific_components()
            
        except Exception as e:
            print(f"界面刷新异常: {e}")
    
    def _refresh_specific_components(self):
        """刷新特定组件（子类可重写）"""
        # 刷新暗光谱列表
        if hasattr(self, 'refresh_dark_list'):
            try:
                self.refresh_dark_list()
            except:
                pass