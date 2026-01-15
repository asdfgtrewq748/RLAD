"""
修复版RLAD: 基于强化学习与GUI交互式人工标注的时间序列异常检测
专用于液压支架工作阻力异常检测 - 修复按钮显示问题
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import deque, namedtuple
import random
import warnings
import argparse
import traceback
import threading
import queue
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# GUI相关导入
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100

# 忽略警告
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    """设置随机种子保证可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def convert_to_serializable(obj):
    """将numpy/torch对象转换为可JSON序列化的格式"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    else:
        try:
            return str(obj) if not isinstance(obj, (int, float, bool, str, type(None))) else obj
        except Exception:
            return f"Unserializable object: {type(obj)}"

# =================================
# 修复的GUI标注界面
# =================================

class AnnotationGUI:
    """修复版的基于Tkinter的可视化标注界面"""
    
    def __init__(self, window_size=288):
        self.window_size = window_size
        self.result = None
        self.root = None
        self.current_window_data = None
        self.current_original_data = None
        self.window_idx = None
        self.auto_prediction = None
        
    def create_gui(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """创建GUI界面"""
        try:
            print(f"开始创建GUI窗口 - 窗口 #{window_idx}")
            
            # 确保之前的窗口已关闭
            if self.root is not None:
                try:
                    self.root.destroy()
                except:
                    pass
                    
            self.current_window_data = window_data
            self.current_original_data = original_data_segment
            self.window_idx = window_idx
            self.auto_prediction = auto_predicted_label
            self.result = None
            
            # 创建主窗口
            self.root = tk.Tk()
            self.root.title(f"液压支架异常检测 - 窗口 #{window_idx} 标注")
            self.root.geometry("1200x900")
            self.root.configure(bg='#f0f0f0')
            
            # 设置窗口居中
            self.root.update_idletasks()
            x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
            y = (self.root.winfo_screenheight() // 2) - (900 // 2)
            self.root.geometry(f"1200x900+{x}+{y}")
            
            # 设置窗口属性
            self.root.resizable(True, True)
            self.root.minsize(1000, 700)
            
            # 强制窗口显示在最前面
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            
            print(f"主窗口创建成功")
            
            # 创建界面组件
            self.create_widgets()
            
            # 绑定关闭事件
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            # 确保窗口获得焦点
            self.root.focus_force()
            
            print(f"GUI窗口创建完成 - 窗口 #{window_idx}")
            return self.root
            
        except Exception as e:
            print(f"创建GUI窗口时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_widgets(self):
        """创建界面组件"""
        try:
            print("开始创建界面组件...")
            
            # 创建主容器（使用Frame而不是ttk.Frame以确保兼容性）
            main_container = tk.Frame(self.root, bg='#f0f0f0')
            main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 1. 标题区域
            title_frame = tk.Frame(main_container, bg='#f0f0f0')
            title_frame.pack(fill=tk.X, pady=(0, 10))
            
            title_label = tk.Label(title_frame, 
                                  text=f"窗口 #{self.window_idx} 异常检测标注", 
                                  font=('Arial', 18, 'bold'),
                                  bg='#f0f0f0',
                                  fg='#2c3e50')
            title_label.pack()
            
            print("标题区域创建完成")
            
            # 2. 信息显示区域
            info_frame = tk.LabelFrame(main_container, 
                                      text="窗口信息", 
                                      font=('Arial', 12, 'bold'),
                                      bg='#f0f0f0',
                                      fg='#34495e',
                                      padx=10, pady=10)
            info_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.create_info_display(info_frame)
            print("信息显示区域创建完成")
            
            # 3. 图表区域（给予更多空间）
            chart_frame = tk.LabelFrame(main_container, 
                                       text="数据可视化", 
                                       font=('Arial', 12, 'bold'),
                                       bg='#f0f0f0',
                                       fg='#34495e',
                                       padx=5, pady=5)
            chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
            
            self.create_charts(chart_frame)
            print("图表区域创建完成")
            
            # 4. 按钮区域（最重要的部分）
            button_frame = tk.Frame(main_container, bg='#f0f0f0')
            button_frame.pack(fill=tk.X, side=tk.BOTTOM)
            
            self.create_buttons(button_frame)
            print("按钮区域创建完成")
            
            # 强制更新界面
            self.root.update_idletasks()
            self.root.update()
            
            print("所有界面组件创建完成")
            
        except Exception as e:
            print(f"创建界面组件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def create_info_display(self, parent):
        """创建信息显示区域"""
        try:
            # 创建水平布局
            info_container = tk.Frame(parent, bg='#f0f0f0')
            info_container.pack(fill=tk.X, expand=True)
            
            # 左侧基本信息
            left_frame = tk.Frame(info_container, bg='#f0f0f0')
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
            
            tk.Label(left_frame, text="基本信息:", 
                    font=('Arial', 12, 'bold'), 
                    bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W, pady=(0, 5))
            
            tk.Label(left_frame, text=f"窗口索引: {self.window_idx}", 
                    font=('Arial', 10), 
                    bg='#f0f0f0', fg='#34495e').pack(anchor=tk.W)
            
            tk.Label(left_frame, text=f"窗口大小: {self.window_size}", 
                    font=('Arial', 10), 
                    bg='#f0f0f0', fg='#34495e').pack(anchor=tk.W)
            
            # AI预测信息
            if self.auto_prediction is not None:
                prediction_text = "异常" if self.auto_prediction == 1 else "正常"
                color = "#e74c3c" if self.auto_prediction == 1 else "#27ae60"
                tk.Label(left_frame, text=f"AI预测: {prediction_text}", 
                        font=('Arial', 11, 'bold'), 
                        bg='#f0f0f0', fg=color).pack(anchor=tk.W, pady=(5, 0))
            
            # 右侧统计信息
            right_frame = tk.Frame(info_container, bg='#f0f0f0')
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            if self.current_original_data is not None:
                tk.Label(right_frame, text="统计信息:", 
                        font=('Arial', 12, 'bold'), 
                        bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W, pady=(0, 5))
                
                data_flat = self.current_original_data.flatten()
                mean_val = np.mean(data_flat)
                max_val = np.max(data_flat)
                min_val = np.min(data_flat)
                std_val = np.std(data_flat)
                
                stats_info = [
                    f"均值: {mean_val:.2f}",
                    f"最大值: {max_val:.2f}",
                    f"最小值: {min_val:.2f}",
                    f"标准差: {std_val:.2f}"
                ]
                
                for stat in stats_info:
                    tk.Label(right_frame, text=stat, 
                            font=('Arial', 10), 
                            bg='#f0f0f0', fg='#34495e').pack(anchor=tk.W)
                            
        except Exception as e:
            print(f"创建信息显示时出错: {e}")
    
    def create_charts(self, parent):
        """创建数据可视化图表"""
        try:
            print("开始创建图表...")
            
            # 创建matplotlib图形
            fig = Figure(figsize=(11, 5), dpi=80)
            fig.patch.set_facecolor('#f0f0f0')
            
            # 第一个子图：标准化数据
            ax1 = fig.add_subplot(211)
            data_to_plot = self.current_window_data.flatten()
            time_steps = np.arange(len(data_to_plot))
            
            ax1.plot(time_steps, data_to_plot, 'b-', linewidth=1.5, alpha=0.8, label='标准化数据')
            ax1.set_title(f'窗口 #{self.window_idx} - 标准化数据', fontsize=11, fontweight='bold')
            ax1.set_xlabel('时间步')
            ax1.set_ylabel('标准化值')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 第二个子图：原始数据
            ax2 = fig.add_subplot(212)
            if self.current_original_data is not None:
                original_data_flat = self.current_original_data.flatten()
                ax2.plot(time_steps, original_data_flat, 'r-', linewidth=1.5, alpha=0.8, label='原始数据')
                ax2.set_title(f'窗口 #{self.window_idx} - 原始阻力数据', fontsize=11, fontweight='bold')
                ax2.set_xlabel('时间步')
                ax2.set_ylabel('阻力值')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # 添加异常阈值线
                try:
                    mean_val = np.mean(original_data_flat)
                    std_val = np.std(original_data_flat)
                    threshold = mean_val + 2 * std_val
                    ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                               label=f'阈值线 (μ+2σ)')
                    ax2.legend()
                except:
                    pass
            else:
                ax2.text(0.5, 0.5, '原始数据不可用', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14, color='gray')
                ax2.set_title(f'窗口 #{self.window_idx} - 原始数据不可用', fontsize=11)
            
            fig.tight_layout(pad=2.0)
            
            # 将图表嵌入到Tkinter中
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            print("图表创建完成")
            
        except Exception as e:
            print(f"创建图表时出错: {e}")
            # 创建错误显示
            error_label = tk.Label(parent, text=f"图表显示错误: {str(e)}", 
                                  font=('Arial', 12), fg='red', bg='#f0f0f0')
            error_label.pack(expand=True)
    
    def create_buttons(self, parent):
        """创建标注按钮 - 重点修复区域"""
        try:
            print("开始创建按钮...")
            
            # 主按钮容器
            button_container = tk.Frame(parent, bg='#f0f0f0')
            button_container.pack(fill=tk.X, pady=15)
            
            # 标题
            title_label = tk.Label(button_container, 
                                  text="请选择标注结果:", 
                                  font=('Arial', 16, 'bold'),
                                  bg='#f0f0f0',
                                  fg='#2c3e50')
            title_label.pack(pady=(0, 20))
            
            # 主要标注按钮行
            main_buttons_frame = tk.Frame(button_container, bg='#f0f0f0')
            main_buttons_frame.pack(pady=10)
            
            # 正常按钮
            normal_btn = tk.Button(main_buttons_frame, 
                                  text="正常 (0)", 
                                  font=('Arial', 14, 'bold'),
                                  bg='#27ae60',
                                  fg='white',
                                  activebackground='#2ecc71',
                                  activeforeground='white',
                                  width=15,
                                  height=2,
                                  command=lambda: self.set_result(0),
                                  relief=tk.RAISED,
                                  bd=3)
            normal_btn.pack(side=tk.LEFT, padx=30)
            
            # 异常按钮
            anomaly_btn = tk.Button(main_buttons_frame, 
                                   text="异常 (1)", 
                                   font=('Arial', 14, 'bold'),
                                   bg='#e74c3c',
                                   fg='white',
                                   activebackground='#c0392b',
                                   activeforeground='white',
                                   width=15,
                                   height=2,
                                   command=lambda: self.set_result(1),
                                   relief=tk.RAISED,
                                   bd=3)
            anomaly_btn.pack(side=tk.LEFT, padx=30)
            
            # 辅助按钮行
            aux_buttons_frame = tk.Frame(button_container, bg='#f0f0f0')
            aux_buttons_frame.pack(pady=15)
            
            # 跳过按钮
            skip_btn = tk.Button(aux_buttons_frame, 
                                text="跳过此窗口", 
                                font=('Arial', 12),
                                bg='#f39c12',
                                fg='white',
                                activebackground='#e67e22',
                                activeforeground='white',
                                width=12,
                                height=1,
                                command=lambda: self.set_result(-1),
                                relief=tk.RAISED,
                                bd=2)
            skip_btn.pack(side=tk.LEFT, padx=15)
            
            # 退出按钮
            exit_btn = tk.Button(aux_buttons_frame, 
                                text="退出标注", 
                                font=('Arial', 12),
                                bg='#95a5a6',
                                fg='white',
                                activebackground='#7f8c8d',
                                activeforeground='white',
                                width=12,
                                height=1,
                                command=lambda: self.set_result(-2),
                                relief=tk.RAISED,
                                bd=2)
            exit_btn.pack(side=tk.LEFT, padx=15)
            
            # 快捷键说明
            shortcut_frame = tk.Frame(button_container, bg='#f0f0f0')
            shortcut_frame.pack(pady=15)
            
            shortcut_label = tk.Label(shortcut_frame, 
                                     text="快捷键: 0=正常, 1=异常, S=跳过, Q/Esc=退出", 
                                     font=('Arial', 10), 
                                     bg='#f0f0f0',
                                     fg='#7f8c8d')
            shortcut_label.pack()
            
            # 绑定键盘事件
            self.root.bind('<Key-0>', lambda e: self.set_result(0))
            self.root.bind('<Key-1>', lambda e: self.set_result(1))
            self.root.bind('<KeyPress-s>', lambda e: self.set_result(-1))
            self.root.bind('<KeyPress-S>', lambda e: self.set_result(-1))
            self.root.bind('<KeyPress-q>', lambda e: self.set_result(-2))
            self.root.bind('<KeyPress-Q>', lambda e: self.set_result(-2))
            self.root.bind('<Escape>', lambda e: self.set_result(-2))
            
            # 确保窗口可以接收键盘焦点
            self.root.focus_set()
            
            # 强制更新界面
            button_container.update_idletasks()
            self.root.update_idletasks()
            
            print(f"按钮创建完成！窗口 #{self.window_idx}")
            print(f"按钮容器: {button_container}")
            print(f"主按钮框架: {main_buttons_frame}")
            print(f"正常按钮: {normal_btn}")
            print(f"异常按钮: {anomaly_btn}")
            
        except Exception as e:
            print(f"创建按钮时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建紧急按钮（简化版本）
            emergency_frame = tk.Frame(parent, bg='#f0f0f0')
            emergency_frame.pack(fill=tk.X, pady=20)
            
            tk.Label(emergency_frame, text="按钮创建失败，请使用以下紧急按钮:", 
                    font=('Arial', 12), bg='#f0f0f0', fg='red').pack(pady=10)
            
            tk.Button(emergency_frame, text="正常", command=lambda: self.set_result(0), 
                     width=10).pack(side=tk.LEFT, padx=10)
            tk.Button(emergency_frame, text="异常", command=lambda: self.set_result(1), 
                     width=10).pack(side=tk.LEFT, padx=10)
            tk.Button(emergency_frame, text="退出", command=lambda: self.set_result(-2), 
                     width=10).pack(side=tk.LEFT, padx=10)
    
    def set_result(self, result):
        """设置标注结果并关闭窗口"""
        try:
            print(f"用户选择了结果: {result}")
            
            if result in [0, 1]:
                # 确认对话框
                label_text = "异常" if result == 1 else "正常"
                confirm = messagebox.askyesno("确认标注", 
                                            f"确认将窗口 #{self.window_idx} 标注为 '{label_text}' 吗？",
                                            parent=self.root)
                if not confirm:
                    print("用户取消了标注")
                    return
            elif result == -1:
                confirm = messagebox.askyesno("确认跳过", 
                                            "确认跳过此窗口的标注吗？",
                                            parent=self.root)
                if not confirm:
                    print("用户取消了跳过")
                    return
            elif result == -2:
                confirm = messagebox.askyesno("确认退出", 
                                            "确认退出标注程序吗？",
                                            parent=self.root)
                if not confirm:
                    print("用户取消了退出")
                    return
            
            self.result = result
            print(f"标注结果已设置: {result}")
            
            # 关闭窗口
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"设置结果时出错: {e}")
            # 即使出错也要设置结果
            self.result = result if result is not None else -2
            if self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
    
    def on_close(self):
        """窗口关闭事件"""
        try:
            confirm = messagebox.askyesno("确认退出", 
                                        "确认关闭标注界面吗？这将退出标注程序。",
                                        parent=self.root)
            if confirm:
                self.result = -2
                self.root.quit()
                self.root.destroy()
        except:
            self.result = -2
            if self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
    
    def get_annotation(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """获取用户标注（主要接口）"""
        try:
            print(f"开始获取标注 - 窗口 #{window_idx}")
            root = self.create_gui(window_data, window_idx, original_data_segment, auto_predicted_label)
            if root is None:
                print("GUI创建失败，使用命令行模式")
                return self.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
            
            print("开始主循环...")
            root.mainloop()
            print(f"主循环结束，结果: {self.result}")
            
            return self.result if self.result is not None else -2
            
        except Exception as e:
            print(f"GUI标注失败: {e}")
            print("回退到命令行模式")
            return self.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
    
    def get_annotation_fallback(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """回退到命令行标注模式"""
        print(f"\n{'='*60}")
        print(f"请对窗口 #{window_idx} 进行标注")
        print(f"{'='*60}")
        
        if original_data_segment is not None:
            print(f"原始数据统计:")
            print(f"  均值: {np.mean(original_data_segment):.2f}")
            print(f"  最大值: {np.max(original_data_segment):.2f}")
            print(f"  最小值: {np.min(original_data_segment):.2f}")
            print(f"  标准差: {np.std(original_data_segment):.2f}")
        
        if auto_predicted_label is not None:
            label_text = "异常" if auto_predicted_label == 1 else "正常"
            print(f"  AI预测: {label_text} (标签={auto_predicted_label})")
        
        while True:
            try:
                print("\n请选择标注:")
                print("  0 = 正常")
                print("  1 = 异常")
                print("  s = 跳过此窗口")
                print("  q = 退出标注程序")
                
                choice = input("您的选择 (0/1/s/q): ").strip().lower()
                
                if choice == 'q':
                    return -2
                elif choice == 's':
                    return -1
                elif choice in ['0', '1']:
                    return int(choice)
                else:
                    print("无效的选择，请输入 0, 1, s 或 q")
                    
            except KeyboardInterrupt:
                print("\n\n用户中断标注程序。")
                return -2
            except Exception as e:
                print(f"输入错误: {e}")
                continue

# 继续使用您原有的其他类和函数，只需要替换AnnotationGUI部分...
# [其他代码保持不变]

# =================================
# 2. 改进的人工标注系统
# =================================

class HumanAnnotationSystem:
    """人工标注系统，集成GUI界面"""
    
    def __init__(self, output_dir: str, window_size: int = 288, use_gui: bool = True):
        self.output_dir = output_dir
        self.window_size = window_size
        self.use_gui = use_gui
        self.annotation_history = []
        self.manual_labels_file = os.path.join(output_dir, 'manual_annotations.json')
        self.gui = AnnotationGUI(window_size) if use_gui else None
        self.load_existing_annotations()
        
    def load_existing_annotations(self):
        """加载已存在的人工标注"""
        if os.path.exists(self.manual_labels_file):
            try:
                with open(self.manual_labels_file, 'r', encoding='utf-8') as f:
                    self.annotation_history = json.load(f)
                print(f"已加载 {len(self.annotation_history)} 条历史标注记录")
            except Exception as e:
                print(f"加载历史标注记录时出错: {e}")
                self.annotation_history = []
    
    def save_annotations(self):
        """保存标注历史到文件"""
        try:
            with open(self.manual_labels_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotation_history, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存标注记录时出错: {e}")
    
    def get_human_annotation(self, window_data: np.ndarray, window_idx: int, 
                           original_data_segment: np.ndarray = None, 
                           auto_predicted_label: int = None) -> int:
        """获取用户对单个窗口的标注"""
        
        # 检查是否已经标注过这个窗口
        for record in self.annotation_history:
            if record.get('window_idx') == window_idx:
                print(f"窗口 #{window_idx} 已被标注为: {record['label']} ({'异常' if record['label'] == 1 else '正常'})")
                return record['label']
        
        # 使用GUI或命令行获取标注
        if self.use_gui and self.gui:
            try:
                label = self.gui.get_annotation(window_data, window_idx, original_data_segment, auto_predicted_label)
            except Exception as e:
                print(f"GUI标注失败，使用命令行模式: {e}")
                label = self.gui.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
        else:
            label = self.get_annotation_cmdline(window_data, window_idx, original_data_segment, auto_predicted_label)
        
        # 保存标注结果
        if label not in [-1, -2]:  # 有效标注
            annotation_record = {
                'window_idx': window_idx,
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'auto_predicted_label': auto_predicted_label,
                'window_stats': {
                    'mean': float(np.mean(original_data_segment)) if original_data_segment is not None else None,
                    'max': float(np.max(original_data_segment)) if original_data_segment is not None else None,
                    'min': float(np.min(original_data_segment)) if original_data_segment is not None else None,
                    'std': float(np.std(original_data_segment)) if original_data_segment is not None else None
                }
            }
            self.annotation_history.append(annotation_record)
            self.save_annotations()
            
            label_text = "异常" if label == 1 else "正常"
            print(f"已标注窗口 #{window_idx} 为: {label_text}")
        
        return label
    
    def get_annotation_cmdline(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """命令行标注模式（回退方案）"""
        print(f"\n{'='*60}")
        print(f"请对窗口 #{window_idx} 进行标注")
        print(f"{'='*60}")
        
        if original_data_segment is not None:
            print(f"原始数据统计:")
            print(f"  均值: {np.mean(original_data_segment):.2f}")
            print(f"  最大值: {np.max(original_data_segment):.2f}")
            print(f"  最小值: {np.min(original_data_segment):.2f}")
            print(f"  标准差: {np.std(original_data_segment):.2f}")
        
        if auto_predicted_label is not None:
            label_text = "异常" if auto_predicted_label == 1 else "正常"
            print(f"  AI预测: {label_text} (标签={auto_predicted_label})")
        
        while True:
            try:
                print("\n请选择标注:")
                print("  0 = 正常")
                print("  1 = 异常")
                print("  s = 跳过此窗口")
                print("  q = 退出标注程序")
                
                choice = input("您的选择 (0/1/s/q): ").strip().lower()
                
                if choice == 'q':
                    return -2
                elif choice == 's':
                    return -1
                elif choice in ['0', '1']:
                    return int(choice)
                else:
                    print("无效的选择，请输入 0, 1, s 或 q")
                    
            except KeyboardInterrupt:
                print("\n\n用户中断标注程序。")
                return -2
            except Exception as e:
                print(f"输入错误: {e}")
                continue

# =================================
# 3. 数据集和模型（保持与原代码相同）
# =================================

class HydraulicDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        window = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            window = self.transform(window)
            
        return window, label

def load_hydraulic_data_improved(data_path, window_size=288, stride=12, specific_feature_column=None):
    """
    改进的数据加载函数
    初步异常标签基于选定列数据的"来压判据" (Q3 + 0.5 * IQR)
    """
    print(f"正在加载数据: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")
    
    actual_selected_column_name = None 

    if specific_feature_column:
        if specific_feature_column in df.columns:
            actual_selected_column_name = specific_feature_column
            print(f"已指定并选择的支架列: {actual_selected_column_name}")
        else:
            print(f"警告: 指定的特征列 '{specific_feature_column}' 不在CSV文件中。将尝试自动选择。")

    if not actual_selected_column_name: 
        support_columns = [col for col in df.columns if '#' in col]
        if len(support_columns) > 0:
            actual_selected_column_name = support_columns[0] 
            print(f"自动选择的支架列 (第一个含'#'的列): {actual_selected_column_name}")
        else:
            print("警告: 未在列名中找到 '#'。将尝试使用第一个数值列作为支架列。")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                actual_selected_column_name = numeric_cols[0]
                print(f"自动选择的数值列: {actual_selected_column_name}")
            else:
                if specific_feature_column:
                     raise ValueError(f"指定的特征列 '{specific_feature_column}' 未找到，且无法自动选择其他有效列。")
                else:
                     raise ValueError("未找到包含 '#' 的列，且数据中没有数值列，也未指定有效的特征列。")
    
    selected_cols = [actual_selected_column_name] 
    print(f"最终选择的列数: {len(selected_cols)}, 列名: {selected_cols}")

    if 'Date' not in df.columns:
        print("警告: 'Date' 列未在CSV中找到。将使用索引作为日期替代。")
        df['Date'] = df.index.astype(str)
        
    df_for_point_mapping = df[['Date'] + selected_cols].copy()
    
    data_values = df[selected_cols].fillna(method='ffill').fillna(method='bfill').fillna(0).values
    print(f"提取的用于模型训练和阈值计算的数据形状: {data_values.shape}")
    
    # 基于选定列计算"来压判据"阈值
    all_points_for_thresholds = data_values.flatten()
    laiya_criterion_point = np.inf 

    if len(all_points_for_thresholds) > 0:
        q1_point = np.percentile(all_points_for_thresholds, 25)
        q3_point = np.percentile(all_points_for_thresholds, 75)
        iqr_point = q3_point - q1_point
        
        laiya_criterion_point = q3_point + 0.5 * iqr_point 
        
        print(f"逐点异常阈值计算 (基于列 '{actual_selected_column_name}'):")
        print(f"  Q1={q1_point:.2f}, Q3={q3_point:.2f}, IQR={iqr_point:.2f}")
        print(f"  计算得到的 来压判据阈值: {laiya_criterion_point:.2f}")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    print("创建滑动窗口...")
    windows_scaled_list = []
    windows_raw_data_list = []  # 保存原始数据用于人工标注
    windows_raw_max_list = [] 
    window_start_indices_all = []

    for i in range(0, len(data_scaled) - window_size + 1, stride):
        window_s = data_scaled[i:i + window_size]
        window_r = data_values[i:i + window_size]  # 原始数据窗口
        windows_scaled_list.append(window_s)
        windows_raw_data_list.append(window_r)
        
        window_r_points = window_r.flatten()
        if len(window_r_points) > 0:
            windows_raw_max_list.append(np.max(window_r_points))
        else:
            windows_raw_max_list.append(-np.inf) 
            
        window_start_indices_all.append(i)
    
    if not windows_scaled_list:
        raise ValueError(f"未能创建滑动窗口。数据长度 {len(data_scaled)}, 窗口大小 {window_size}。请检查参数。")

    X = np.array(windows_scaled_list)
    windows_raw_data = np.array(windows_raw_data_list)  # 原始数据窗口
    windows_raw_max_np = np.array(windows_raw_max_list)
    window_start_indices_all_np = np.array(window_start_indices_all)
    
    N = len(X)
    if N == 0:
        raise ValueError("未能创建滑动窗口。请检查 window_size, stride, 和数据长度。")
    
    # 基于"来压判据"的打标签逻辑
    y = np.zeros(N, dtype=int) 
    num_anomalies_found_by_laiya = 0 
    for i in range(N):
        max_val_in_raw_window = windows_raw_max_np[i]
        if max_val_in_raw_window > laiya_criterion_point: 
            y[i] = 1 
            num_anomalies_found_by_laiya += 1
    
    print(f"初步窗口标签统计:")
    print(f"  通过来压判据标记的异常窗口数: {num_anomalies_found_by_laiya}")
    print(f"  总计初步异常窗口数 (y==1): {np.sum(y==1)}")
    print(f"  总计初步正常窗口数 (y==0): {np.sum(y==0)}")

    # 设置一部分为未标注状态，供人工标注
    unlabeled_mask = np.random.random(N) < 0.05  # 5%设为未标注
    y[unlabeled_mask] = -1 
    
    print(f"设置未标注后的标签分布: 正常={np.sum(y==0)}, 异常={np.sum(y==1)}, 未标注={np.sum(y==-1)}")
    
    train_size = int(0.7 * N)
    val_size = int(0.15 * N)
    
    indices = np.arange(N)
    np.random.shuffle(indices)

    X_train = X[indices[:train_size]]
    y_train = y[indices[:train_size]]
    windows_raw_train = windows_raw_data[indices[:train_size]]  # 训练集原始数据窗口
    
    X_val = X[indices[train_size:train_size + val_size]]
    y_val = y[indices[train_size:train_size + val_size]]
    
    X_test = X[indices[train_size + val_size:]]
    y_test = y[indices[train_size + val_size:]]

    test_window_original_indices = window_start_indices_all_np[indices[train_size + val_size:]]
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, \
           test_window_original_indices, df_for_point_mapping, actual_selected_column_name, \
           windows_raw_train  # 返回训练集的原始数据窗口
# PART 2/2

# =================================
# 4. 网络结构（保持不变）
# =================================

class EnhancedRLADAgent(nn.Module):
    """增强版RLAD智能体网络架构"""
    
    def __init__(self, input_dim, seq_len=288, hidden_size=128, num_layers=3):
        super(EnhancedRLADAgent, self).__init__()
        
        self.input_dim = input_dim 
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        self.ln_attention = nn.LayerNorm(lstm_output_size)
        
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 2)
        
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name :
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        
    def forward(self, x, return_features=False, return_attention_weights=False):
        lstm_out, (hidden, cell) = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out) 
        lstm_out = self.ln_attention(lstm_out + attn_out) 
        avg_pool = torch.mean(lstm_out, dim=1) 
        max_pool, _ = torch.max(lstm_out, dim=1) 
        combined_pool = avg_pool + max_pool 
        
        x_fc = self.fc1(combined_pool)
        x_fc = self.ln1(x_fc)
        x_fc = self.gelu(x_fc)
        x_fc = self.dropout(x_fc)
        
        features = x_fc 
        
        x_fc = self.fc2(x_fc)
        x_fc = self.ln2(x_fc)
        x_fc = self.gelu(x_fc)
        x_fc = self.dropout(x_fc)
        
        q_values = self.fc3(x_fc) 
        
        if return_features:
            if return_attention_weights:
                return q_values, features, attn_weights
            return q_values, features
        else:
            if return_attention_weights:
                return q_values, attn_weights
            return q_values
    
    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 1) 
        else:
            was_training = self.training
            self.eval() 
            with torch.no_grad():
                if state.ndim == 2: 
                    state = state.unsqueeze(0) 
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
            if was_training:
                self.train() 
            return action

# =================================
# 5. 经验回放和奖励系统
# =================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity=20000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        experience = Experience(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs_sum = probs.sum()

        if probs_sum == 0 or len(self.buffer) == 0:
            if len(self.buffer) == 0: return None
            num_available_samples = len(self.buffer)
            actual_batch_size = min(batch_size, num_available_samples)
            if actual_batch_size == 0: return None
            indices = np.random.choice(num_available_samples, actual_batch_size, replace=False)
            weights = np.ones(actual_batch_size, dtype=np.float32)
        else:
            probs /= probs_sum
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)

        experiences = [self.buffer[idx] for idx in indices]
        states = torch.stack([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

def enhanced_compute_reward(action, true_label, confidence=1.0, is_human_labeled=False):
    """
    计算奖励，人工标注的样本给予更高的权重
    """
    if true_label == -1: 
        return 0.0 
    
    # 人工标注的样本权重更高
    weight_multiplier = 3.0 if is_human_labeled else 1.0
    
    base_reward = confidence * weight_multiplier
    TP_REWARD, TN_REWARD, FN_PENALTY, FP_PENALTY = 5.0, 1.0, -3.0, -0.5 
    
    if action == true_label: 
        reward = base_reward * TP_REWARD if true_label == 1 else base_reward * TN_REWARD
    else: 
        if true_label == 1 and action == 0: 
            reward = base_reward * FN_PENALTY 
        elif true_label == 0 and action == 1: 
            reward = base_reward * FP_PENALTY 
        else: 
            reward = -base_reward 
    
    return reward

# =================================
# 6. 交互式训练函数
# =================================

def interactive_train_rlad_gui(agent, target_agent, optimizer, scheduler, replay_buffer, 
                              X_train, y_train, windows_raw_train, X_val, y_val, device, 
                              annotation_system, num_episodes=150, target_update_freq=15,
                              epsilon_start=0.95, epsilon_end=0.02, epsilon_decay_rate=0.995,
                              batch_size_rl=64, output_dir="./output", annotation_frequency=10):
    """
    交互式RLAD训练，包含GUI人工标注系统
    """
    os.makedirs(output_dir, exist_ok=True)
    training_history = {
        'episodes': [], 'train_loss': [], 'avg_td_error': [], 'val_f1': [], 
        'val_precision': [], 'val_recall': [], 'epsilon': [], 
        'replay_buffer_size': [], 'learning_rate': [], 'anomaly_f1': [], 'normal_f1': [],
        'human_annotations_count': [], 'human_labeled_accuracy': []
    }
    
    best_val_f1, best_anomaly_val_f1, patience, patience_counter = 0.0, 0.0, 40, 0
    human_labeled_indices = set()  # 记录人工标注的样本索引
    
    print("开始GUI交互式RLAD训练...")
    print("注意：在标注过程中，GUI窗口会弹出，请在GUI中进行标注操作。")
    
    # 找出所有已标注和未标注的样本
    labeled_train_mask = (y_train != -1)
    unlabeled_train_mask = (y_train == -1)
    
    X_train_labeled = X_train[labeled_train_mask]
    y_train_labeled = y_train[labeled_train_mask]
    
    unlabeled_indices = np.where(unlabeled_train_mask)[0]
    print(f"训练集中有 {len(unlabeled_indices)} 个未标注样本可供人工标注")
    
    epsilon = epsilon_start
    
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        episode_losses, episode_td_errors = [], []
        epsilon = max(epsilon_end, epsilon * epsilon_decay_rate)
        beta = min(1.0, 0.4 + episode * (1.0 - 0.4) / (num_episodes * 0.8))
        
        # 定期进行人工标注
        if episode > 0 and episode % annotation_frequency == 0 and len(unlabeled_indices) > 0:
            print(f"\n=== Episode {episode}: 开始GUI人工标注阶段 ===")
            print("GUI标注窗口即将弹出，请在GUI中进行标注...")
            
            # 随机选择一些未标注的样本进行人工标注
            num_to_annotate = min(3, len(unlabeled_indices))  # 每次最多标注3个
            samples_to_annotate = np.random.choice(unlabeled_indices, num_to_annotate, replace=False)
            
            newly_labeled_count = 0
            for sample_idx in samples_to_annotate:
                print(f"\n正在标注样本 {sample_idx} ({newly_labeled_count + 1}/{num_to_annotate})")
                
                # 获取Agent当前的预测
                agent.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(X_train[sample_idx]).unsqueeze(0).to(device)
                    q_values = agent(state_tensor)
                    auto_prediction = q_values.argmax(dim=1).item()
                agent.train()
                
                # 获取人工标注（通过GUI）
                try:
                    human_label = annotation_system.get_human_annotation(
                        window_data=X_train[sample_idx],
                        window_idx=sample_idx,
                        original_data_segment=windows_raw_train[sample_idx],
                        auto_predicted_label=auto_prediction
                    )
                except Exception as e:
                    print(f"标注过程中出错: {e}")
                    print("将跳过此样本...")
                    human_label = -1
                
                if human_label == -2:  # 用户选择退出
                    print("用户选择退出标注，继续训练...")
                    break
                elif human_label != -1:  # 用户提供了有效标注
                    y_train[sample_idx] = human_label
                    human_labeled_indices.add(sample_idx)
                    newly_labeled_count += 1
                    
                    # 从未标注列表中移除
                    unlabeled_indices = unlabeled_indices[unlabeled_indices != sample_idx]
                    
                    print(f"样本 {sample_idx} 已被标注为: {'异常' if human_label == 1 else '正常'}")
            
            print(f"本轮标注完成，共标注 {newly_labeled_count} 个样本")
            print(f"剩余未标注样本: {len(unlabeled_indices)} 个")
            
            # 更新已标注数据
            labeled_train_mask = (y_train != -1)
            X_train_labeled = X_train[labeled_train_mask]
            y_train_labeled = y_train[labeled_train_mask]
        
        # 正常的RL训练循环
        agent.train()
        if len(X_train_labeled) == 0:
            print("警告：没有已标注的训练样本，跳过此episode")
            continue
            
        shuffled_indices = np.random.permutation(len(X_train_labeled))
        
        for step_idx in range(len(X_train_labeled)):
            current_sample_idx_in_labeled = shuffled_indices[step_idx]
            state_np, true_label = X_train_labeled[current_sample_idx_in_labeled], y_train_labeled[current_sample_idx_in_labeled]
            
            state = torch.FloatTensor(state_np).to(device)
            action = agent.get_action(state, epsilon)
            
            # 检查是否为人工标注的样本
            original_idx = np.where(labeled_train_mask)[0][current_sample_idx_in_labeled]
            is_human_labeled = original_idx in human_labeled_indices
            
            reward = enhanced_compute_reward(action, true_label, 1.0, is_human_labeled)
            
            next_state_np = X_train_labeled[shuffled_indices[(step_idx + 1) % len(X_train_labeled)]]
            next_state = torch.FloatTensor(next_state_np).to(device)
            
            replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), False)
            
            if len(replay_buffer) >= batch_size_rl * 2:
                loss, td_err = enhanced_train_dqn_step(agent, target_agent, replay_buffer, 
                                                     optimizer, device, batch_size=batch_size_rl, beta=beta)
                if loss is not None and loss > 0:
                    episode_losses.append(loss)
                if td_err is not None and td_err > 0:
                    episode_td_errors.append(td_err)
        
        # 更新目标网络
        if episode % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        scheduler.step()
        
        # 验证集评估
        val_dataset = HydraulicDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        val_metrics = enhanced_evaluate_model(agent, val_loader, device)
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0.0
        
        current_anomaly_f1 = val_metrics['f1_per_class'][1] if len(val_metrics['f1_per_class']) > 1 else 0.0
        current_normal_f1 = val_metrics['f1_per_class'][0] if len(val_metrics['f1_per_class']) > 0 else 0.0
        
        # 计算人工标注样本的准确率
        human_labeled_accuracy = 0.0
        if len(human_labeled_indices) > 0:
            correct_human_predictions = 0
            total_human_samples = 0
            agent.eval()
            with torch.no_grad():
                for idx in human_labeled_indices:
                    if y_train[idx] != -1:  # 确保有标签
                        state_tensor = torch.FloatTensor(X_train[idx]).unsqueeze(0).to(device)
                        q_values = agent(state_tensor)
                        prediction = q_values.argmax(dim=1).item()
                        if prediction == y_train[idx]:
                            correct_human_predictions += 1
                        total_human_samples += 1
            agent.train()
            
            if total_human_samples > 0:
                human_labeled_accuracy = correct_human_predictions / total_human_samples
        
        # 记录训练历史
        training_history['episodes'].append(episode)
        training_history['train_loss'].append(avg_loss)
        training_history['avg_td_error'].append(avg_td_error)
        training_history['val_f1'].append(val_metrics['f1'])
        training_history['val_precision'].append(val_metrics['precision'])
        training_history['val_recall'].append(val_metrics['recall'])
        training_history['epsilon'].append(epsilon)
        training_history['replay_buffer_size'].append(len(replay_buffer))
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        training_history['anomaly_f1'].append(current_anomaly_f1)
        training_history['normal_f1'].append(current_normal_f1)
        training_history['human_annotations_count'].append(len(human_labeled_indices))
        training_history['human_labeled_accuracy'].append(human_labeled_accuracy)
        
        # 保存最佳模型
        improved = False
        if current_anomaly_f1 > best_anomaly_val_f1:
            best_anomaly_val_f1, best_val_f1, improved = current_anomaly_f1, val_metrics['f1'], True
        elif current_anomaly_f1 == best_anomaly_val_f1 and val_metrics['f1'] > best_val_f1:
            best_val_f1, improved = val_metrics['f1'], True
        
        if improved:
            patience_counter = 0
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'target_agent_state_dict': target_agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'episode': episode,
                'best_val_f1': best_val_f1,
                'best_anomaly_val_f1': best_anomaly_val_f1,
                'human_labeled_indices': list(human_labeled_indices),
                'training_history_partial': convert_to_serializable(training_history)
            }, os.path.join(output_dir, 'best_enhanced_rlad_model_gui.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停在episode {episode}")
            break
            
        if episode % 5 == 0 or episode == num_episodes - 1:
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"Loss: {avg_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, Anomaly F1: {current_anomaly_f1:.4f}")
            print(f"人工标注样本数: {len(human_labeled_indices)}, 人工标注准确率: {human_labeled_accuracy:.4f}")
    
    print(f"\n训练完成！最佳验证集F1: {best_val_f1:.4f}, 最佳异常F1: {best_anomaly_val_f1:.4f}")
    print(f"总共进行了 {len(human_labeled_indices)} 个人工标注")
    
    return training_history

# =================================
# 7. 辅助训练函数
# =================================

def enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, 
                           gamma=0.99, batch_size=64, beta=0.4):
    if len(replay_buffer) < batch_size: 
        return 0.0, 0.0
    
    sample_result = replay_buffer.sample(batch_size, beta)
    if sample_result is None: 
        return 0.0, 0.0
    
    states, actions, rewards, next_states, dones, indices, weights = sample_result
    states, actions, rewards, next_states, dones = states.to(device), actions.to(device), rewards.to(device), next_states.to(device), dones.to(device)
    weights = torch.FloatTensor(weights).to(device)
    
    current_q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_actions = agent(next_states).argmax(dim=1)
        next_q_values_target = target_agent(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (gamma * next_q_values_target * ~dones)
    
    td_errors = target_q_values - current_q_values
    loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none', beta=1.0)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()
    
    priorities_np = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
    replay_buffer.update_priorities(indices, priorities_np)
    
    return loss.item(), td_errors.abs().mean().item()

def enhanced_evaluate_model(agent, data_loader, device):
    agent.eval()
    all_predictions, all_labels = [], []
    
    with torch.no_grad():
        for states, labels in data_loader:
            states = states.to(device)
            labeled_mask = (labels != -1)
            if labeled_mask.sum() == 0: 
                continue
                
            states_labeled, labels_labeled = states[labeled_mask], labels[labeled_mask]
            if states_labeled.size(0) == 0: 
                continue
            
            q_values = agent(states_labeled)
            predictions = q_values.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_labeled.cpu().numpy())
    
    agent.train()
    
    if len(all_predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "precision_per_class": [0.0, 0.0], 
                "recall_per_class": [0.0, 0.0], "f1_per_class": [0.0, 0.0]}
    
    all_predictions_np, all_labels_np = np.array(all_predictions), np.array(all_labels)
    
    precision_w = precision_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    recall_w = recall_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    f1_w = f1_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    
    try:
        precision_pc = precision_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
        recall_pc = recall_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
        f1_pc = f1_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
    except:
        precision_pc, recall_pc, f1_pc = np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
    
    return {
        "precision": float(precision_w), 
        "recall": float(recall_w), 
        "f1": float(f1_w),
        "precision_per_class": [float(x) for x in precision_pc], 
        "recall_per_class": [float(x) for x in recall_pc], 
        "f1_per_class": [float(x) for x in f1_pc]
    }

# =================================
# 8. 主函数
# =================================

def parse_args():
    parser = argparse.ArgumentParser(description='GUI交互式RLAD液压支架异常检测')
    parser.add_argument('--data_path', type=str, 
                       default="C:/Users/18104/Desktop/Python files/deeplearning/example/timeseries/examples/RLAD/clean_data.csv", 
                       help='数据文件路径')
    parser.add_argument('--feature_column_name', type=str, default=None, 
                       help='要处理的特定特征列名')
    parser.add_argument('--window_size', type=int, default=288, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=12, help='滑动窗口步长')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--num_episodes', type=int, default=100, help='训练轮数')
    parser.add_argument('--annotation_frequency', type=int, default=10, help='人工标注频率（每多少个episode）')
    parser.add_argument('--use_gui', action='store_true', default=True, help='是否使用GUI界面进行标注')
    parser.add_argument('--no_gui', action='store_true', help='禁用GUI，使用命令行标注')
    parser.add_argument('--output_dir', type=str, 
                       default="C:/Users/18104/Desktop/Python files/deeplearning/example/timeseries/examples/RLAD/output_gui", 
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default="auto", help='设备选择')
    parser.add_argument('--batch_size_rl', type=int, default=64, help='强化学习训练批次大小')
    parser.add_argument('--target_update_freq', type=int, default=10, help='目标网络更新频率')
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99, help='Epsilon指数衰减率')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 处理GUI设置
    if args.no_gui:
        args.use_gui = False
    
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"GUI模式: {'启用' if args.use_gui else '禁用（使用命令行）'}")
    
    try:
        # 数据加载
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, \
        test_window_original_indices, df_for_point_mapping, actual_selected_column_name, \
        windows_raw_train = load_hydraulic_data_improved(
            data_path=args.data_path, 
            window_size=args.window_size, 
            stride=args.stride,
            specific_feature_column=args.feature_column_name
        )
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_feature_name = "".join(c if c.isalnum() else "_" for c in actual_selected_column_name)
        mode_suffix = "gui" if args.use_gui else "cmd"
        output_dir_timestamped = os.path.join(args.output_dir, f"rlad_{mode_suffix}_{clean_feature_name}_{timestamp}")
        os.makedirs(output_dir_timestamped, exist_ok=True)
        
        # 初始化人工标注系统
        annotation_system = HumanAnnotationSystem(output_dir_timestamped, args.window_size, use_gui=args.use_gui)
        
        # 保存配置
        config_to_save = convert_to_serializable(vars(args))
        config_to_save['output_dir_actual'] = output_dir_timestamped
        config_to_save['actual_feature_column_used'] = actual_selected_column_name
        with open(os.path.join(output_dir_timestamped, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=4)
        
        # 初始化模型
        input_dim = X_train.shape[-1]
        agent = EnhancedRLADAgent(input_dim=input_dim, seq_len=args.window_size, 
                                 hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
        target_agent = EnhancedRLADAgent(input_dim=input_dim, seq_len=args.window_size, 
                                        hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
        target_agent.load_state_dict(agent.state_dict())
        target_agent.eval()
        
        optimizer = optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(10, args.num_episodes // 5))
        replay_buffer = PrioritizedReplayBuffer(capacity=50000, alpha=0.6)
        
        print(f"模型参数数量: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")
        print(f"所有输出将保存到: {output_dir_timestamped}")
        
        if args.use_gui:
            print("\n=== GUI模式启动 ===")
            print("在标注过程中，将会弹出图形界面窗口。")
            print("请在GUI中查看数据图表并进行标注选择。")
            print("支持快捷键：0=正常, 1=异常, S=跳过, Q/Esc=退出")
        
        # 开始交互式训练
        training_history = interactive_train_rlad_gui(
            agent, target_agent, optimizer, scheduler, replay_buffer,
            X_train, y_train, windows_raw_train, X_val, y_val, device,
            annotation_system, 
            num_episodes=args.num_episodes,
            target_update_freq=args.target_update_freq,
            epsilon_decay_rate=args.epsilon_decay_rate,
            batch_size_rl=args.batch_size_rl,
            output_dir=output_dir_timestamped,
            annotation_frequency=args.annotation_frequency
        )
        
        # 保存训练历史
        with open(os.path.join(output_dir_timestamped, 'training_history_final.json'), 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(training_history), f, ensure_ascii=False, indent=4)
        
        # 最终测试评估
        print("\n=== 最终测试评估 ===")
        test_dataset = HydraulicDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = enhanced_evaluate_model(agent, test_loader, device)
        
        print(f"最终测试集结果: F1={test_metrics['f1']:.4f}, 精确率={test_metrics['precision']:.4f}, 召回率={test_metrics['recall']:.4f}")
        if len(test_metrics['f1_per_class']) >= 2:
            print(f"正常类F1: {test_metrics['f1_per_class'][0]:.4f}, 异常类F1: {test_metrics['f1_per_class'][1]:.4f}")
        
        with open(os.path.join(output_dir_timestamped, 'test_results_final.json'), 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(test_metrics), f, ensure_ascii=False, indent=4)
        
        print(f"\nGUI交互式RLAD训练完成！所有结果保存在: {output_dir_timestamped}")
        
    except Exception as e:
        print(f"程序运行过程中发生错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()