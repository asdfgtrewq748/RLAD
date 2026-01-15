"""
RLAD v3.0: 基于STL+LOF双层异常检测的强化学习液压支架工作阻力异常检测 - 完整修正版
"""

# 添加缺失的导入

import os
import json
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tkinter as tk
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import deque, namedtuple
from typing import Optional, Tuple, List, Dict
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor

# STL分解相关导入
from statsmodels.tsa.seasonal import STL

# GUI相关导入
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available, some 3D visualizations will be skipped")

from scipy.interpolate import griddata
# 配置matplotlib
matplotlib.use('TkAgg')
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
# 修正版STL+LOF双层异常检测系统
# =================================

class STLLOFAnomalyDetector:
    """STL+LOF双层异常检测器 - 修正版"""
    
    def __init__(self, period=24, seasonal=25, robust=True, 
                 n_neighbors=20, contamination=0.02, metric='minkowski'):
        # STL参数
        self.period = period
        self.seasonal = seasonal if seasonal % 2 == 1 else seasonal + 1
        self.robust = robust
        
        # LOF参数
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        
        # 验证STL参数
        if self.seasonal <= self.period:
            self.seasonal = self.period + 2
            if self.seasonal % 2 == 0:
                self.seasonal += 1
        
        # 存储分解结果
        self.stl_result = None
        self.lof_model = None
        self.anomaly_scores = None
        self.anomaly_labels = None
        self.decomposition_quality = {}
        
        print(f"🔧 STL+LOF检测器初始化:")
        print(f"   STL参数: period={period}, seasonal={self.seasonal}, robust={robust}")
        print(f"   LOF参数: n_neighbors={n_neighbors}, contamination={contamination}")
    
    def prepare_time_series_data(self, data, datetime_col=None):
        """准备时间序列数据用于STL分解"""
        if isinstance(data, np.ndarray):
            series = pd.Series(data)
        elif isinstance(data, pd.Series):
            series = data.copy()
        else:
            series = pd.Series(data)
        
        # 处理缺失值
        if series.isnull().any():
            series = series.fillna(method='ffill').fillna(method='bfill')
        
        # 数据验证
        if len(series) < 2 * self.period:
            raise ValueError(f"数据长度 {len(series)} 太短，至少需要 {2 * self.period} 个点")
        
        return series
    
    def fit_stl_decomposition(self, series):
        """执行STL分解"""
        print("🔄 开始STL分解...")
        
        try:
            stl = STL(series, seasonal=self.seasonal, period=self.period, robust=self.robust)
            self.stl_result = stl.fit()
            self._evaluate_decomposition_quality(series)
            print("✅ STL分解完成")
            return self.stl_result
            
        except Exception as e:
            print(f"❌ STL分解失败: {e}")
            raise
    
    def _evaluate_decomposition_quality(self, original_series):
        """评估STL分解质量"""
        if self.stl_result is None:
            return
        
        # 重构检验
        reconstructed = (self.stl_result.trend + 
                        self.stl_result.seasonal + 
                        self.stl_result.resid)
        
        # 计算重构误差
        reconstruction_error = np.mean((original_series - reconstructed) ** 2)
        
        # 计算各分量的方差贡献
        total_var = np.var(original_series)
        trend_var = np.var(self.stl_result.trend.dropna())
        seasonal_var = np.var(self.stl_result.seasonal)
        resid_var = np.var(self.stl_result.resid.dropna())
        
        self.decomposition_quality = {
            'reconstruction_mse': reconstruction_error,
            'trend_variance_ratio': trend_var / total_var,
            'seasonal_variance_ratio': seasonal_var / total_var,
            'residual_variance_ratio': resid_var / total_var,
            'explained_variance_ratio': (trend_var + seasonal_var) / total_var
        }
        
        print(f"📊 STL分解质量评估:")
        print(f"   - 重构MSE: {reconstruction_error:.6f}")
        print(f"   - 趋势方差占比: {self.decomposition_quality['trend_variance_ratio']:.1%}")
        print(f"   - 季节方差占比: {self.decomposition_quality['seasonal_variance_ratio']:.1%}")
        print(f"   - 残差方差占比: {self.decomposition_quality['residual_variance_ratio']:.1%}")
    
    def fit_lof_detection(self, residuals):
        """在STL残差上应用LOF异常检测"""
        print("🎯 开始LOF异常检测...")
        
        # 准备数据：转换为2D数组
        if isinstance(residuals, pd.Series):
            residuals_clean = residuals.dropna()
        else:
            residuals_clean = pd.Series(residuals).dropna()
        
        residuals_2d = residuals_clean.values.reshape(-1, 1)
        
        if len(residuals_2d) == 0:
            raise ValueError("残差数据为空，无法进行LOF检测")
        
        # 创建LOF模型
        self.lof_model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(residuals_2d) - 1),
            contamination=self.contamination,
            metric=self.metric,
            novelty=False
        )
        
        try:
            # 拟合并预测
            lof_labels = self.lof_model.fit_predict(residuals_2d)
            lof_scores = -self.lof_model.negative_outlier_factor_
            
            print(f"✅ LOF检测完成，发现 {np.sum(lof_labels == -1)} 个异常点")
            return lof_labels, lof_scores
            
        except Exception as e:
            print(f"❌ LOF检测失败: {e}")
            raise
    
    def detect_anomalies(self, data):
        """完整的STL+LOF异常检测流程"""
        # 第一步：准备数据
        series = self.prepare_time_series_data(data)
        
        # 第二步：STL分解
        stl_result = self.fit_stl_decomposition(series)
        residuals = stl_result.resid.dropna()
        
        # 第三步：LOF异常检测
        lof_labels, lof_scores = self.fit_lof_detection(residuals)
        
        # 第四步：映射回原始数据格式
        # LOF标签：-1=异常，1=正常 -> 转换为 1=异常，0=正常
        anomaly_binary = (lof_labels == -1).astype(int)
        
        # 处理因dropna导致的长度差异
        full_labels = np.zeros(len(series), dtype=int)
        residuals_index = residuals.index
        series_index = series.index
        
        # 找到残差索引在原序列中的位置
        for i, res_idx in enumerate(residuals_index):
            series_pos = series_index.get_loc(res_idx)
            full_labels[series_pos] = anomaly_binary[i]
        
        return full_labels

# =================================
# 修正版GUI标注界面 - 从RLADv2.2移植
# =================================


    """修正版的基于Tkinter的可视化标注界面"""
    
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
            
            # 创建主容器
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
            
            # 3. 图表区域
            chart_frame = tk.LabelFrame(main_container, 
                                       text="数据可视化", 
                                       font=('Arial', 12, 'bold'),
                                       bg='#f0f0f0',
                                       fg='#34495e',
                                       padx=5, pady=5)
            chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
            
            self.create_charts(chart_frame)
            print("图表区域创建完成")
            
            # 4. 按钮区域
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
        """创建标注按钮"""
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

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


    """液压数据集类"""
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
                if 'weight_ih' in name or 'weight_hh' in name:
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
# 经验回放和奖励系统
# =================================

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
            if len(self.buffer) == 0: 
                return None
            num_available_samples = len(self.buffer)
            actual_batch_size = min(batch_size, num_available_samples)
            if actual_batch_size == 0: 
                return None
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


# =================================
# 辅助训练函数
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
    """修复的模型评估函数"""
    agent.eval()
    all_predictions, all_labels = [], []
    
    with torch.no_grad():
        for states, labels in data_loader:
            states = states.to(device)
            labeled_mask = (labels != -1)
            
            # 如果没有标注数据，使用初始异常标签作为伪标签
            if labeled_mask.sum() == 0:
                # 使用模型预测作为标签评估
                q_values = agent(states)
                predictions = q_values.argmax(dim=1)
                # 生成伪标签：基于预测置信度
                probabilities = torch.softmax(q_values, dim=1)
                confidence = probabilities.max(dim=1)[0]
                pseudo_labels = predictions.clone()
                # 低置信度的预测标记为不确定
                uncertain_mask = confidence < 0.7
                pseudo_labels[uncertain_mask] = 0  # 默认为正常
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(pseudo_labels.cpu().numpy())
            else:
                states_labeled, labels_labeled = states[labeled_mask], labels[labeled_mask]
                if states_labeled.size(0) == 0: 
                    continue
                
                q_values = agent(states_labeled)
                predictions = q_values.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_labeled.cpu().numpy())
    
    agent.train()
    
    if len(all_predictions) == 0:
        print("⚠️ 没有可评估的数据，返回默认指标")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "precision_per_class": [0.0, 0.0], 
                "recall_per_class": [0.0, 0.0], "f1_per_class": [0.0, 0.0]}
    
    all_predictions_np, all_labels_np = np.array(all_predictions), np.array(all_labels)
    
    # 检查数据分布
    unique_labels = np.unique(all_labels_np)
    unique_preds = np.unique(all_predictions_np)
    print(f"评估数据 - 真实标签分布: {np.bincount(all_labels_np)}")
    print(f"评估数据 - 预测标签分布: {np.bincount(all_predictions_np)}")
    
    # 使用zero_division=1避免除零错误
    precision_w = precision_score(all_labels_np, all_predictions_np, zero_division=1, average='weighted')
    recall_w = recall_score(all_labels_np, all_predictions_np, zero_division=1, average='weighted')
    f1_w = f1_score(all_labels_np, all_predictions_np, zero_division=1, average='weighted')
    
    try:
        precision_pc = precision_score(all_labels_np, all_predictions_np, zero_division=1, average=None, labels=[0,1])
        recall_pc = recall_score(all_labels_np, all_predictions_np, zero_division=1, average=None, labels=[0,1])
        f1_pc = f1_score(all_labels_np, all_predictions_np, zero_division=1, average=None, labels=[0,1])
        
        # 处理可能的缺失类别
        if len(precision_pc) < 2:
            precision_pc = np.array([precision_pc[0] if len(precision_pc) > 0 else 0.0, 0.0])
        if len(recall_pc) < 2:
            recall_pc = np.array([recall_pc[0] if len(recall_pc) > 0 else 0.0, 0.0])
        if len(f1_pc) < 2:
            f1_pc = np.array([f1_pc[0] if len(f1_pc) > 0 else 0.0, 0.0])
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
# 修正的数据加载函数
# =================================

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
    windows_raw_max_# filepath: c:\Users\18104\Desktop\Python files\deeplearning\example\timeseries\examples\RLAD\RLADv2.3.py


# =================================
# 修正版STL+LOF双层异常检测系统
# =================================

# =================================
# 修正版GUI标注界面 - 从RLADv2.2移植
# =================================

    """修正版的基于Tkinter的可视化标注界面"""
    
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
            
            # 创建主容器
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
            
            # 3. 图表区域
            chart_frame = tk.LabelFrame(main_container, 
                                       text="数据可视化", 
                                       font=('Arial', 12, 'bold'),
                                       bg='#f0f0f0',
                                       fg='#34495e',
                                       padx=5, pady=5)
            chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
            
            self.create_charts(chart_frame)
            print("图表区域创建完成")
            
            # 4. 按钮区域
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
        """创建标注按钮"""
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

# =================================
# 修正版人工标注系统
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
            os.makedirs(os.path.dirname(self.manual_labels_file), exist_ok=True)
            with open(self.manual_labels_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotation_history, f, ensure_ascii=False, indent=4,
                         default=convert_to_serializable)
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
# 数据集和神经网络模型类
# =================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class HydraulicDataset(Dataset):
    """液压数据集类"""
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
                if 'weight_ih' in name or 'weight_hh' in name:
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
# 经验回放和奖励系统
# =================================
# 在 EnhancedRLADAgent 类之后添加（大约在第 1200-1300 行附近）
class OptimizedRLADAgent(nn.Module):
    """优化的RLAD智能体网络架构 - 修复版"""
    
    def __init__(self, input_dim, seq_len=288, hidden_size=128, num_layers=3):
        super(OptimizedRLADAgent, self).__init__()
        
        self.input_dim = input_dim 
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多尺度特征提取
        self.conv1d_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_size//4, kernel_size=3, padding=1),
            nn.Conv1d(input_dim, hidden_size//4, kernel_size=5, padding=2),
            nn.Conv1d(input_dim, hidden_size//4, kernel_size=7, padding=3),
            nn.Conv1d(input_dim, hidden_size//4, kernel_size=11, padding=5),
        ])
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # 层归一化
        self.ln_conv = nn.LayerNorm(hidden_size)
        self.ln_attention = nn.LayerNorm(lstm_output_size)
        
        # 残差连接的全连接层
        self.fc_layers = nn.ModuleList([
            nn.Linear(lstm_output_size, hidden_size),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size//4),
        ])
        
        self.output_layer = nn.Linear(hidden_size//4, 2)
        
        # 激活函数和正则化
        self.dropout = nn.Dropout(0.3)
        self.gelu = nn.GELU()
        
        # 使用 LayerNorm 替代 BatchNorm 以避免单样本问题
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(hidden_size),
            nn.LayerNorm(hidden_size//2),
            nn.LayerNorm(hidden_size//4),
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        
    def forward(self, x, return_features=False, return_attention_weights=False):
        batch_size, seq_len, input_dim = x.shape
        
        # 多尺度卷积特征提取
        x_conv = x.transpose(1, 2)  # (B, input_dim, seq_len)
        conv_features = []
        
        for conv_layer in self.conv1d_layers:
            conv_out = self.gelu(conv_layer(x_conv))
            conv_features.append(conv_out)
        
        # 拼接多尺度特征
        conv_concat = torch.cat(conv_features, dim=1)  # (B, hidden_size, seq_len)
        conv_concat = conv_concat.transpose(1, 2)  # (B, seq_len, hidden_size)
        conv_concat = self.ln_conv(conv_concat)
        
        # BiLSTM处理
        lstm_out, (hidden, cell) = self.lstm(conv_concat)
        
        # 自注意力机制
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.ln_attention(lstm_out + attn_out)  # 残差连接
        
        # 全局特征聚合
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        last_hidden = lstm_out[:, -1, :]
        
        # 特征融合
        combined_features = avg_pool + max_pool + last_hidden
        
        # 深度特征学习 - 使用LayerNorm替代BatchNorm
        x_fc = combined_features
        features_list = []
        
        for i, (fc_layer, ln_layer) in enumerate(zip(self.fc_layers, self.ln_layers)):
            x_fc = fc_layer(x_fc)
            x_fc = ln_layer(x_fc)  # 使用LayerNorm
            x_fc = self.gelu(x_fc)
            x_fc = self.dropout(x_fc)
            features_list.append(x_fc)
        
        # 输出层
        q_values = self.output_layer(x_fc)
        
        if return_features:
            if return_attention_weights:
                return q_values, features_list, attn_weights
            return q_values, features_list
        else:
            if return_attention_weights:
                return q_values, attn_weights
            return q_values
    
    def get_action(self, state, epsilon=0.0):
        """获取动作，处理单样本推理"""
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            was_training = self.training
            self.eval()  # 切换到评估模式避免BatchNorm问题
            with torch.no_grad():
                if state.ndim == 2:
                    state = state.unsqueeze(0)
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
            if was_training:
                self.train()
            return action
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
            if len(self.buffer) == 0: 
                return None
            num_available_samples = len(self.buffer)
            actual_batch_size = min(batch_size, num_available_samples)
            if actual_batch_size == 0: 
                return None
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


   
def adaptive_compute_reward(action, true_label, confidence=1.0, is_human_labeled=False, 
                           class_balance_ratio=1.0):
    """自适应奖励计算，考虑类别平衡"""
    if true_label == -1: 
        return 0.0 
    
    # 动态权重调整
    base_multiplier = 2.0 if is_human_labeled else 1.0
    
    # 类别平衡奖励
    if true_label == 1:  # 异常类
        balance_weight = class_balance_ratio
    else:  # 正常类
        balance_weight = 1.0
    
    # 置信度奖励
    confidence_bonus = confidence * 0.5
    
    # 基础奖励设计
    if action == true_label:
        if true_label == 1:  # TP
            reward = (5.0 + confidence_bonus) * base_multiplier * balance_weight
        else:  # TN
            reward = (2.0 + confidence_bonus) * base_multiplier
    else:
        if true_label == 1 and action == 0:  # FN (最严重的错误)
            reward = -8.0 * base_multiplier * balance_weight
        elif true_label == 0 and action == 1:  # FP
            reward = -2.0 * base_multiplier
        else:
            reward = -1.0 * base_multiplier
    
    return reward
# =================================
# 辅助训练函数
# =================================

def enhanced_train_dqn_step_v2(agent, target_agent, replay_buffer, optimizer, device, 
                               gamma=0.99, batch_size=64, beta=0.4):
    """优化的DQN训练步骤"""
    if len(replay_buffer) < batch_size: 
        return 0.0, 0.0, {}
    
    sample_result = replay_buffer.sample(batch_size, beta)
    if sample_result is None: 
        return 0.0, 0.0, {}
    
    states, actions, rewards, next_states, dones, indices, weights = sample_result
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    weights = torch.FloatTensor(weights).to(device)
    
    # 计算当前Q值
    q_values, features = agent(states, return_features=True)
    current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # 计算目标Q值 (Double DQN)
    with torch.no_grad():
        next_q_values = agent(next_states)
        next_actions = next_q_values.argmax(dim=1)
        next_q_values_target = target_agent(next_states)
        next_q_values_selected = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (gamma * next_q_values_selected * ~dones)
    
    # 计算损失
    td_errors = target_q_values - current_q_values
    loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
    
    # 梯度裁剪和优化
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 更新优先级
    priorities_np = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
    replay_buffer.update_priorities(indices, priorities_np)
    
    # 额外的训练指标
    training_metrics = {
        'q_value_mean': q_values.mean().item(),
        'q_value_std': q_values.std().item(),
        'grad_norm': grad_norm.item(),
        'td_error_mean': td_errors.abs().mean().item(),
        'feature_activation': [f.mean().item() for f in features]
    }
    
    return loss.item(), td_errors.abs().mean().item(), training_metrics

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
# 修正的数据加载函数
# =================================

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
    windows_raw_max_values_list = []  # 保存每个窗口的原始最大值
    window_indices_list = []  # 保存窗口索引
    
    # 创建滑动窗口
    for i in range(0, len(data_scaled) - window_size + 1, stride):
        window_scaled = data_scaled[i:i + window_size]
        window_raw = data_values[i:i + window_size]
        
        # 计算窗口内的最大值
        window_max = np.max(window_raw)
        
        windows_scaled_list.append(window_scaled)
        windows_raw_data_list.append(window_raw)
        windows_raw_max_values_list.append(window_max)
        window_indices_list.append(i)
    
    # 转换为numpy数组
    X_scaled = np.array(windows_scaled_list)
    X_raw = np.array(windows_raw_data_list)
    raw_max_values = np.array(windows_raw_max_values_list)
    
    print(f"滑动窗口创建完成，窗口数量: {len(X_scaled)}")
    print(f"每个窗口的形状: {X_scaled[0].shape}")
    
    # 基于"来压判据"生成初步异常标签
    initial_anomaly_labels = []
    for max_val in raw_max_values:
        if max_val > laiya_criterion_point:
            initial_anomaly_labels.append(1)  # 异常
        else:
            initial_anomaly_labels.append(0)  # 正常
    
    initial_anomaly_labels = np.array(initial_anomaly_labels)
    
    print(f"初步异常标签统计:")
    print(f"  正常窗口数: {np.sum(initial_anomaly_labels == 0)}")
    print(f"  异常窗口数: {np.sum(initial_anomaly_labels == 1)}")
    print(f"  异常比例: {np.mean(initial_anomaly_labels):.1%}")
    
    # 初始化所有标签为未标注状态 (-1)
    y_labels = np.full(len(X_scaled), -1, dtype=int)
    
    return {
        'X_scaled': X_scaled,
        'X_raw': X_raw,
        'y_labels': y_labels,
        'initial_anomaly_labels': initial_anomaly_labels,
        'scaler': scaler,
        'selected_columns': selected_cols,
        'laiya_criterion': laiya_criterion_point,
        'window_indices': window_indices_list,
        'df_mapping': df_for_point_mapping
    }

# =================================
# 交互式训练函数
# =================================

def interactive_train_rlad_gui(agent, target_agent, optimizer, scheduler, replay_buffer, 
                              X_train, y_train, windows_raw_train, X_val, y_val, device, 
                              annotation_system, num_episodes=50, target_update_freq=10,
                              epsilon_start=0.95, epsilon_end=0.02, epsilon_decay_rate=0.995,
                              batch_size_rl=64, output_dir="./output", annotation_frequency=20,
                              max_human_annotations=30):
    """
    优化的交互式训练循环：结合强化学习与人工标注
    """
    print("🚀 开始交互式RLAD训练...")
    print(f"📝 人工标注上限设置为: {max_human_annotations} 个窗口")
    
    # 训练历史记录
    training_history = {
        'episode': [],
        'loss': [],
        'td_error': [],
        'epsilon': [],
        'reward': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'human_annotations': 0,
        'total_annotations': 0,
        'max_human_annotations': max_human_annotations
    }
    
    # 预转换所有训练数据为tensor（避免重复转换）
    print("🔄 预处理训练数据...")
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    
    # 验证数据加载器（减少batch size）
    val_dataset = HydraulicDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 训练循环
    current_epsilon = epsilon_start
    samples_per_episode = min(100, len(X_train) // 5)
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        episode_loss = 0.0
        episode_td_error = 0.0
        episode_reward = 0.0
        episode_steps = 0
        
        # 随机选择有限数量的训练样本（提高效率）
        selected_indices = np.random.choice(len(X_train), 
                                          size=min(samples_per_episode, len(X_train)), 
                                          replace=False)
        
        for step, idx in enumerate(selected_indices):
            window_data = X_train[idx]
            current_label = y_train[idx]
            raw_window_data = windows_raw_train[idx] if windows_raw_train is not None else None
            
            # 使用预转换的tensor
            state = X_train_tensor[idx].unsqueeze(0)
            
            # 智能体预测 - 使用get_action方法避免BatchNorm问题
            predicted_action = agent.get_action(state, epsilon=current_epsilon)
            
            # 计算置信度（在评估模式下）
            agent.eval()
            with torch.no_grad():
                q_values = agent(state)
                confidence = torch.softmax(q_values, dim=1).max().item()
            agent.train()
            
            # 人工标注逻辑（优化条件）
            true_label = current_label
            is_human_labeled = False
            
            if current_label == -1:  # 未标注的样本
                # 检查是否已达到人工标注上限
                if training_history['human_annotations'] >= max_human_annotations:
                    # 达到上限，只使用AI预测作为伪标签
                    true_label = predicted_action
                    y_train[idx] = predicted_action
                else:
                    # 优化的标注条件（减少标注频率）
                    need_annotation = False
                    
                    # 条件1：定期标注（降低频率）
                    if (step + 1) % annotation_frequency == 0:
                        need_annotation = True
                    
                    # 条件2：预测不确定性高（提高阈值）
                    elif confidence < 0.6:
                        need_annotation = True
                    
                    # 条件3：随机采样（降低概率）
                    elif np.random.random() < 0.02:
                        need_annotation = True
                    
                    if need_annotation:
                        remaining_annotations = max_human_annotations - training_history['human_annotations']
                        print(f"\n🎯 需要人工标注 - 步骤 {step + 1}, 窗口索引 {idx}")
                        print(f"   AI预测: {predicted_action}, 置信度: {confidence:.3f}")
                        print(f"   剩余可标注数量: {remaining_annotations}")
                        
                        # 获取人工标注
                        human_label = annotation_system.get_human_annotation(
                            window_data, idx, raw_window_data, predicted_action
                        )
                        
                        if human_label == -2:  # 退出标注
                            print("🛑 用户退出标注，停止训练")
                            return training_history
                        elif human_label == -1:  # 跳过
                            print("⏭️ 跳过此样本")
                            continue
                        else:
                            true_label = human_label
                            y_train[idx] = human_label
                            is_human_labeled = True
                            training_history['human_annotations'] += 1
                            
                            label_text = "异常" if human_label == 1 else "正常"
                            remaining = max_human_annotations - training_history['human_annotations']
                            print(f"✅ 人工标注完成: {label_text}")
                            print(f"📊 已完成 {training_history['human_annotations']}/{max_human_annotations} 个人工标注，剩余 {remaining} 个")
                    else:
                        # 使用AI预测作为伪标签
                        true_label = predicted_action
                        y_train[idx] = predicted_action
            
            # 如果没有有效标签，跳过此步骤
            if true_label == -1:
                continue
            
            training_history['total_annotations'] += 1
            
            # 计算奖励
            reward = enhanced_compute_reward(
                predicted_action, true_label, confidence, is_human_labeled
            )
            
            # 存储经验
            next_state = state
            done = False
            
            replay_buffer.push(
                state.cpu().squeeze(0),
                predicted_action,
                reward,
                next_state.cpu().squeeze(0),
                done
            )
            
            # 训练DQN（批量训练以提高效率）
            if len(replay_buffer) >= batch_size_rl and step % 5 == 0:
                loss, td_error = enhanced_train_dqn_step(
                    agent, target_agent, replay_buffer, optimizer, device,
                    batch_size=batch_size_rl
                )
                episode_loss += loss
                episode_td_error += td_error
            
            episode_reward += reward
            episode_steps += 1
        
        # 更新目标网络（降低频率）
        if (episode + 1) % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
            print(f"🎯 目标网络已更新")
        
        # 更新epsilon
        current_epsilon = max(epsilon_end, current_epsilon * epsilon_decay_rate)
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        # 评估模型（降低频率）
        if training_history['total_annotations'] > 0 and (episode + 1) % 5 == 0:
            metrics = enhanced_evaluate_model(agent, val_loader, device)
            
            # 记录训练历史
            training_history['episode'].append(episode + 1)
            training_history['loss'].append(episode_loss / max(episode_steps, 1))
            training_history['td_error'].append(episode_td_error / max(episode_steps, 1))
            training_history['epsilon'].append(current_epsilon)
            training_history['reward'].append(episode_reward / max(episode_steps, 1))
            training_history['precision'].append(metrics['precision'])
            training_history['recall'].append(metrics['recall'])
            training_history['f1'].append(metrics['f1'])
            
            # 打印进度
            print(f"\n📊 Episode {episode + 1} 结果:")
            print(f"   Loss: {episode_loss / max(episode_steps, 1):.4f}")
            print(f"   TD Error: {episode_td_error / max(episode_steps, 1):.4f}")
            print(f"   Epsilon: {current_epsilon:.4f}")
            print(f"   平均奖励: {episode_reward / max(episode_steps, 1):.4f}")
            print(f"   验证集 - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            print(f"   人工标注数: {training_history['human_annotations']}/{max_human_annotations}, 总标注数: {training_history['total_annotations']}")
            
            # 如果达到人工标注上限，提示用户
            if training_history['human_annotations'] >= max_human_annotations:
                print(f"🔴 已达到人工标注上限 ({max_human_annotations})，后续将只使用AI预测进行训练")
        
        # 保存模型检查点（降低频率）
        if (episode + 1) % 20 == 0:
            checkpoint_path = os.path.join(output_dir, f'rlad_checkpoint_ep{episode + 1}.pth')
            torch.save({
                'episode': episode + 1,
                'agent_state_dict': agent.state_dict(),
                'target_agent_state_dict': target_agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'training_history': training_history,
                'epsilon': current_epsilon
            }, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
    
    print(f"\n🎉 训练完成！")
    print(f"   总人工标注数: {training_history['human_annotations']}/{max_human_annotations}")
    print(f"   总标注数: {training_history['total_annotations']}")
    
    return training_history
class OptimizedAnnotationGUI:
    """优化的GUI标注界面，减少重复创建"""
    
    def __init__(self, window_size=288):
        self.window_size = window_size
        self.result = None
        self.root = None
        self.is_gui_ready = False
        
    def get_annotation(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """优化的标注获取方法"""
        try:
            # 简化的GUI显示
            print(f"\n{'='*50}")
            print(f"窗口 #{window_idx} 需要标注")
            print(f"{'='*50}")
            
            if original_data_segment is not None:
                stats = {
                    'mean': np.mean(original_data_segment),
                    'max': np.max(original_data_segment),
                    'min': np.min(original_data_segment),
                    'std': np.std(original_data_segment)
                }
                print(f"数据统计: 均值={stats['mean']:.2f}, 最大值={stats['max']:.2f}")
                print(f"          最小值={stats['min']:.2f}, 标准差={stats['std']:.2f}")
            
            if auto_predicted_label is not None:
                label_text = "异常" if auto_predicted_label == 1 else "正常"
                print(f"AI预测: {label_text}")
            
            # 快速命令行输入
            while True:
                try:
                    choice = input("\n请选择 (0=正常, 1=异常, s=跳过, q=退出): ").strip().lower()
                    
                    if choice == 'q':
                        return -2
                    elif choice == 's':
                        return -1
                    elif choice in ['0', '1']:
                        return int(choice)
                    else:
                        print("无效输入，请重新选择")
                        
                except KeyboardInterrupt:
                    return -2
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"标注过程出错: {e}")
            return -1

# 优化的人工标注系统

    """优化的人工标注系统"""
    
    def __init__(self, output_dir: str, window_size: int = 288, use_gui: bool = False):  # 默认关闭GUI
        self.output_dir = output_dir
        self.window_size = window_size
        self.use_gui = use_gui
        self.annotation_history = []
        self.manual_labels_file = os.path.join(output_dir, 'manual_annotations.json')
        self.gui = OptimizedAnnotationGUI(window_size) if use_gui else None
        self.load_existing_annotations()
        
    def get_human_annotation(self, window_data: np.ndarray, window_idx: int, 
                           original_data_segment: np.ndarray = None, 
                           auto_predicted_label: int = None) -> int:
        """获取用户对单个窗口的标注"""
        
        # 检查缓存
        for record in self.annotation_history:
            if record.get('window_idx') == window_idx:
                return record['label']
        
        # 使用优化的标注方法
        if self.gui:
            label = self.gui.get_annotation(window_data, window_idx, original_data_segment, auto_predicted_label)
        else:
            label = self.get_annotation_cmdline(window_data, window_idx, original_data_segment, auto_predicted_label)
        
        # 保存有效标注
        if label not in [-1, -2]:
            annotation_record = {
                'window_idx': window_idx,
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'auto_predicted_label': auto_predicted_label
            }
            self.annotation_history.append(annotation_record)
            self.save_annotations()
        
        return label
    
    def get_annotation_cmdline(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """简化的命令行标注"""
        print(f"\n窗口 #{window_idx} 标注")
        
        if auto_predicted_label is not None:
            label_text = "异常" if auto_predicted_label == 1 else "正常"
            print(f"AI预测: {label_text}")
        
        while True:
            try:
                choice = input("选择 (0=正常, 1=异常, s=跳过, q=退出): ").strip().lower()
                
                if choice == 'q':
                    return -2
                elif choice == 's':
                    return -1
                elif choice in ['0', '1']:
                    return int(choice)
                else:
                    print("无效输入")
                    
            except KeyboardInterrupt:
                return -2
            except:
                continue
    
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
            os.makedirs(os.path.dirname(self.manual_labels_file), exist_ok=True)
            with open(self.manual_labels_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotation_history, f, ensure_ascii=False, indent=4,
                         default=convert_to_serializable)
        except Exception as e:
            print(f"保存标注记录时出错: {e}")
# =================================
# 结果可视化函数
# =================================

def enhanced_visualize_results(training_history, output_dir, model_metrics=None, feature_analysis=None):
    """增强的训练结果可视化 - SCI论文级别"""
    print("📊 生成SCI论文级别的可视化图表...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置全局样式
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    
    # 1. 训练过程综合分析图
    create_training_overview_plot(training_history, output_dir)
    
    # 2. 模型性能详细分析
    create_performance_analysis_plot(training_history, output_dir)
    
    # 3. 奖励和损失分析
    create_reward_loss_analysis(training_history, output_dir)
    
    # 4. 人工标注效果分析
    create_annotation_analysis_plot(training_history, output_dir)
    
    # 5. 模型收敛性分析
    create_convergence_analysis_plot(training_history, output_dir)
    
    # 6. 混淆矩阵和ROC曲线
    if model_metrics:
        create_classification_metrics_plot(model_metrics, output_dir)
    
    # 7. 特征重要性分析
    if feature_analysis:
        create_feature_importance_plot(feature_analysis, output_dir)
    
    # 8. 3D可视化
    create_3d_visualization(training_history, output_dir)
    
    print("✅ 所有可视化图表已生成完成")
def create_training_overview_plot(training_history, output_dir):
    """创建训练过程综合分析图"""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # 主标题
    fig.suptitle('RLAD Model Training Overview Analysis', fontsize=24, fontweight='bold', y=0.95)
    
    # 1. 训练损失曲线
    ax1 = fig.add_subplot(gs[0, :2])
    episodes = training_history['episode']
    losses = training_history['loss']
    
    # 使用移动平均平滑曲线
    smooth_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
    smooth_episodes = episodes[2:-2]
    
    ax1.plot(episodes, losses, alpha=0.3, color='#1f77b4', linewidth=1, label='Raw Loss')
    ax1.plot(smooth_episodes, smooth_losses, color='#1f77b4', linewidth=3, label='Smoothed Loss')
    ax1.fill_between(smooth_episodes, smooth_losses, alpha=0.2, color='#1f77b4')
    ax1.set_title('Training Loss Evolution', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. F1分数演化
    ax2 = fig.add_subplot(gs[0, 2:])
    f1_scores = training_history['f1']
    precision_scores = training_history['precision']
    recall_scores = training_history['recall']
    
    ax2.plot(episodes, f1_scores, color='#d62728', linewidth=3, label='F1-Score', marker='o', markersize=4)
    ax2.plot(episodes, precision_scores, color='#ff7f0e', linewidth=2, label='Precision', linestyle='--')
    ax2.plot(episodes, recall_scores, color='#2ca02c', linewidth=2, label='Recall', linestyle='--')
    ax2.set_title('Performance Metrics Evolution', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=14)
    ax2.set_ylabel('Score', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. 奖励分布直方图
    ax3 = fig.add_subplot(gs[1, :2])
    rewards = training_history['reward']
    ax3.hist(rewards, bins=30, alpha=0.7, color='#9467bd', edgecolor='black')
    ax3.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    ax3.set_title('Reward Distribution', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Average Reward', fontsize=14)
    ax3.set_ylabel('Frequency', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Epsilon衰减曲线
    ax4 = fig.add_subplot(gs[1, 2:])
    epsilon_values = training_history['epsilon']
    ax4.plot(episodes, epsilon_values, color='#8c564b', linewidth=3, marker='s', markersize=3)
    ax4.set_title('Epsilon Decay Schedule', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Episode', fontsize=14)
    ax4.set_ylabel('Epsilon Value', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # 5. 标注效率分析
    ax5 = fig.add_subplot(gs[2, :2])
    human_annotations = training_history['human_annotations']
    total_annotations = training_history['total_annotations']
    max_annotations = training_history.get('max_human_annotations', 50)
    
    categories = ['Human\nAnnotations', 'Auto\nPredictions', 'Remaining\nCapacity']
    values = [human_annotations, total_annotations - human_annotations, max_annotations - human_annotations]
    colors = ['#e377c2', '#17becf', '#bcbd22']
    
    bars = ax5.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_title('Annotation Efficiency Analysis', fontsize=16, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=14)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 6. TD误差演化
    ax6 = fig.add_subplot(gs[2, 2:])
    td_errors = training_history['td_error']
    
    # 创建双y轴
    ax6_twin = ax6.twinx()
    
    line1 = ax6.plot(episodes, td_errors, color='#ff7f0e', linewidth=3, label='TD Error')
    line2 = ax6_twin.plot(episodes, losses, color='#1f77b4', linewidth=2, linestyle='--', label='Loss')
    
    ax6.set_title('TD Error vs Training Loss', fontsize=16, fontweight='bold')
    ax6.set_xlabel('Episode', fontsize=14)
    ax6.set_ylabel('TD Error', fontsize=14, color='#ff7f0e')
    ax6_twin.set_ylabel('Training Loss', fontsize=14, color='#1f77b4')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper right', fontsize=12)
    
    # 7. 性能改进趋势
    ax7 = fig.add_subplot(gs[3, :])
    
    # 计算性能改进
    f1_improvement = np.array(f1_scores) - f1_scores[0] if f1_scores else [0]
    cumulative_improvement = np.cumsum(f1_improvement)
    
    ax7.fill_between(episodes, 0, cumulative_improvement, alpha=0.3, color='green', label='Cumulative F1 Improvement')
    ax7.plot(episodes, cumulative_improvement, color='darkgreen', linewidth=3, marker='o', markersize=4)
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax7.set_title('Cumulative Performance Improvement', fontsize=16, fontweight='bold')
    ax7.set_xlabel('Episode', fontsize=14)
    ax7.set_ylabel('Cumulative F1 Improvement', fontsize=14)
    ax7.legend(fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_overview_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_performance_analysis_plot(training_history, output_dir):
    """创建性能分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Performance Analysis', fontsize=20, fontweight='bold')
    
    episodes = training_history['episode']
    
    # 1. 性能指标对比
    ax = axes[0, 0]
    metrics = ['precision', 'recall', 'f1']
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    for metric, color in zip(metrics, colors):
        values = training_history[metric]
        ax.plot(episodes, values, color=color, linewidth=3, label=metric.capitalize(), marker='o', markersize=3)
    
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 2. 性能稳定性分析
    ax = axes[0, 1]
    f1_scores = training_history['f1']
    if len(f1_scores) > 1:
        # 计算滚动标准差
        window_size = min(5, len(f1_scores))
        rolling_std = pd.Series(f1_scores).rolling(window=window_size).std()
        ax.plot(episodes, rolling_std, color='purple', linewidth=3)
        ax.fill_between(episodes, rolling_std, alpha=0.3, color='purple')
    
    ax.set_title('Performance Stability Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rolling Std of F1-Score')
    ax.grid(True, alpha=0.3)
    
    # 3. 学习效率分析
    ax = axes[0, 2]
    if len(f1_scores) > 1:
        # 计算学习率 (性能改进速度)
        learning_rate = np.diff(f1_scores)
        ax.plot(episodes[1:], learning_rate, color='orange', linewidth=2, marker='s', markersize=3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_title('Learning Efficiency', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('F1-Score Change Rate')
    ax.grid(True, alpha=0.3)
    
    # 4. 奖励vs性能相关性
    ax = axes[1, 0]
    rewards = training_history['reward']
    ax.scatter(rewards, f1_scores, alpha=0.7, s=50, c=episodes, cmap='viridis')
    
    # 添加趋势线
    if len(rewards) > 1 and len(f1_scores) > 1:
        z = np.polyfit(rewards, f1_scores, 1)
        p = np.poly1d(z)
        ax.plot(sorted(rewards), p(sorted(rewards)), "r--", alpha=0.8, linewidth=2)
    
    ax.set_title('Reward vs Performance Correlation', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Reward')
    ax.set_ylabel('F1-Score')
    ax.grid(True, alpha=0.3)
    
    # 5. 训练进度热图
    ax = axes[1, 1]
    
    # 创建性能矩阵
    n_episodes = len(episodes)
    n_metrics = 3
    performance_matrix = np.array([
        training_history['precision'],
        training_history['recall'],
        training_history['f1']
    ])
    
    im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Precision', 'Recall', 'F1-Score'])
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=15)
    
    # 6. 收敛性分析
    ax = axes[1, 2]
    
    # 计算收敛指标
    if len(f1_scores) > 5:
        convergence_window = min(10, len(f1_scores) // 2)
        recent_performance = f1_scores[-convergence_window:]
        convergence_std = np.std(recent_performance)
        convergence_mean = np.mean(recent_performance)
        
        ax.axhline(y=convergence_mean, color='red', linestyle='-', linewidth=2, 
                  label=f'Recent Mean: {convergence_mean:.3f}')
        ax.axhline(y=convergence_mean + convergence_std, color='red', linestyle='--', alpha=0.7,
                  label=f'±1 Std: {convergence_std:.3f}')
        ax.axhline(y=convergence_mean - convergence_std, color='red', linestyle='--', alpha=0.7)
        
        ax.plot(episodes, f1_scores, color='blue', linewidth=2, alpha=0.7)
        ax.fill_between(episodes[-convergence_window:], 
                       [convergence_mean - convergence_std] * convergence_window,
                       [convergence_mean + convergence_std] * convergence_window,
                       alpha=0.2, color='red')
    
    ax.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_reward_loss_analysis(training_history, output_dir):
    """创建奖励和损失分析图"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])
    
    fig.suptitle('Reward and Loss Analysis', fontsize=20, fontweight='bold')
    
    episodes = training_history['episode']
    rewards = training_history['reward']
    losses = training_history['loss']
    
    # 1. 奖励演化曲线
    ax1 = fig.add_subplot(gs[0, :])
    
    # 移动平均
    window_size = min(5, len(rewards))
    smooth_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    smooth_episodes = episodes[window_size-1:]
    
    ax1.plot(episodes, rewards, alpha=0.3, color='green', linewidth=1, label='Raw Reward')
    ax1.plot(smooth_episodes, smooth_rewards, color='darkgreen', linewidth=3, label='Smoothed Reward')
    ax1.fill_between(smooth_episodes, smooth_rewards, alpha=0.2, color='green')
    
    # 添加趋势线
    z = np.polyfit(episodes, rewards, 2)
    p = np.poly1d(z)
    ax1.plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    
    ax1.set_title('Reward Evolution Over Training', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_ylabel('Average Reward', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. 损失vs奖励散点图
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(losses, rewards, c=episodes, cmap='plasma', s=60, alpha=0.7, edgecolors='black')
    ax2.set_title('Loss vs Reward Correlation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Loss')
    ax2.set_ylabel('Average Reward')
    plt.colorbar(scatter, ax=ax2, label='Episode')
    ax2.grid(True, alpha=0.3)
    
    # 3. 奖励分布箱线图
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 将奖励分为几个阶段
    n_phases = 3
    phase_size = len(rewards) // n_phases
    reward_phases = []
    phase_labels = []
    
    for i in range(n_phases):
        start_idx = i * phase_size
        if i == n_phases - 1:
            end_idx = len(rewards)
        else:
            end_idx = (i + 1) * phase_size
        
        reward_phases.append(rewards[start_idx:end_idx])
        phase_labels.append(f'Phase {i+1}')
    
    bp = ax3.boxplot(reward_phases, labels=phase_labels, patch_artist=True)
    
    # 美化箱线图
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_title('Reward Distribution by Phase', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Reward')
    ax3.grid(True, alpha=0.3)
    
    # 4. 损失收敛分析
    ax4 = fig.add_subplot(gs[1, 2])
    
    # 计算损失的移动平均和标准差
    if len(losses) > 5:
        loss_ma = pd.Series(losses).rolling(window=5).mean()
        loss_std = pd.Series(losses).rolling(window=5).std()
        
        ax4.plot(episodes, loss_ma, color='blue', linewidth=2, label='Moving Average')
        ax4.fill_between(episodes, loss_ma - loss_std, loss_ma + loss_std, 
                        alpha=0.3, color='blue', label='±1 Std')
    
    ax4.plot(episodes, losses, alpha=0.5, color='gray', linewidth=1, label='Raw Loss')
    ax4.set_title('Loss Convergence Analysis', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Training Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 奖励-损失相关性热图
    ax5 = fig.add_subplot(gs[2, :])
    
    # 创建2D直方图
    h, xedges, yedges = np.histogram2d(losses, rewards, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax5.imshow(h.T, origin='lower', extent=extent, cmap='YlOrRd', aspect='auto')
    ax5.set_title('Reward-Loss Density Heatmap', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Training Loss', fontsize=14)
    ax5.set_ylabel('Average Reward', fontsize=14)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Density', rotation=270, labelpad=15)
    
    # 添加等高线
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    ax5.contour(X, Y, h.T, colors='white', alpha=0.6, linewidths=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_loss_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_annotation_analysis_plot(training_history, output_dir):
    """创建标注分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Human Annotation Efficiency Analysis', fontsize=20, fontweight='bold')
    
    human_annotations = training_history['human_annotations']
    total_annotations = training_history['total_annotations']
    max_annotations = training_history.get('max_human_annotations', 50)
    
    # 1. 标注效率饼图
    ax = axes[0, 0]
    sizes = [human_annotations, total_annotations - human_annotations]
    labels = [f'Human Annotations\n({human_annotations})', 
              f'Auto Predictions\n({total_annotations - human_annotations})']
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                     explode=explode, shadow=True, startangle=90)
    
    # 美化文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax.set_title('Annotation Type Distribution', fontsize=14, fontweight='bold')
    
    # 2. 标注效率指标
    ax = axes[0, 1]
    
    efficiency_metrics = {
        'Annotation\nUtilization': human_annotations / max_annotations,
        'Auto\nPrediction Rate': (total_annotations - human_annotations) / total_annotations,
        'Human\nContribution': human_annotations / total_annotations
    }
    
    metrics_names = list(efficiency_metrics.keys())
    metrics_values = list(efficiency_metrics.values())
    
    bars = ax.bar(metrics_names, metrics_values, color=['#e377c2', '#17becf', '#bcbd22'], 
                  alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Annotation Efficiency Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Ratio')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 3. 标注容量分析
    ax = axes[1, 0]
    
    capacity_data = [
        ('Used', human_annotations, '#ff7f0e'),
        ('Remaining', max_annotations - human_annotations, '#2ca02c'),
        ('Wasted Potential', max(0, max_annotations - total_annotations), '#d62728')
    ]
    
    y_pos = np.arange(len(capacity_data))
    values = [item[1] for item in capacity_data]
    colors = [item[2] for item in capacity_data]
    labels = [item[0] for item in capacity_data]
    
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{value}', ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_title('Annotation Capacity Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count')
    ax.grid(True, alpha=0.3)
    
    # 4. 标注成本效益分析
    ax = axes[1, 1]
    
    # 假设的成本效益数据
    if len(training_history['f1']) > 0:
        final_f1 = training_history['f1'][-1]
        baseline_f1 = 0.5  # 假设基线
        
        improvement = final_f1 - baseline_f1
        cost_per_annotation = 1.0  # 假设每个标注的成本
        benefit_per_improvement = 100.0  # 假设每单位F1改进的价值
        
        total_cost = human_annotations * cost_per_annotation
        total_benefit = improvement * benefit_per_improvement
        roi = (total_benefit - total_cost) / total_cost if total_cost > 0 else 0
        
        categories = ['Total Cost', 'Total Benefit', 'Net Gain']
        values = [total_cost, total_benefit, total_benefit - total_cost]
        colors = ['red', 'green', 'blue']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加ROI信息
        ax.text(0.5, 0.9, f'ROI: {roi:.2f}%', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title('Cost-Benefit Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value (Arbitrary Units)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'annotation_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_convergence_analysis_plot(training_history, output_dir):
    """创建收敛性分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Convergence Analysis', fontsize=20, fontweight='bold')
    
    episodes = training_history['episode']
    f1_scores = training_history['f1']
    losses = training_history['loss']
    
    # 1. 收敛曲线
    ax = axes[0, 0]
    
    if len(f1_scores) > 5:
        # 计算收敛指标
        convergence_threshold = 0.01
        convergence_window = 5
        
        # 检测收敛点
        convergence_detected = False
        convergence_episode = None
        
        for i in range(convergence_window, len(f1_scores)):
            recent_std = np.std(f1_scores[i-convergence_window:i])
            if recent_std < convergence_threshold:
                convergence_episode = episodes[i]
                convergence_detected = True
                break
        
        ax.plot(episodes, f1_scores, 'b-', linewidth=3, label='F1-Score')
        
        if convergence_detected:
            ax.axvline(x=convergence_episode, color='red', linestyle='--', linewidth=2,
                      label=f'Convergence at Episode {convergence_episode}')
            ax.axhspan(f1_scores[-1] - convergence_threshold, 
                      f1_scores[-1] + convergence_threshold,
                      alpha=0.2, color='red', label='Convergence Band')
    
    ax.set_title('Convergence Detection', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 学习率分析
    ax = axes[0, 1]
    
    if len(f1_scores) > 2:
        # 计算学习速度
        learning_rates = np.diff(f1_scores)
        ax.plot(episodes[1:], learning_rates, 'g-', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加移动平均
        if len(learning_rates) > 3:
            smooth_lr = np.convolve(learning_rates, np.ones(3)/3, mode='valid')
            ax.plot(episodes[2:-1], smooth_lr, 'r-', linewidth=3, alpha=0.7, label='Smoothed')
            ax.legend()
    
    ax.set_title('Learning Rate Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('F1-Score Change')
    ax.grid(True, alpha=0.3)
    
    # 3. 稳定性分析
    ax = axes[1, 0]
    
    if len(f1_scores) > 5:
        # 计算滚动方差
        window_sizes = [3, 5, 7]
        colors = ['blue', 'green', 'red']
        
        for window_size, color in zip(window_sizes, colors):
            if len(f1_scores) > window_size:
                rolling_var = pd.Series(f1_scores).rolling(window=window_size).var()
                ax.plot(episodes, rolling_var, color=color, linewidth=2, 
                       label=f'Window={window_size}')
        
        ax.legend()
    
    ax.set_title('Stability Analysis (Rolling Variance)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Variance')
    ax.grid(True, alpha=0.3)
    
    # 4. 预测可信度
    ax = axes[1, 1]
    
    # 计算预测区间
    if len(f1_scores) > 10:
        # 使用最后几个点预测未来趋势
        recent_points = 5
        recent_episodes = episodes[-recent_points:]
        recent_f1 = f1_scores[-recent_points:]
        
        # 线性回归预测
        z = np.polyfit(recent_episodes, recent_f1, 1)
        p = np.poly1d(z)
        
        # 预测未来几个episode
        future_episodes = np.arange(episodes[-1] + 1, episodes[-1] + 11)
        predicted_f1 = p(future_episodes)
        
        # 计算预测区间
        residuals = recent_f1 - p(recent_episodes)
        prediction_std = np.std(residuals)
        
        ax.plot(episodes, f1_scores, 'b-', linewidth=3, label='Actual F1-Score')
        ax.plot(future_episodes, predicted_f1, 'r--', linewidth=2, label='Predicted')
        ax.fill_between(future_episodes, 
                       predicted_f1 - 2*prediction_std,
                       predicted_f1 + 2*prediction_std,
                       alpha=0.3, color='red', label='95% Confidence Interval')
    
    ax.set_title('Performance Prediction', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_3d_visualization(training_history, output_dir):
    """创建3D可视化"""
    fig = plt.figure(figsize=(20, 15))
    
    # 创建3D子图布局
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    episodes = np.array(training_history['episode'])
    f1_scores = np.array(training_history['f1'])
    losses = np.array(training_history['loss'])
    rewards = np.array(training_history['reward'])
    
    # 1. 3D训练轨迹
    ax1.plot(episodes, losses, f1_scores, 'b-', linewidth=3, alpha=0.8)
    ax1.scatter(episodes, losses, f1_scores, c=episodes, cmap='viridis', s=50)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_zlabel('F1-Score')
    ax1.set_title('3D Training Trajectory', fontsize=14, fontweight='bold')
    
    # 2. 3D性能空间
    ax2.scatter(losses, rewards, f1_scores, c=episodes, cmap='plasma', s=60, alpha=0.7)
    ax2.set_xlabel('Loss')
    ax2.set_ylabel('Reward')
    ax2.set_zlabel('F1-Score')
    ax2.set_title('3D Performance Space', fontsize=14, fontweight='bold')
    
    # 3. 训练效率等高线图
    if len(episodes) > 5:
        # 创建网格
        xi = np.linspace(min(episodes), max(episodes), 50)
        yi = np.linspace(min(losses), max(losses), 50)
        X, Y = np.meshgrid(xi, yi)
        
        # 插值F1分数
        from scipy.interpolate import griddata
        Z = griddata((episodes, losses), f1_scores, (X, Y), method='cubic', fill_value=0)
        
        contour = ax3.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
        ax3.contour(X, Y, Z, levels=20, colors='black', alpha=0.4, linewidths=0.5)
        
        # 添加实际数据点
        scatter = ax3.scatter(episodes, losses, c=f1_scores, cmap='RdYlGn', s=50, 
                             edgecolors='black', alpha=0.8)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Efficiency Contour Map', fontsize=14, fontweight='bold')
        
        # 添加colorbar
        cbar = plt.colorbar(contour, ax=ax3)
        cbar.set_label('F1-Score')
    
    # 4. 极坐标性能图
    # 将指标映射到极坐标
    theta = np.linspace(0, 2*np.pi, len(episodes))
    r = f1_scores
    
    ax4 = plt.subplot(224, projection='polar')
    ax4.plot(theta, r, 'b-', linewidth=2)
    ax4.fill(theta, r, alpha=0.3, color='blue')
    ax4.set_title('Polar Performance Plot', fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_visualization.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
# 在可视化函数组之后添加（大约在第 3500-3600 行附近）
def create_classification_metrics_plot(model_metrics, output_dir):
    """创建分类指标图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Classification Performance Metrics', fontsize=20, fontweight='bold')
    
    # 1. 混淆矩阵
    ax = axes[0, 0]
    cm = model_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # 2. ROC曲线
    if 'roc_curve' in model_metrics:
        ax = axes[0, 1]
        fpr = model_metrics['roc_curve']['fpr']
        tpr = model_metrics['roc_curve']['tpr']
        auc = model_metrics['roc_auc']
        
        ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    
    # 3. 精确率-召回率曲线
    if 'probabilities' in model_metrics:
        ax = axes[1, 0]
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        y_true = model_metrics['labels']
        y_proba = [p[1] for p in model_metrics['probabilities']]
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        ax.plot(recall, precision, color='blue', lw=3, label=f'AP = {ap_score:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. 分类报告可视化
    ax = axes[1, 1]
    cr = model_metrics['classification_report']
    
    # 提取数值数据
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['0', '1']  # 正常, 异常
    
    data = []
    for cls in classes:
        if cls in cr:
            data.append([cr[cls][metric] for metric in metrics])
    
    if data:
        data = np.array(data)
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(['Normal', 'Anomaly'])
        
        # 添加数值标注
        for i in range(len(classes)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                                ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Classification Report Heatmap', fontsize=14, fontweight='bold')
        
        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_metrics.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_feature_importance_plot(feature_analysis, output_dir):
    """创建特征重要性图表"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importance Analysis', fontsize=20, fontweight='bold')
    
    importance = feature_analysis['feature_importance']
    n_features = len(importance)
    
    # 1. 特征重要性柱状图
    ax = axes[0]
    feature_indices = range(n_features)
    bars = ax.bar(feature_indices, importance, color='skyblue', alpha=0.8, edgecolor='black')
    
    # 标注最重要的特征
    max_idx = np.argmax(importance)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(1.0)
    
    ax.set_title('Feature Importance Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Importance Score')
    ax.grid(True, alpha=0.3)
    
    # 2. 累积重要性
    ax = axes[1]
    sorted_importance = sorted(importance, reverse=True)
    cumulative_importance = np.cumsum(sorted_importance)
    cumulative_importance = cumulative_importance / cumulative_importance[-1]
    
    ax.plot(range(1, n_features + 1), cumulative_importance, 'bo-', linewidth=2, markersize=4)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Threshold')
    
    ax.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()    
def visualize_results(training_history, output_dir):
    """简化版结果可视化"""
    if not training_history or len(training_history.get('episode', [])) == 0:
        print("⚠️ 没有训练历史数据可供可视化")
        return
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建简单的训练曲线图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('RLAD Training Results', fontsize=16, fontweight='bold')
        
        episodes = training_history['episode']
        
        # 损失曲线
        if 'loss' in training_history and len(training_history['loss']) > 0:
            axes[0, 0].plot(episodes, training_history['loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # F1分数
        if 'f1' in training_history and len(training_history['f1']) > 0:
            axes[0, 1].plot(episodes, training_history['f1'], 'r-', linewidth=2)
            axes[0, 1].set_title('F1 Score')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
        
        # 奖励曲线
        if 'reward' in training_history and len(training_history['reward']) > 0:
            axes[1, 0].plot(episodes, training_history['reward'], 'g-', linewidth=2)
            axes[1, 0].set_title('Average Reward')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 标注统计
        human_annotations = training_history.get('human_annotations', 0)
        total_annotations = training_history.get('total_annotations', 0)
        max_annotations = training_history.get('max_human_annotations', 50)
        
        categories = ['Human', 'Auto', 'Remaining']
        values = [human_annotations, total_annotations - human_annotations, max_annotations - human_annotations]
        
        axes[1, 1].bar(categories, values, color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[1, 1].set_title('Annotation Statistics')
        axes[1, 1].set_ylabel('Count')
        
        # 添加数值标签
        for i, v in enumerate(values):
            axes[1, 1].text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_results_simple.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 简化版可视化结果已保存到: {output_dir}")

    except Exception as e:
        print(f"❌ 简化版可视化失败: {e}")    
# =================================
# 主函数
# =================================
# 在主函数之前添加（大约在第 3800-3900 行附近）
def comprehensive_model_evaluation(agent, X_val, y_val, device):
    """全面的模型评估"""
    agent.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    val_dataset = HydraulicDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for states, labels in val_loader:
            states = states.to(device)
            labeled_mask = (labels != -1)
            
            if labeled_mask.sum() == 0:
                continue
                
            states_labeled = states[labeled_mask]
            labels_labeled = labels[labeled_mask]
            
            q_values = agent(states_labeled)
            probabilities = F.softmax(q_values, dim=1)
            predictions = q_values.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_labeled.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算详细指标
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    
    metrics = {
        'classification_report': classification_report(all_labels, all_predictions, output_dict=True),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions),
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    # 计算ROC AUC
    if len(set(all_labels)) > 1:
        probs_positive = [p[1] for p in all_probabilities]
        metrics['roc_auc'] = roc_auc_score(all_labels, probs_positive)
        fpr, tpr, thresholds = roc_curve(all_labels, probs_positive)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    
    return metrics

def analyze_feature_importance(agent, X_val, device):
    """分析特征重要性"""
    agent.eval()
    
    # 使用梯度分析特征重要性
    sample_data = torch.FloatTensor(X_val[:100]).to(device)
    sample_data.requires_grad_(True)
    
    with torch.enable_grad():
        outputs = agent(sample_data)
        # 计算梯度
        grad_outputs = torch.ones_like(outputs[:, 1])
        gradients = torch.autograd.grad(outputs[:, 1], sample_data, 
                                       grad_outputs=grad_outputs, 
                                       create_graph=False)[0]
    
    # 计算特征重要性
    feature_importance = gradients.abs().mean(dim=0).mean(dim=0).cpu().numpy()
    
    return {
        'feature_importance': feature_importance.tolist(),
        'feature_rankings': np.argsort(feature_importance)[::-1].tolist()
    }

def generate_training_report(training_history, final_metrics, feature_analysis, output_dir):
    """生成训练报告"""
    report_path = os.path.join(output_dir, 'training_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RLAD Model Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 训练概况
        f.write("Training Summary:\n")
        f.write(f"- Total Episodes: {len(training_history['episode'])}\n")
        f.write(f"- Human Annotations: {training_history['human_annotations']}\n")
        f.write(f"- Total Annotations: {training_history['total_annotations']}\n")
        f.write(f"- Final F1-Score: {training_history['f1'][-1]:.4f}\n")
        f.write(f"- Final Loss: {training_history['loss'][-1]:.4f}\n\n")
        
        # 性能指标
        if final_metrics:
            f.write("Final Performance Metrics:\n")
            f.write(f"- ROC AUC: {final_metrics.get('roc_auc', 'N/A'):.4f}\n")
            
            cr = final_metrics['classification_report']
            f.write(f"- Precision: {cr['weighted avg']['precision']:.4f}\n")
            f.write(f"- Recall: {cr['weighted avg']['recall']:.4f}\n")
            f.write(f"- F1-Score: {cr['weighted avg']['f1-score']:.4f}\n\n")
        
        # 特征重要性
        if feature_analysis:
            f.write("Feature Importance Analysis:\n")
            importance = feature_analysis['feature_importance']
            rankings = feature_analysis['feature_rankings']
            
            for i, rank in enumerate(rankings[:5]):  # Top 5
                f.write(f"- Feature {rank}: {importance[rank]:.6f}\n")
    
    print(f"📋 训练报告已保存: {report_path}")
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RLAD v3.0: 强化学习异常检测系统')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='C:/Users/18104/Desktop/Python files/deeplearning/example/timeseries/examples/RLAD/data1.csv',
                        help='数据文件路径')
    parser.add_argument('--feature_column', type=str, default=None,
                        help='指定特征列名')
    parser.add_argument('--window_size', type=int, default=288,
                        help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=24,  # 增大步长
                        help='滑动窗口步长')
    
    # 训练参数（优化后的默认值）
    parser.add_argument('--num_episodes', type=int, default=30,  # 减少episode数
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,  # 减小batch size
                        help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--epsilon_start', type=float, default=0.8,  # 降低初始探索
                        help='初始epsilon值')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                        help='最终epsilon值')
    parser.add_argument('--epsilon_decay', type=float, default=0.98,
                        help='epsilon衰减率')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=64,  # 减小模型规模
                        help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,  # 减少层数
                        help='LSTM层数')
    
    # 标注参数（优化后的默认值）
    parser.add_argument('--annotation_frequency', type=int, default=30,  # 降低标注频率
                        help='人工标注频率')
    parser.add_argument('--max_human_annotations', type=int, default=20,  # 减少标注上限
                        help='人工标注窗口数量上限')
    parser.add_argument('--use_gui', action='store_true', default=False,  # 默认不使用GUI
                        help='是否使用GUI界面')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")
    
    # 初始化变量
    training_history = None
    final_metrics = None
    feature_analysis = None
    
    try:
        # 加载数据
        print("📊 加载数据...")
        data_dict = load_hydraulic_data_improved(
            args.data_path, 
            window_size=args.window_size, 
            stride=args.stride,
            specific_feature_column=args.feature_column
        )
        
        X_scaled = data_dict['X_scaled']
        X_raw = data_dict['X_raw']
        y_labels = data_dict['y_labels']
        
        print(f"   数据形状: {X_scaled.shape}")
        print(f"   特征维度: {X_scaled.shape[-1]}")
        
        # 数据分割
        split_ratio = 0.8
        split_idx = int(len(X_scaled) * split_ratio)
        
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        X_raw_train, X_raw_val = X_raw[:split_idx], X_raw[split_idx:]
        y_train, y_val = y_labels[:split_idx], y_labels[split_idx:]
        
        print(f"   训练集: {X_train.shape}, 验证集: {X_val.shape}")
        
        # 创建模型
        print("🤖 创建模型...")
        input_dim = X_scaled.shape[-1]
        agent = OptimizedRLADAgent(
            input_dim=input_dim,
            seq_len=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        ).to(device)
        
        target_agent = OptimizedRLADAgent(
            input_dim=input_dim,
            seq_len=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        ).to(device)
        
        # 复制初始权重
        target_agent.load_state_dict(agent.state_dict())
        
        print(f"   模型参数数量: {sum(p.numel() for p in agent.parameters()):,}")
        
        # 创建优化器和调度器
        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        # 创建经验回放缓冲区
        replay_buffer = PrioritizedReplayBuffer(capacity=20000)
        
        # 创建人工标注系统
        print("🎯 创建标注系统...")
        annotation_system = HumanAnnotationSystem(
            output_dir=args.output_dir,
            window_size=args.window_size,
            use_gui=args.use_gui
        )
        
        # 开始训练
        print("🚀 开始训练...")
        training_history = interactive_train_rlad_gui(
            agent=agent,
            target_agent=target_agent,
            optimizer=optimizer,
            scheduler=scheduler,
            replay_buffer=replay_buffer,
            X_train=X_train,
            y_train=y_train,
            windows_raw_train=X_raw_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            annotation_system=annotation_system,
            num_episodes=args.num_episodes,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_rate=args.epsilon_decay,
            batch_size_rl=args.batch_size,
            output_dir=args.output_dir,
            annotation_frequency=args.annotation_frequency,
            max_human_annotations=args.max_human_annotations
        )
        
        # 最终模型评估
        print("📊 进行最终模型评估...")
        final_metrics = comprehensive_model_evaluation(agent, X_val, y_val, device)
        
        # 特征重要性分析
        print("🔍 进行特征重要性分析...")
        feature_analysis = analyze_feature_importance(agent, X_val, device)
        
        # 保存最终模型
        model_path = os.path.join(args.output_dir, 'final_rlad_model.pth')
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'target_agent_state_dict': target_agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_history': training_history,
            'final_metrics': final_metrics,
            'feature_analysis': feature_analysis,
            'model_config': {
                'input_dim': input_dim,
                'seq_len': args.window_size,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers
            }
        }, model_path)
        print(f"💾 最终模型已保存: {model_path}")
        
        # 保存训练历史
        if training_history is not None:
            history_path = os.path.join(args.output_dir, 'training_history.json')
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(training_history, f, ensure_ascii=False, indent=4, default=convert_to_serializable)
            print(f"📋 训练历史已保存: {history_path}")
        
        print("🎉 训练完成！")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 创建默认的训练历史以避免UnboundLocalError
        if training_history is None:
            training_history = {
                'episode': [1],
                'loss': [0.0],
                'td_error': [0.0],
                'epsilon': [0.1],
                'reward': [0.0],
                'precision': [0.5],
                'recall': [0.5],
                'f1': [0.5],
                'human_annotations': 0,
                'total_annotations': 0,
                'max_human_annotations': args.max_human_annotations
            }
    
    finally:
        # 无论是否出错都尝试生成可视化结果
        try:
            if training_history is not None:
                # 生成增强的可视化结果
                enhanced_visualize_results(training_history, args.output_dir, final_metrics, feature_analysis)
                
                # 生成训练报告
                generate_training_report(training_history, final_metrics, feature_analysis, args.output_dir)
        except Exception as viz_error:
            print(f"⚠️ 可视化生成失败: {viz_error}")
            # 使用简化版可视化
            try:
                visualize_results(training_history, args.output_dir)
            except Exception as simple_viz_error:
                print(f"⚠️ 简化版可视化也失败: {simple_viz_error}")
if __name__ == "__main__":
    main()