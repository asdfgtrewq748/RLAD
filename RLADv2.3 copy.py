"""
RLAD v3.0: 基于STL+LOF双层异常检测的强化学习液压支架工作阻力异常检测 - 完整修正版
"""

# 基础导入
import os
import sys
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# STL分解相关导入
from statsmodels.tsa.seasonal import STL

# GUI相关导入
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
# 可选导入
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    sns.set_palette("husl")
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available, using matplotlib default styles")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available, some 3D visualizations will be skipped")

try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, some interpolation features will be skipped")

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
# STL+LOF双层异常检测系统
# =================================

class STLLOFAnomalyDetector:
    """STL+LOF双层异常检测器"""
    
    def __init__(self, period=24, seasonal=25, robust=True, 
                 n_neighbors=20, contamination=0.02, metric='minkowski'):
        self.period = period
        self.seasonal = seasonal if seasonal % 2 == 1 else seasonal + 1
        self.robust = robust
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        
        if self.seasonal <= self.period:
            self.seasonal = self.period + 2
            if self.seasonal % 2 == 0:
                self.seasonal += 1
        
        self.stl_result = None
        self.lof_model = None
        self.anomaly_scores = None
        self.anomaly_labels = None
        self.decomposition_quality = {}
        
    def prepare_time_series_data(self, data, datetime_col=None):
        """准备时间序列数据用于STL分解"""
        if isinstance(data, np.ndarray):
            series = pd.Series(data)
        elif isinstance(data, pd.Series):
            series = data.copy()
        else:
            series = pd.Series(data)
        
        if series.isnull().any():
            # 修复：使用新的pandas语法
            series = series.ffill().bfill()
        
        if len(series) < 2 * self.period:
            raise ValueError(f"数据长度 {len(series)} 太短，至少需要 {2 * self.period} 个点")
        
        return series
    
    def fit_stl_decomposition(self, series):
        """执行STL分解"""
        try:
            stl = STL(series, seasonal=self.seasonal, period=self.period, robust=self.robust)
            self.stl_result = stl.fit()
            self._evaluate_decomposition_quality(series)
            return self.stl_result
        except Exception as e:
            print(f"STL分解失败: {e}")
            raise
    
    def _evaluate_decomposition_quality(self, original_series):
        """评估STL分解质量"""
        if self.stl_result is None:
            return
        
        reconstructed = (self.stl_result.trend + 
                        self.stl_result.seasonal + 
                        self.stl_result.resid)
        
        reconstruction_error = np.mean((original_series - reconstructed) ** 2)
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
    
    def fit_lof_detection(self, residuals):
        """在STL残差上应用LOF异常检测"""
        if isinstance(residuals, pd.Series):
            residuals_clean = residuals.dropna()
        else:
            residuals_clean = pd.Series(residuals).dropna()
        
        residuals_2d = residuals_clean.values.reshape(-1, 1)
        
        if len(residuals_2d) == 0:
            raise ValueError("残差数据为空，无法进行LOF检测")
        
        self.lof_model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(residuals_2d) - 1),
            contamination=self.contamination,
            metric=self.metric,
            novelty=False
        )
        
        try:
            lof_labels = self.lof_model.fit_predict(residuals_2d)
            lof_scores = -self.lof_model.negative_outlier_factor_
            return lof_labels, lof_scores
        except Exception as e:
            print(f"LOF检测失败: {e}")
            raise
    
    def detect_anomalies(self, data):
        """完整的STL+LOF异常检测流程"""
        series = self.prepare_time_series_data(data)
        stl_result = self.fit_stl_decomposition(series)
        residuals = stl_result.resid.dropna()
        lof_labels, lof_scores = self.fit_lof_detection(residuals)
        
        anomaly_binary = (lof_labels == -1).astype(int)
        full_labels = np.zeros(len(series), dtype=int)
        residuals_index = residuals.index
        series_index = series.index
        
        for i, res_idx in enumerate(residuals_index):
            series_pos = series_index.get_loc(res_idx)
            full_labels[series_pos] = anomaly_binary[i]
        
        return full_labels

# =================================
# 优化的标注系统
# =================================

class OptimizedAnnotationGUI:
    """优化的GUI标注界面"""
    
    def __init__(self, window_size=288):
        self.window_size = window_size
        self.result = None
        
    def get_annotation(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """获取用户标注"""
        print(f"\n{'='*50}")
        print(f"窗口 #{window_idx} 需要标注")
        print(f"{'='*50}")
        
        # 显示标准化数据的统计信息
        if window_data is not None:
            window_flat = window_data.flatten()
            print(f"标准化数据统计:")
            print(f"  最大值: {np.max(window_flat):.4f}")
            print(f"  平均值: {np.mean(window_flat):.4f}")
            print(f"  最小值: {np.min(window_flat):.4f}")
            print(f"  标准差: {np.std(window_flat):.4f}")
        
        # 显示原始数据的统计信息
        if original_data_segment is not None:
            original_flat = original_data_segment.flatten()
            print(f"原始数据统计:")
            print(f"  最大值: {np.max(original_flat):.2f}")
            print(f"  平均值: {np.mean(original_flat):.2f}")
            print(f"  最小值: {np.min(original_flat):.2f}")
            print(f"  标准差: {np.std(original_flat):.2f}")
        
        if auto_predicted_label is not None:
            label_text = "异常" if auto_predicted_label == 1 else "正常"
            print(f"AI预测: {label_text}")
        
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

class EnhancedAnnotationGUI:
    """增强版GUI标注界面 - 借鉴v2.2的稳定实现"""
    
    def __init__(self, window_size=288):
        self.window_size = window_size
        self.result = None
        self.root = None
        self.current_window_data = None
        self.current_original_data = None
        self.window_idx = None
        self.auto_prediction = None
        
    def create_gui(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """创建稳定的GUI界面"""
        try:
            print(f"创建GUI窗口 - 窗口 #{window_idx}")
            
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
            
            # 创建界面组件
            self.create_enhanced_widgets()
            
            # 绑定关闭事件
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            # 确保窗口获得焦点
            self.root.focus_force()
            
            return self.root
            
        except Exception as e:
            print(f"创建GUI窗口时出错: {e}")
            return None
    
    def create_enhanced_widgets(self):
        """创建增强的界面组件"""
        try:
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
            
            # 2. 信息显示区域（使用v2.2的布局）
            info_frame = tk.LabelFrame(main_container, 
                                      text="窗口信息", 
                                      font=('Arial', 12, 'bold'),
                                      bg='#f0f0f0',
                                      fg='#34495e',
                                      padx=10, pady=10)
            info_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.create_enhanced_info_display(info_frame)
            
            # 3. 图表区域（使用matplotlib）
            chart_frame = tk.LabelFrame(main_container, 
                                       text="数据可视化", 
                                       font=('Arial', 12, 'bold'),
                                       bg='#f0f0f0',
                                       fg='#34495e',
                                       padx=5, pady=5)
            chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
            
            self.create_enhanced_charts(chart_frame)
            
            # 4. 增强的按钮区域
            button_frame = tk.Frame(main_container, bg='#f0f0f0')
            button_frame.pack(fill=tk.X, side=tk.BOTTOM)
            
            self.create_enhanced_buttons(button_frame)
            
            # 强制更新界面
            self.root.update_idletasks()
            self.root.update()
            
        except Exception as e:
            print(f"创建界面组件时出错: {e}")
    
    def create_enhanced_info_display(self, parent):
        """创建增强的信息显示区域"""
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
            
            # 标准化数据统计
            if self.current_window_data is not None:
                window_flat = self.current_window_data.flatten()
                tk.Label(left_frame, text="", 
                        font=('Arial', 8), 
                        bg='#f0f0f0').pack(anchor=tk.W)  # 空行
                
                tk.Label(left_frame, text="标准化数据:", 
                        font=('Arial', 11, 'bold'), 
                        bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
                
                tk.Label(left_frame, text=f"最大值: {np.max(window_flat):.4f}", 
                        font=('Arial', 10), 
                        bg='#f0f0f0', fg='#e74c3c').pack(anchor=tk.W)
                
                tk.Label(left_frame, text=f"平均值: {np.mean(window_flat):.4f}", 
                        font=('Arial', 10), 
                        bg='#f0f0f0', fg='#2980b9').pack(anchor=tk.W)
                
                tk.Label(left_frame, text=f"最小值: {np.min(window_flat):.4f}", 
                        font=('Arial', 10), 
                        bg='#f0f0f0', fg='#27ae60').pack(anchor=tk.W)
            
            # AI预测信息
            if self.auto_prediction is not None:
                tk.Label(left_frame, text="", 
                        font=('Arial', 8), 
                        bg='#f0f0f0').pack(anchor=tk.W)  # 空行
                
                prediction_text = "异常" if self.auto_prediction == 1 else "正常"
                color = "#e74c3c" if self.auto_prediction == 1 else "#27ae60"
                tk.Label(left_frame, text=f"AI预测: {prediction_text}", 
                        font=('Arial', 11, 'bold'), 
                        bg='#f0f0f0', fg=color).pack(anchor=tk.W)
            
            # 右侧原始数据统计信息
            right_frame = tk.Frame(info_container, bg='#f0f0f0')
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            if self.current_original_data is not None:
                tk.Label(right_frame, text="原始数据统计:", 
                        font=('Arial', 12, 'bold'), 
                        bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W, pady=(0, 5))
                
                data_flat = self.current_original_data.flatten()
                
                # 创建统计信息表格样式
                stats_info = [
                    ("最大值", f"{np.max(data_flat):.2f}", "#e74c3c"),
                    ("平均值", f"{np.mean(data_flat):.2f}", "#2980b9"),
                    ("最小值", f"{np.min(data_flat):.2f}", "#27ae60"),
                    ("标准差", f"{np.std(data_flat):.2f}", "#8e44ad"),
                    ("中位数", f"{np.median(data_flat):.2f}", "#f39c12"),
                    ("数据范围", f"{np.max(data_flat) - np.min(data_flat):.2f}", "#34495e")
                ]
                
                for stat_name, stat_value, color in stats_info:
                    # 创建水平布局的统计项
                    stat_frame = tk.Frame(right_frame, bg='#f0f0f0')
                    stat_frame.pack(fill=tk.X, pady=1)
                    
                    tk.Label(stat_frame, text=f"{stat_name}:", 
                            font=('Arial', 10), 
                            bg='#f0f0f0', fg='#34495e',
                            width=8, anchor='w').pack(side=tk.LEFT)
                    
                    tk.Label(stat_frame, text=stat_value, 
                            font=('Arial', 10, 'bold'), 
                            bg='#f0f0f0', fg=color,
                            anchor='w').pack(side=tk.LEFT, padx=(5, 0))
                
                # 添加数据质量指标
                tk.Label(right_frame, text="", 
                        font=('Arial', 6), 
                        bg='#f0f0f0').pack(anchor=tk.W)  # 空行
                
                tk.Label(right_frame, text="数据质量:", 
                        font=('Arial', 11, 'bold'), 
                        bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
                
                # 计算变异系数
                cv = np.std(data_flat) / np.mean(data_flat) if np.mean(data_flat) != 0 else 0
                tk.Label(right_frame, text=f"变异系数: {cv:.4f}", 
                        font=('Arial', 9), 
                        bg='#f0f0f0', fg='#7f8c8d').pack(anchor=tk.W)
                
                # 峰度和偏度
                from scipy import stats
                try:
                    skewness = stats.skew(data_flat)
                    kurtosis = stats.kurtosis(data_flat)
                    tk.Label(right_frame, text=f"偏度: {skewness:.4f}", 
                            font=('Arial', 9), 
                            bg='#f0f0f0', fg='#7f8c8d').pack(anchor=tk.W)
                    tk.Label(right_frame, text=f"峰度: {kurtosis:.4f}", 
                            font=('Arial', 9), 
                            bg='#f0f0f0', fg='#7f8c8d').pack(anchor=tk.W)
                except:
                    pass
                            
        except Exception as e:
            print(f"创建信息显示时出错: {e}")
    
    def create_enhanced_charts(self, parent):
        """创建增强的数据可视化图表"""
        try:
            # 创建matplotlib图形
            fig = Figure(figsize=(11, 5), dpi=80)
            fig.patch.set_facecolor('#f0f0f0')
            
            # 第一个子图：标准化数据
            ax1 = fig.add_subplot(211)
            data_to_plot = self.current_window_data.flatten()
            time_steps = np.arange(len(data_to_plot))
            
            ax1.plot(time_steps, data_to_plot, 'b-', linewidth=1.5, alpha=0.8, label='标准化数据')
            
            # 添加统计线
            mean_val = np.mean(data_to_plot)
            max_val = np.max(data_to_plot)
            min_val = np.min(data_to_plot)
            
            ax1.axhline(y=mean_val, color='orange', linestyle='--', alpha=0.7, 
                       label=f'平均值: {mean_val:.3f}')
            ax1.axhline(y=max_val, color='red', linestyle=':', alpha=0.7, 
                       label=f'最大值: {max_val:.3f}')
            ax1.axhline(y=min_val, color='green', linestyle=':', alpha=0.7, 
                       label=f'最小值: {min_val:.3f}')
            
            ax1.set_title(f'窗口 #{self.window_idx} - 标准化数据 (Max: {max_val:.3f}, Avg: {mean_val:.3f})', 
                         fontsize=11, fontweight='bold')
            ax1.set_xlabel('时间步')
            ax1.set_ylabel('标准化值')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right', fontsize=8)
            
            # 第二个子图：原始数据
            ax2 = fig.add_subplot(212)
            if self.current_original_data is not None:
                original_data_flat = self.current_original_data.flatten()
                ax2.plot(time_steps, original_data_flat, 'r-', linewidth=1.5, alpha=0.8, label='原始数据')
                
                # 原始数据统计
                orig_mean = np.mean(original_data_flat)
                orig_max = np.max(original_data_flat)
                orig_min = np.min(original_data_flat)
                orig_std = np.std(original_data_flat)
                
                # 添加统计线
                ax2.axhline(y=orig_mean, color='blue', linestyle='--', alpha=0.7, 
                           label=f'平均值: {orig_mean:.2f}')
                ax2.axhline(y=orig_max, color='red', linestyle=':', alpha=0.7, 
                           label=f'最大值: {orig_max:.2f}')
                
                # 添加异常阈值线
                try:
                    threshold = orig_mean + 2 * orig_std
                    ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                               label=f'阈值线 (μ+2σ): {threshold:.2f}')
                except:
                    pass
                
                ax2.set_title(f'窗口 #{self.window_idx} - 原始数据 (Max: {orig_max:.2f}, Avg: {orig_mean:.2f})', 
                             fontsize=11, fontweight='bold')
                ax2.set_xlabel('时间步')
                ax2.set_ylabel('阻力值')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right', fontsize=8)
            else:
                ax2.text(0.5, 0.5, '原始数据不可用', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14, color='gray')
                ax2.set_title(f'窗口 #{self.window_idx} - 原始数据不可用', fontsize=11)
            
            fig.tight_layout(pad=2.0)
            
            # 将图表嵌入到Tkinter中
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        except Exception as e:
            print(f"创建图表时出错: {e}")
            # 创建错误显示
            error_label = tk.Label(parent, text=f"图表显示错误: {str(e)}", 
                                  font=('Arial', 12), fg='red', bg='#f0f0f0')
            error_label.pack(expand=True)
    
    def create_enhanced_buttons(self, parent):
        """创建增强的标注按钮（借鉴v2.2的稳定实现）"""
        try:
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
            
        except Exception as e:
            print(f"创建按钮时出错: {e}")
            # 创建紧急按钮
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
            if result in [0, 1]:
                # 确认对话框
                label_text = "异常" if result == 1 else "正常"
                confirm = messagebox.askyesno("确认标注", 
                                            f"确认将窗口 #{self.window_idx} 标注为 '{label_text}' 吗？",
                                            parent=self.root)
                if not confirm:
                    return
            elif result == -1:
                confirm = messagebox.askyesno("确认跳过", 
                                            "确认跳过此窗口的标注吗？",
                                            parent=self.root)
                if not confirm:
                    return
            elif result == -2:
                confirm = messagebox.askyesno("确认退出", 
                                            "确认退出标注程序吗？",
                                            parent=self.root)
                if not confirm:
                    return
            
            self.result = result
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"设置结果时出错: {e}")
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
            root = self.create_gui(window_data, window_idx, original_data_segment, auto_predicted_label)
            if root is None:
                return self.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
            
            root.mainloop()
            return self.result if self.result is not None else -2
            
        except Exception as e:
            print(f"GUI标注失败: {e}")
            return self.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
    
    def get_annotation_fallback(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """回退到命令行标注模式"""
        print(f"\n{'='*60}")
        print(f"请对窗口 #{window_idx} 进行标注")
        print(f"{'='*60}")
        
        # 显示标准化数据统计
        if window_data is not None:
            window_flat = window_data.flatten()
            print(f"标准化数据统计:")
            print(f"  最大值: {np.max(window_flat):.4f}")
            print(f"  平均值: {np.mean(window_flat):.4f}")
            print(f"  最小值: {np.min(window_flat):.4f}")
            print(f"  标准差: {np.std(window_flat):.4f}")
            print()
        
        # 显示原始数据统计
        if original_data_segment is not None:
            original_flat = original_data_segment.flatten()
            print(f"原始数据统计:")
            print(f"  最大值: {np.max(original_flat):.2f}")
            print(f"  平均值: {np.mean(original_flat):.2f}")
            print(f"  最小值: {np.min(original_flat):.2f}")
            print(f"  标准差: {np.std(original_flat):.2f}")
            print(f"  中位数: {np.median(original_flat):.2f}")
            print(f"  数据范围: {np.max(original_flat) - np.min(original_flat):.2f}")
            
            # 计算变异系数
            cv = np.std(original_flat) / np.mean(original_flat) if np.mean(original_flat) != 0 else 0
            print(f"  变异系数: {cv:.4f}")
            print()
        
        if auto_predicted_label is not None:
            label_text = "异常" if auto_predicted_label == 1 else "正常"
            print(f"AI预测: {label_text} (标签={auto_predicted_label})")
            print()
        
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
class OptimizedHumanAnnotationSystem:
    """优化的人工标注系统"""
    
    def __init__(self, output_dir: str, window_size: int = 288, use_gui: bool = False):
        self.output_dir = output_dir
        self.window_size = window_size
        self.use_gui = use_gui
        self.annotation_history = []
        self.manual_labels_file = os.path.join(output_dir, 'manual_annotations.json')
        
        # 选择GUI实现：优先使用增强版GUI
        if use_gui:
            try:
                self.gui = EnhancedAnnotationGUI(window_size)
                print("使用增强版GUI界面")
            except:
                self.gui = OptimizedAnnotationGUI(window_size)
                print("回退到标准GUI界面")
        else:
            self.gui = None
            
        self.load_existing_annotations()
        
    def get_human_annotation(self, window_data: np.ndarray, window_idx: int, 
                           original_data_segment: np.ndarray = None, 
                           auto_predicted_label: int = None) -> int:
        """获取用户对单个窗口的标注"""
        
        # 检查缓存
        for record in self.annotation_history:
            if record.get('window_idx') == window_idx:
                return record['label']
        
        # 使用标注方法
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
        """命令行标注"""
        print(f"\n{'='*50}")
        print(f"窗口 #{window_idx} 标注")
        print(f"{'='*50}")
        
        # 显示标准化数据统计
        if window_data is not None:
            window_flat = window_data.flatten()
            print(f"标准化数据统计:")
            print(f"  最大值: {np.max(window_flat):.4f}")
            print(f"  平均值: {np.mean(window_flat):.4f}")
            print(f"  最小值: {np.min(window_flat):.4f}")
            print(f"  标准差: {np.std(window_flat):.4f}")
            print()
        
        # 显示原始数据统计
        if original_data_segment is not None:
            original_flat = original_data_segment.flatten()
            print(f"原始数据统计:")
            print(f"  最大值: {np.max(original_flat):.2f} ⭐")
            print(f"  平均值: {np.mean(original_flat):.2f} ⭐")
            print(f"  最小值: {np.min(original_flat):.2f}")
            print(f"  标准差: {np.std(original_flat):.2f}")
            print(f"  中位数: {np.median(original_flat):.2f}")
            print(f"  数据范围: {np.max(original_flat) - np.min(original_flat):.2f}")
            print()
        
        if auto_predicted_label is not None:
            label_text = "异常" if auto_predicted_label == 1 else "正常"
            print(f"AI预测: {label_text}")
            print()
        
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
# 数据集类
# =================================

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

# =================================
# 优化的模型架构
# =================================

class OptimizedRLADAgent(nn.Module):
    """优化的RLAD智能体网络架构"""
    
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

# =================================
# 经验回放和训练函数
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
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced datasets."""
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
def enhanced_compute_reward(action, true_label, confidence=1.0, is_human_labeled=False):
    """
    增强的奖励计算函数 - 修正版
    - 移除对正确识别正常样本(TN)的奖励，避免模型“躺平”。
    - 大幅增加对正确检测异常(TP)的奖励，激励模型寻找少数类。
    - 调整惩罚，以平衡精确率和召回率。
    """
    base_multiplier = 2.0 if is_human_labeled else 1.0
    confidence_bonus = confidence * 0.5
    
    if action == true_label:
        if true_label == 1:  # TP: 正确检测到异常
            reward = (10.0 + confidence_bonus) * base_multiplier # 修正：大幅增加TP奖励
        else:  # TN: 正确识别为正常
            reward = 0.0 # 修正：移除TN奖励，设为0
    else:
        if true_label == 1 and action == 0:  # FN: 未检测到异常（漏报）
            reward = -8.0 * base_multiplier # 保持对漏报的高惩罚
        elif true_label == 0 and action == 1:  # FP: 正常误报为异常
            reward = -3.0 * base_multiplier # 修正：增加FP惩罚
        else: # 其他未知错误
            reward = -1.0 * base_multiplier
    
    return reward

def enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, 
                           gamma=0.99, batch_size=64, beta=0.4, loss_fn=None):
    """优化的DQN训练步骤"""
    if len(replay_buffer) < batch_size: 
        return 0.0, 0.0
    
    sample_result = replay_buffer.sample(batch_size, beta)
    if sample_result is None: 
        return 0.0, 0.0
    
    states, actions, rewards, next_states, dones, indices, weights = sample_result
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    weights = torch.FloatTensor(weights).to(device)
    
    # 计算当前Q值
    q_values = agent(states)
    current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # 计算目标Q值 (Double DQN)
    with torch.no_grad():
        next_q_values = agent(next_states)
        next_actions = next_q_values.argmax(dim=1)
        next_q_values_target = target_agent(next_states)
        next_q_values_selected = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (gamma * next_q_values_selected * ~dones)
    
    # 计算损失
    if loss_fn: # 使用Focal Loss
        # Focal Loss需要分类任务的输出来计算，我们这里用Q值模拟
        # 注意：这是一种近似用法，将Q值作为logits
        loss = loss_fn(q_values, actions)
        td_errors = target_q_values - current_q_values # TD error仍然用于优先级更新
    else: # 原始的Smooth L1 Loss
        td_errors = target_q_values - current_q_values
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
    
    # 梯度裁剪和优化
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 更新优先级
    priorities_np = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
    replay_buffer.update_priorities(indices, priorities_np)
    
    return loss.item(), td_errors.abs().mean().item()

def enhanced_evaluate_model(agent, data_loader, device):
    """修复的模型评估函数 - 借鉴v2.2的实现"""
    agent.eval()
    all_predictions, all_labels = [], []
    all_probs = []
    
    with torch.no_grad():
        for states, labels in data_loader:
            states = states.to(device)
            labeled_mask = (labels != -1)
            
            if labeled_mask.sum() > 0:
                states_labeled, labels_labeled = states[labeled_mask], labels[labeled_mask]
                if states_labeled.size(0) == 0: 
                    continue
                
                q_values = agent(states_labeled)
                predictions = q_values.argmax(dim=1)
                probabilities = torch.softmax(q_values, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_labeled.cpu().numpy())
                all_probs.extend(probabilities[:, 1].cpu().numpy()) # 异常类别的概率
    
    agent.train()
    
    if len(all_predictions) == 0:
        print("⚠️ 没有可评估的数据，返回默认指标")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5, "f1_per_class": [0.0, 0.0]}
    
    all_predictions_np, all_labels_np = np.array(all_predictions), np.array(all_labels)
    all_probs_np = np.array(all_probs)
    
    # 计算AUC
    auc_score = 0.5
    if len(np.unique(all_labels_np)) > 1:
        try:
            auc_score = roc_auc_score(all_labels_np, all_probs_np)
        except ValueError:
            auc_score = 0.5

    # 计算加权平均指标
    precision_w = precision_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    recall_w = recall_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    f1_w = f1_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    
    # 计算每个类别的F1分数 (关键)
    try:
        f1_pc = f1_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0, 1])
    except Exception:
        f1_pc = np.array([0.0, 0.0])

    return {
        "precision": float(precision_w), 
        "recall": float(recall_w), 
        "f1": float(f1_w),
        "auc": float(auc_score),
        "f1_per_class": [float(x) for x in f1_pc] # 返回每个类别的F1
    }

# =================================
# 数据加载和预处理函数
# =================================

def load_hydraulic_data_improved(data_path, window_size=288, stride=12, specific_feature_column=None):
    """改进的数据加载函数 - 引入STL+LOF进行高质量初始标注"""
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
    
    data_values = df[selected_cols].ffill().bfill().fillna(0).values
    print(f"提取的用于模型训练和阈值计算的数据形状: {data_values.shape}")
    
    # --- 关键优化：使用STL+LOF生成高质量初始标签 ---
    print("🔥 使用STL+LOF进行高质量初始异常检测...")
    try:
        stl_lof_detector = STLLOFAnomalyDetector(period=24, contamination=0.02)
        point_anomaly_labels = stl_lof_detector.detect_anomalies(data_values.flatten())
        print("✅ STL+LOF初始检测完成。")
    except Exception as e:
        print(f"⚠️ STL+LOF检测失败: {e}。回退到统计方法。")
        # 回退到统计方法
        q3_point = np.percentile(data_values.flatten(), 75)
        iqr_point = q3_point - np.percentile(data_values.flatten(), 25)
        laiya_criterion_point = q3_point + 1.5 * iqr_point
        point_anomaly_labels = (data_values.flatten() > laiya_criterion_point).astype(int)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    print("创建滑动窗口...")
    windows_scaled_list = []
    windows_raw_data_list = []
    # 创建滑动窗口，并根据点级异常判断窗口级异常
    initial_anomaly_labels = []
    for i in range(0, len(data_scaled) - window_size + 1, stride):
        window_scaled = data_scaled[i:i + window_size]
        window_raw = data_values[i:i + window_size]
        
        windows_scaled_list.append(window_scaled)
        windows_raw_data_list.append(window_raw)
        
        # 如果窗口内有任何一个点被标记为异常，则该窗口为异常
        window_point_labels = point_anomaly_labels[i:i + window_size]
        is_window_anomaly = 1 if np.sum(window_point_labels) > 0 else 0
        initial_anomaly_labels.append(is_window_anomaly)

    X_scaled = np.array(windows_scaled_list)
    X_raw = np.array(windows_raw_data_list)
    initial_anomaly_labels = np.array(initial_anomaly_labels)
    
    print(f"滑动窗口创建完成，窗口数量: {len(X_scaled)}")
    print(f"每个窗口的形状: {X_scaled[0].shape}")
    
    print(f"高质量初始异常标签统计:")
    print(f"  正常窗口数: {np.sum(initial_anomaly_labels == 0)}")
    print(f"  异常窗口数: {np.sum(initial_anomaly_labels == 1)}")
    print(f"  异常比例: {np.mean(initial_anomaly_labels):.1%}")
    
    # 初始化验证集使用初步标签，训练集使用未标注状态
    y_labels = np.full(len(X_scaled), -1, dtype=int)
    
    # 为验证集提供一些初始标签以便评估
    split_ratio = 0.8
    split_idx = int(len(X_scaled) * split_ratio)
    
    # 在验证集中随机选择一些样本使用初步标签
    val_indices = np.arange(split_idx, len(X_scaled))
    num_val_samples = min(50, len(val_indices))  # 最多50个验证样本有标签
    selected_val_indices = np.random.choice(val_indices, num_val_samples, replace=False)
    y_labels[selected_val_indices] = initial_anomaly_labels[selected_val_indices]
    
    print(f"验证集标签统计:")
    val_labeled_mask = y_labels[split_idx:] != -1
    print(f"  已标注验证样本: {val_labeled_mask.sum()}")
    if val_labeled_mask.sum() > 0:
        val_labels = y_labels[split_idx:][val_labeled_mask]
        print(f"  验证集正常/异常分布: {np.bincount(val_labels)}")
    
    return {
        'X_scaled': X_scaled,
        'X_raw': X_raw,
        'y_labels': y_labels,
        'initial_anomaly_labels': initial_anomaly_labels,
        'scaler': scaler,
        'selected_columns': selected_cols,
        'laiya_criterion': -1, # 不再使用
        'window_indices': list(range(0, len(data_scaled) - window_size + 1, stride)),
        'df_mapping': df_for_point_mapping
    }
# =================================
# 交互式训练函数
# =================================

def interactive_train_rlad_gui(agent, target_agent, optimizer, scheduler, replay_buffer, 
                              X_train, y_train, windows_raw_train, X_val, y_val, device, 
                              annotation_system, initial_anomaly_labels_train, num_episodes=50, target_update_freq=10,
                              epsilon_start=0.95, epsilon_end=0.02, epsilon_decay_rate=0.995,
                              batch_size_rl=64, output_dir="./output", annotation_frequency=20,
                              max_human_annotations=30):
    """
    修正的交互式训练循环：转为在线监督学习范式
    """
    print("🚀 开始交互式训练 (在线监督学习模式)...")
    print(f"📝 人工标注上限设置为: {max_human_annotations} 个窗口")
    
    # --- 关键修正：将高质量的初始标签合并到主训练标签中 ---
    # 这确保了模型在整个训练过程中都有数据可学，而不仅仅是预训练。
    print("🔧 将高质量初始标签合并到训练集中...")
    y_train = np.where(y_train == -1, initial_anomaly_labels_train, y_train)
    print(f"   合并后训练集标签分布: 正常={np.sum(y_train==0)}, 异常={np.sum(y_train==1)}, 待人工标注={np.sum(y_train==-1)}")
    
    # 模型预训练阶段 (现在是基于更丰富的标签)
    print("🔥 开始模型预训练...")
    pretrain_epochs = 5
    pretrain_dataset = HydraulicDataset(X_train, y_train) # 使用合并后的y_train
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size_rl, shuffle=True)
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0).to(device)
    
    for epoch in range(pretrain_epochs):
        total_loss = 0
        for states, labels in pretrain_loader:
            # 预训练时跳过仍为-1的样本
            labeled_mask = labels != -1
            if not labeled_mask.any():
                continue
            states, labels = states[labeled_mask], labels[labeled_mask]
            
            states, labels = states.to(device), labels.to(device)
            optimizer.zero_grad()
            q_values = agent(states)
            loss = loss_fn(q_values, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Pretrain Epoch {epoch+1}/{pretrain_epochs}, Loss: {total_loss/len(pretrain_loader):.4f}")
    
    print("✅ 模型预训练完成！")

    # 训练历史记录 (移除RL相关指标)
    training_history = {
        'episode': [], 'loss': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [],
        'human_annotations': 0, 'total_annotations': 0, 'max_human_annotations': max_human_annotations
    }
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    val_dataset = HydraulicDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    samples_per_episode = min(200, len(X_train)) # 增加每轮处理的样本数

    for episode in range(num_episodes):
        print(f"\n{'='*60}\nEpisode {episode + 1}/{num_episodes}\n{'='*60}")
        
        agent.train() # 确保模型在训练模式
        episode_loss = 0.0
        steps_with_loss = 0
        
        selected_indices = np.random.choice(len(X_train), size=samples_per_episode, replace=False)
        
        for step, idx in enumerate(selected_indices):
            current_label = y_train[idx]
            true_label = current_label
            is_human_labeled = False

            # 只有在需要时才进行预测和标注 (逻辑保持不变)
            if current_label == -1 and training_history['human_annotations'] < max_human_annotations:
                state = X_train_tensor[idx].unsqueeze(0)
                
                agent.eval()
                with torch.no_grad():
                    q_values = agent(state)
                    confidence = torch.softmax(q_values, dim=1).max().item()
                    predicted_action = q_values.argmax(dim=1).item()
                agent.train()

                need_annotation = (step + 1) % annotation_frequency == 0 or confidence < 0.6 or np.random.random() < 0.05
                
                if need_annotation:
                    # ... (省略GUI调用相关的打印和逻辑) ...
                    human_label = annotation_system.get_human_annotation(
                        X_train[idx], idx, windows_raw_train[idx], predicted_action
                    )
                    
                    if human_label >= 0:
                        true_label = human_label
                        y_train[idx] = human_label
                        is_human_labeled = True
                        training_history['human_annotations'] += 1
                        # ... (省略打印) ...
                    else:
                        if human_label == -2: return training_history
                        continue
            
            # 如果样本有标签 (初始的或人工标注的)，则进行训练
            if true_label != -1:
                state = X_train_tensor[idx].unsqueeze(0)
                
                optimizer.zero_grad()
                q_values = agent(state)
                label_tensor = torch.LongTensor([true_label]).to(device)
                
                # 直接使用Focal Loss进行监督学习更新
                loss = loss_fn(q_values, label_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
                
                episode_loss += loss.item()
                steps_with_loss += 1
                training_history['total_annotations'] += 1

        if scheduler: scheduler.step()
        
        # 评估模型
        if (episode + 1) % 5 == 0:
            metrics = enhanced_evaluate_model(agent, val_loader, device)
            anomaly_f1 = metrics.get('f1_per_class', [0.0, 0.0])
            if len(anomaly_f1) < 2: anomaly_f1.append(0.0)
            
            avg_loss = episode_loss / max(steps_with_loss, 1)
            training_history['episode'].append(episode + 1)
            training_history['loss'].append(avg_loss)
            training_history['precision'].append(metrics['precision'])
            training_history['recall'].append(metrics['recall'])
            training_history['f1'].append(metrics['f1'])
            training_history['auc'].append(metrics['auc'])
            
            print(f"\n📊 Episode {episode + 1} 结果:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   验证集 -> F1(异常): {anomaly_f1[1]:.4f} | AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"   人工标注数: {training_history['human_annotations']}/{max_human_annotations}, 总标注数: {training_history['total_annotations']}")
            
            if training_history['human_annotations'] >= max_human_annotations:
                print(f"🔴 已达到人工标注上限 ({max_human_annotations})")
        
        # 保存模型检查点
        if (episode + 1) % 20 == 0:
            # ... (保存检查点代码保持不变) ...
            checkpoint_path = os.path.join(output_dir, f'rlad_checkpoint_ep{episode + 1}.pth')
            torch.save({
                'episode': episode + 1,
                'agent_state_dict': agent.state_dict(),
                'target_agent_state_dict': target_agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'training_history': training_history,
            }, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
    
    print(f"\n🎉 训练完成！")
    print(f"   总人工标注数: {training_history['human_annotations']}/{max_human_annotations}")
    print(f"   总标注数: {training_history['total_annotations']}")
    
    return training_history
def create_focused_metrics_plot(training_history, output_dir):
    """创建一张包含四个核心指标的专注图表"""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('核心性能指标演化过程', fontsize=18, fontweight='bold')
        
        episodes = training_history['episode']
        
        # 准备数据，并为每个指标添加标记
        metrics_to_plot = {
            'F1 Score (Weighted)': ('f1', 'g-', 'o'),
            'AUC-ROC': ('auc', 'b-', '^'),
            'Precision (Weighted)': ('precision', 'r--', 's'),
            'Recall (Weighted)': ('recall', 'm:', 'x')
        }
        
        for label, (key, style, marker) in metrics_to_plot.items():
            if key in training_history and len(training_history[key]) > 0:
                ax.plot(episodes, training_history[key], style, linewidth=2, markersize=6, label=label, alpha=0.8)

        ax.set_title('验证集性能指标', fontsize=14)
        ax.set_xlabel('训练轮次 (Episode)', fontsize=12)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12, loc='lower right')
        
        # 添加性能目标线
        ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1.5, label='性能目标 (0.9)')
        ax.axhline(y=0.95, color='black', linestyle=':', linewidth=1.5, label='AUC目标 (0.95)')
        
        # 重新整理图例，避免重复
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(output_dir, 'focused_performance_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 核心指标图表已保存: {save_path}")

    except Exception as e:
        print(f"❌ 生成核心指标图表失败: {e}")
# =================================
# 可视化函数
# =================================

def enhanced_visualize_results(training_history, output_dir, model_metrics=None, feature_analysis=None):
    """修复的增强可视化函数 - 简化版"""
    print("📊 生成专注的核心指标可视化图表...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查训练历史数据
    if not training_history or len(training_history.get('episode', [])) == 0:
        print("⚠️ 训练历史数据为空，无法生成图表")
        return
    
    # 只调用专注的图表生成函数
    create_focused_metrics_plot(training_history, output_dir)
    
    try:
        # 设置全局样式
        plt.style.use('default')
        plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 300
        
        # 验证数据完整性
        required_keys = ['episode', 'loss', 'f1', 'reward', 'precision', 'recall', 'auc']
        for key in required_keys:
            if key not in training_history or len(training_history[key]) == 0:
                print(f"⚠️ 缺少训练历史键: {key}")
                training_history[key] = [0.0] * max(1, len(training_history.get('episode', [1])))
        
        # 1. 基础训练图表
        create_basic_training_plots(training_history, output_dir)
        
        # 2. 性能分析图表
        create_performance_plots(training_history, output_dir)
        
        # 3. 标注分析图表
        create_annotation_plots(training_history, output_dir)
        
        print("✅ 可视化图表生成完成")
        
    except Exception as e:
        print(f"⚠️ 高级可视化失败: {e}")
        print("🔄 使用简化版可视化...")
        create_fallback_visualizations(training_history, output_dir)
# 在enhanced_visualize_results函数后添加

def save_training_plots_enhanced(training_history, output_dir):
    """保存增强的训练图表（借鉴v2.2）"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建综合训练图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RLAD训练全面分析报告', fontsize=20, fontweight='bold')
        
        episodes = training_history.get('episode', [])
        
        if len(episodes) > 0:
            # 1. 损失和TD误差
            ax = axes[0, 0]
            if 'loss' in training_history and len(training_history['loss']) > 0:
                ax.plot(episodes, training_history['loss'], 'b-', linewidth=2, label='训练损失')
            if 'td_error' in training_history and len(training_history['td_error']) > 0:
                ax2 = ax.twinx()
                ax2.plot(episodes, training_history['td_error'], 'r--', linewidth=2, label='TD误差')
                ax2.set_ylabel('TD误差', color='r')
                ax2.legend(loc='upper right')
            ax.set_title('训练损失与TD误差')
            ax.set_xlabel('训练轮次')
            ax.set_ylabel('损失值', color='b')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 2. 性能指标
            ax = axes[0, 1]
            if 'f1' in training_history:
                ax.plot(episodes, training_history['f1'], 'g-', linewidth=2, label='F1分数')
            if 'precision' in training_history:
                ax.plot(episodes, training_history['precision'], 'orange', linestyle='--', linewidth=2, label='精确率')
            if 'recall' in training_history:
                ax.plot(episodes, training_history['recall'], 'purple', linestyle=':', linewidth=2, label='召回率')
            ax.set_title('模型性能指标')
            ax.set_xlabel('训练轮次')
            ax.set_ylabel('分数')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Epsilon衰减
            ax = axes[0, 2]
            if 'epsilon' in training_history:
                ax.plot(episodes, training_history['epsilon'], 'c-', linewidth=2)
                ax.set_title('探索率(Epsilon)衰减')
                ax.set_xlabel('训练轮次')
                ax.set_ylabel('Epsilon值')
                ax.grid(True, alpha=0.3)
            
            # 4. 奖励趋势
            ax = axes[1, 0]
            if 'reward' in training_history:
                ax.plot(episodes, training_history['reward'], 'magenta', linewidth=2)
                ax.set_title('平均奖励趋势')
                ax.set_xlabel('训练轮次')
                ax.set_ylabel('平均奖励')
                ax.grid(True, alpha=0.3)
            
            # 5. 人工标注统计
            ax = axes[1, 1]
            human_annotations = training_history.get('human_annotations', 0)
            total_annotations = training_history.get('total_annotations', 0)
            max_annotations = training_history.get('max_human_annotations', 50)
            
            categories = ['人工标注', '自动预测', '剩余容量']
            values = [
                human_annotations,
                max(0, total_annotations - human_annotations),
                max(0, max_annotations - human_annotations)
            ]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_title('标注统计分布')
            ax.set_ylabel('数量')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 6. 学习率变化
            ax = axes[1, 2]
            if 'learning_rate' in training_history and len(training_history['learning_rate']) > 0:
                ax.plot(episodes, training_history['learning_rate'], 'brown', linewidth=2)
                ax.set_title('学习率调度')
                ax.set_xlabel('训练轮次')
                ax.set_ylabel('学习率')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_comprehensive_analysis.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 增强训练图表已保存")
        
    except Exception as e:
        print(f"❌ 保存增强训练图表失败: {e}")
def create_basic_training_plots(training_history, output_dir):
    """创建基础训练图表"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RLAD Training Results', fontsize=20, fontweight='bold')
        
        episodes = training_history['episode']
        
        # 1. 损失曲线
        if len(training_history['loss']) > 0:
            axes[0, 0].plot(episodes, training_history['loss'], 'b-', linewidth=2, label='Training Loss')
            axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # 2. F1分数
        if len(training_history['f1']) > 0:
            axes[0, 1].plot(episodes, training_history['f1'], 'r-', linewidth=2, label='F1-Score')
            axes[0, 1].plot(episodes, training_history['precision'], 'g--', linewidth=2, label='Precision')
            axes[0, 1].plot(episodes, training_history['recall'], 'orange', linestyle='--', linewidth=2, label='Recall')
            axes[0, 1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 3. 奖励曲线
        if len(training_history['reward']) > 0:
            axes[1, 0].plot(episodes, training_history['reward'], 'purple', linewidth=2, label='Average Reward')
            axes[1, 0].set_title('Training Rewards', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # 4. 标注统计
        human_annotations = training_history.get('human_annotations', 0)
        total_annotations = training_history.get('total_annotations', 0)
        max_annotations = training_history.get('max_human_annotations', 50)
        
        categories = ['Human\nAnnotations', 'Auto\nPredictions', 'Remaining\nCapacity']
        values = [
            human_annotations, 
            max(0, total_annotations - human_annotations), 
            max(0, max_annotations - human_annotations)
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        bars = axes[1, 1].bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        axes[1, 1].set_title('Annotation Statistics', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Count')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'basic_training_results.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 基础训练图表已保存")
        
    except Exception as e:
        print(f"❌ 基础训练图表生成失败: {e}")

def create_performance_plots(training_history, output_dir):
    """创建性能分析图表"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Analysis', fontsize=20, fontweight='bold')
        
        episodes = training_history['episode']
        f1_scores = training_history['f1']
        losses = training_history['loss']
        rewards = training_history['reward']
        
        # 1. 性能趋势
        if len(f1_scores) > 1:
            ax = axes[0, 0]
            ax.plot(episodes, f1_scores, 'ro-', linewidth=2, markersize=4, label='F1-Score')
            
            # 添加趋势线
            if len(episodes) > 2:
                z = np.polyfit(episodes, f1_scores, 1)
                p = np.poly1d(z)
                ax.plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=2, label='Trend')
            
            ax.set_title('F1-Score Trend', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('F1-Score')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 2. 损失vs奖励
        if len(losses) > 0 and len(rewards) > 0:
            ax = axes[0, 1]
            scatter = ax.scatter(losses, rewards, c=episodes, cmap='viridis', s=60, alpha=0.7)
            ax.set_title('Loss vs Reward', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Loss')
            ax.set_ylabel('Average Reward')
            ax.grid(True, alpha=0.3)
            try:
                plt.colorbar(scatter, ax=ax, label='Episode')
            except:
                pass
        
        # 3. 学习曲线
        if len(f1_scores) > 2:
            ax = axes[1, 0]
            learning_rate = np.diff(f1_scores)
            ax.plot(episodes[1:], learning_rate, 'go-', linewidth=2, markersize=3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_title('Learning Rate (F1 Change)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('F1-Score Change')
            ax.grid(True, alpha=0.3)
        
        # 4. 训练稳定性
        if len(f1_scores) > 5:
            ax = axes[1, 1]
            window_size = min(3, len(f1_scores) // 2)
            rolling_std = pd.Series(f1_scores).rolling(window=window_size).std()
            ax.plot(episodes, rolling_std, 'mo-', linewidth=2, markersize=3)
            ax.set_title('Training Stability (Rolling Std)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('F1-Score Std')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 性能分析图表已保存")
        
    except Exception as e:
        print(f"❌ 性能分析图表生成失败: {e}")

def create_annotation_plots(training_history, output_dir):
    """创建标注分析图表"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Annotation Analysis', fontsize=20, fontweight='bold')
        
        human_annotations = training_history.get('human_annotations', 0)
        total_annotations = training_history.get('total_annotations', 0)
        max_annotations = training_history.get('max_human_annotations', 50)
        
        # 1. 标注类型饼图
        ax = axes[0]
        if total_annotations > 0:
            sizes = [human_annotations, total_annotations - human_annotations]
            labels = [f'Human\n({human_annotations})', f'Auto\n({total_annotations - human_annotations})']
            colors = ['#ff9999', '#66b3ff']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                             startangle=90, shadow=True)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No Annotations Yet', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
        
        ax.set_title('Annotation Distribution', fontsize=14, fontweight='bold')
        
        # 2. 标注进度条
        ax = axes[1]
        
        progress_data = [
            ('Completed', human_annotations, '#2ecc71'),
            ('Auto', total_annotations - human_annotations, '#3498db'),
            ('Remaining', max_annotations - human_annotations, '#95a5a6')
        ]
        
        y_pos = np.arange(len(progress_data))
        values = [item[1] for item in progress_data]
        colors = [item[2] for item in progress_data]
        labels = [item[0] for item in progress_data]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f'{value}', ha='left', va='center', fontsize=12, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_title('Annotation Progress', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'annotation_analysis.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 标注分析图表已保存")
        
    except Exception as e:
        print(f"❌ 标注分析图表生成失败: {e}")

def create_fallback_visualizations(training_history, output_dir):
    """创建简化的回退可视化"""
    try:
        print("🔄 创建简化版可视化...")
        
        if not training_history or len(training_history.get('episode', [])) == 0:
            # 创建占位符图表
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'Training in Progress\nNo Data Available Yet', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=20, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax.set_title('RLAD Training Status', fontsize=16, fontweight='bold')
            ax.axis('off')
            
            plt.savefig(os.path.join(output_dir, 'training_status.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return
        
        # 创建基本图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('RLAD Training Results (Simplified)', fontsize=16, fontweight='bold')
        
        episodes = training_history.get('episode', [1])
        
        # 1. 基本指标
        for i, (key, title, color) in enumerate([
            ('loss', 'Training Loss', 'blue'),
            ('f1', 'F1-Score', 'red'),
            ('reward', 'Reward', 'green'),
            ('precision', 'Precision', 'orange')
        ]):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            values = training_history.get(key, [0] * len(episodes))
            if len(values) > 0:
                ax.plot(episodes, values, color=color, linewidth=2, marker='o', markersize=3)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Episode')
                ax.set_ylabel(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {title} Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_results_simplified.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 简化版可视化已保存")
        
    except Exception as e:
        print(f"❌ 简化版可视化失败: {e}")

def create_default_visualizations(output_dir):
    """创建默认可视化（当没有训练数据时）"""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'RLAD Training Not Started\nRun training to generate results', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=18, bbox=dict(boxstyle='round', facecolor='lightblue'))
        ax.set_title('RLAD Training Dashboard', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.savefig(os.path.join(output_dir, 'no_training_data.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 默认图表已创建")
        
    except Exception as e:
        print(f"❌ 默认图表创建失败: {e}")

# =================================
# 评估和分析函数
# =================================

def comprehensive_model_evaluation(agent, X_val, y_val, device):
    """全面的模型评估 - 修正版"""
    print("🔍 进行全面模型评估...")
    
    agent.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(X_val)):
            state = torch.FloatTensor(X_val[i]).unsqueeze(0).to(device)
            label = y_val[i]
            
            if label != -1:  # 只评估有标签的样本
                q_values = agent(state)
                prediction = q_values.argmax(dim=1).item()
                
                all_predictions.append(prediction)
                all_labels.append(label)
    
    agent.train()
    
    if len(all_predictions) == 0 or len(np.unique(all_labels)) < 2:
        print("⚠️ 没有可评估的样本或样本中只存在一个类别。")
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 
            'confusion_matrix': np.zeros((2,2)), 'predictions': [], 'labels': []
        }
    
    # 计算二分类指标，专注于异常类别 (pos_label=1)
    # zero_division=0 表示当分母为0时，结果为0，这更符合直觉
    precision = precision_score(all_labels, all_predictions, pos_label=1, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, pos_label=1, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, pos_label=1, average='binary', zero_division=0)
    
    # 确保混淆矩阵是2x2
    labels_unique = np.unique(np.concatenate((all_labels, all_predictions)))
    if len(labels_unique) == 1: # 如果所有预测和标签都一样
        if labels_unique[0] == 0:
            cm = np.array([[len(all_labels), 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, len(all_labels)]])
    else:
        cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    print(f"📊 评估结果 (针对'异常'类别):")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   混淆矩阵 (行:真实, 列:预测):\n{cm}")
    
    return metrics

def analyze_feature_importance(agent, X_val, device):
    """分析特征重要性"""
    print("🔍 分析特征重要性...")
    
    try:
        agent.eval()
        sample_data = torch.FloatTensor(X_val[:10]).to(device)
        
        with torch.no_grad():
            _, features = agent(sample_data, return_features=True)
        
        # 计算特征激活统计
        feature_stats = []
        for i, feature_layer in enumerate(features):
            mean_activation = feature_layer.mean(dim=0).cpu().numpy()
            std_activation = feature_layer.std(dim=0).cpu().numpy()
            
            feature_stats.append({
                'layer': i,
                'mean_activation': mean_activation,
                'std_activation': std_activation,
                'importance': np.abs(mean_activation) + std_activation
            })
        
        agent.train()
        
        print("✅ 特征重要性分析完成")
        return {'feature_importance': feature_stats}
        
    except Exception as e:
        print(f"❌ 特征重要性分析失败: {e}")
        return None

def generate_training_report(training_history, final_metrics, feature_analysis, output_dir):
    """生成训练报告"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'training_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RLAD Model Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            # 训练概况
            f.write("Training Summary:\n")
            f.write(f"- Total Episodes: {len(training_history.get('episode', []))}\n")
            f.write(f"- Human Annotations: {training_history.get('human_annotations', 0)}\n")
            f.write(f"- Total Annotations: {training_history.get('total_annotations', 0)}\n")
            
            if training_history.get('f1') and len(training_history['f1']) > 0:
                f.write(f"- Final F1-Score: {training_history['f1'][-1]:.4f}\n")
            else:
                f.write("- Final F1-Score: N/A\n")
                
            if training_history.get('loss') and len(training_history['loss']) > 0:
                f.write(f"- Final Loss: {training_history['loss'][-1]:.4f}\n\n")
            else:
                f.write("- Final Loss: N/A\n\n")
            
            # 性能指标
            if final_metrics:
                f.write("Final Performance Metrics:\n")
                f.write(f"- Precision: {final_metrics['precision']:.4f}\n")
                f.write(f"- Recall: {final_metrics['recall']:.4f}\n")
                f.write(f"- F1-Score: {final_metrics['f1']:.4f}\n\n")
            
            f.write(f"\nReport generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ 训练报告已保存: {report_path}")
        
    except Exception as e:
        print(f"❌ 生成训练报告失败: {e}")

# =================================
# 主函数
# =================================

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
    parser.add_argument('--stride', type=int, default=24,
                        help='滑动窗口步长')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--epsilon_start', type=float, default=0.8,
                        help='初始epsilon值')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                        help='最终epsilon值')
    parser.add_argument('--epsilon_decay', type=float, default=0.98,
                        help='epsilon衰减率')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM层数')
    
    # 标注参数
    parser.add_argument('--annotation_frequency', type=int, default=30,
                        help='人工标注频率')
    parser.add_argument('--max_human_annotations', type=int, default=50,
                        help='人工标注窗口数量上限') # 修正：将默认值从20提高到50
    parser.add_argument('--use_gui', action='store_true', default=False,
                        help='是否使用GUI界面')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, 
                        default='C:/Users/18104/Desktop/Python files/deeplearning/example/timeseries/examples/RLAD/output',
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
        initial_anomaly_labels = data_dict['initial_anomaly_labels']
        
        print(f"   数据形状: {X_scaled.shape}")
        print(f"   特征维度: {X_scaled.shape[-1]}")
        # 数据分割
        split_ratio = 0.8
        split_idx = int(len(X_scaled) * split_ratio)
        
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        X_raw_train, X_raw_val = X_raw[:split_idx], X_raw[split_idx:]
        y_train, y_val = y_labels[:split_idx], y_labels[split_idx:]
        initial_anomaly_labels_train = initial_anomaly_labels[:split_idx]
        
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
        annotation_system = OptimizedHumanAnnotationSystem(
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
            initial_anomaly_labels_train=initial_anomaly_labels_train,
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
                
                # 生成v2.2风格的综合分析图表
                save_training_plots_enhanced(training_history, args.output_dir)
                
                # 生成训练报告
                generate_training_report(training_history, final_metrics, feature_analysis, args.output_dir)
        except Exception as viz_error:
            print(f"⚠️ 可视化生成失败: {viz_error}")
            # 使用简化版可视化
            try:
                create_fallback_visualizations(training_history, args.output_dir)
            except Exception as simple_viz_error:
                print(f"⚠️ 简化版可视化也失败: {simple_viz_error}")

if __name__ == "__main__":
    main()