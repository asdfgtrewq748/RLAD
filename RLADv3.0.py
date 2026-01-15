"""
RLAD v3.0 (GUI增强版): 基于STL+LOF与强化学习的交互式液压支架工作阻力异常检测
"""

# 基础及深度学习库导入
import os
import sys
import json
import time
import random
import warnings
import argparse
import traceback
import threading
import queue
from pathlib import Path
from datetime import datetime
from collections import deque, namedtuple
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 数据处理与评估库导入
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL

# GUI及绘图库导入
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =================================
# 全局配置
# =================================

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100

# 忽略警告
warnings.filterwarnings("ignore")

# =================================
# 辅助函数
# =================================

def set_seed(seed=42):
    """设置随机种子保证可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def convert_to_serializable(obj):
    """将numpy/torch等对象转换为可JSON序列化的格式"""
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
    elif isinstance(obj, set):
        return list(obj)
    else:
        try:
            return str(obj) if not isinstance(obj, (int, float, bool, str, type(None))) else obj
        except Exception:
            return f"Unserializable object: {type(obj)}"

# =================================
# 1. STL+LOF双层异常检测系统
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
        
        print(f"🔧 STL+LOF检测器初始化:")
        print(f"   STL参数: period={period}, seasonal={self.seasonal}, robust={robust}")
        print(f"   LOF参数: n_neighbors={n_neighbors}, contamination={contamination}")
    
    def prepare_time_series_data(self, data):
        """准备时间序列数据用于STL分解"""
        series = pd.Series(data.flatten())
        if series.isnull().any():
            series = series.fillna(method='ffill').fillna(method='bfill')
        if len(series) < 2 * self.period:
            raise ValueError(f"数据长度 {len(series)} 太短，STL分解至少需要 {2 * self.period} 个点")
        return series
    
    def fit_stl_decomposition(self, series):
        """执行STL分解"""
        try:
            stl = STL(series, seasonal=self.seasonal, period=self.period, robust=self.robust)
            self.stl_result = stl.fit()
            return self.stl_result
        except Exception as e:
            print(f"❌ STL分解失败: {e}")
            raise
    
    def fit_lof_detection(self, residuals):
        """在STL残差上应用LOF异常检测"""
        residuals_clean = residuals.dropna()
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
            return lof_labels, residuals_clean.index
        except Exception as e:
            print(f"❌ LOF检测失败: {e}")
            raise
    
    def detect_anomalies(self, data):
        """完整的STL+LOF异常检测流程，返回逐点异常标签"""
        print("🔄 开始执行STL+LOF逐点异常检测...")
        series = self.prepare_time_series_data(data)
        
        # STL分解
        stl_result = self.fit_stl_decomposition(series)
        residuals = stl_result.resid
        
        # LOF检测
        lof_labels, valid_indices = self.fit_lof_detection(residuals)
        
        # 映射回原始数据格式 (1=异常, 0=正常)
        anomaly_binary = (lof_labels == -1).astype(int)
        full_labels = np.zeros(len(series), dtype=int)
        
        # 将检测到的异常标签放回原序列对应位置
        np.put(full_labels, valid_indices, anomaly_binary)
        
        print(f"✅ STL+LOF检测完成，发现 {np.sum(full_labels)} 个异常点 ({np.mean(full_labels):.2%})")
        return full_labels

# =================================
# 2. GUI交互式标注界面
# =================================

class AnnotationGUI:
    """基于Tkinter的可视化标注界面"""
    
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
            if self.root is not None:
                try: self.root.destroy()
                except: pass
                    
            self.current_window_data = window_data
            self.current_original_data = original_data_segment
            self.window_idx = window_idx
            self.auto_prediction = auto_predicted_label
            self.result = None
            
            self.root = tk.Tk()
            self.root.title(f"液压支架异常检测 - 窗口 #{window_idx} 标注")
            self.root.geometry("1200x900")
            self.root.configure(bg='#f0f0f0')
            
            self.root.update_idletasks()
            x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
            y = (self.root.winfo_screenheight() // 2) - (900 // 2)
            self.root.geometry(f"1200x900+{x}+{y}")
            self.root.minsize(1000, 700)
            
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            
            self.create_widgets()
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.root.focus_force()
            return self.root
            
        except Exception as e:
            print(f"❌ 创建GUI窗口时出错: {e}")
            traceback.print_exc()
            return None
    
    def create_widgets(self):
        """创建界面组件"""
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题
        title_frame = tk.Frame(main_container, bg='#f0f0f0')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(title_frame, text=f"窗口 #{self.window_idx} 异常检测标注", font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#2c3e50').pack()
        
        # 信息显示
        info_frame = tk.LabelFrame(main_container, text="窗口信息", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#34495e', padx=10, pady=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        self.create_info_display(info_frame)
        
        # 图表
        chart_frame = tk.LabelFrame(main_container, text="数据可视化", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#34495e', padx=5, pady=5)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.create_charts(chart_frame)
        
        # 按钮
        button_frame = tk.Frame(main_container, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.create_buttons(button_frame)
        
        self.root.update_idletasks()
    
    def create_info_display(self, parent):
        """创建信息显示区域"""
        info_container = tk.Frame(parent, bg='#f0f0f0')
        info_container.pack(fill=tk.X, expand=True)
        
        left_frame = tk.Frame(info_container, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        tk.Label(left_frame, text="基本信息:", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
        tk.Label(left_frame, text=f"窗口索引: {self.window_idx}", font=('Arial', 10), bg='#f0f0f0').pack(anchor=tk.W)
        tk.Label(left_frame, text=f"窗口大小: {self.window_size}", font=('Arial', 10), bg='#f0f0f0').pack(anchor=tk.W)
        
        if self.auto_prediction is not None:
            pred_text = "异常" if self.auto_prediction == 1 else "正常"
            color = "#e74c3c" if self.auto_prediction == 1 else "#27ae60"
            tk.Label(left_frame, text=f"AI预测: {pred_text}", font=('Arial', 11, 'bold'), bg='#f0f0f0', fg=color).pack(anchor=tk.W, pady=(5, 0))
        
        right_frame = tk.Frame(info_container, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        if self.current_original_data is not None:
            tk.Label(right_frame, text="统计信息:", font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
            data_flat = self.current_original_data.flatten()
            stats = [f"均值: {np.mean(data_flat):.2f}", f"最大值: {np.max(data_flat):.2f}",
                     f"最小值: {np.min(data_flat):.2f}", f"标准差: {np.std(data_flat):.2f}"]
            for stat in stats:
                tk.Label(right_frame, text=stat, font=('Arial', 10), bg='#f0f0f0').pack(anchor=tk.W)
    
    def create_charts(self, parent):
        """创建数据可视化图表"""
        fig = Figure(figsize=(11, 5), dpi=90)
        fig.patch.set_facecolor('#f0f0f0')
        
        time_steps = np.arange(self.window_size)
        
        # 标准化数据图
        ax1 = fig.add_subplot(211)
        ax1.plot(time_steps, self.current_window_data.flatten(), 'b-', lw=1.5, label='标准化数据')
        ax1.set_title(f'窗口 #{self.window_idx} - 标准化数据', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 原始数据图
        ax2 = fig.add_subplot(212)
        if self.current_original_data is not None:
            original_flat = self.current_original_data.flatten()
            ax2.plot(time_steps, original_flat, 'r-', lw=1.5, label='原始数据')
            ax2.set_title(f'窗口 #{self.window_idx} - 原始阻力数据', fontsize=11)
            mean, std = np.mean(original_flat), np.std(original_flat)
            ax2.axhline(y=mean + 2 * std, color='orange', ls='--', lw=1, label=f'阈值线 (μ+2σ)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '原始数据不可用', ha='center', va='center', transform=ax2.transAxes)
        
        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_buttons(self, parent):
        """创建标注按钮"""
        button_container = tk.Frame(parent, bg='#f0f0f0')
        button_container.pack(fill=tk.X, pady=15)
        
        tk.Label(button_container, text="请选择标注结果:", font=('Arial', 16, 'bold'), bg='#f0f0f0').pack(pady=(0, 20))
        
        main_buttons_frame = tk.Frame(button_container, bg='#f0f0f0')
        main_buttons_frame.pack(pady=10)
        
        tk.Button(main_buttons_frame, text="正常 (0)", font=('Arial', 14, 'bold'), bg='#27ae60', fg='white', width=15, height=2, command=lambda: self.set_result(0)).pack(side=tk.LEFT, padx=30)
        tk.Button(main_buttons_frame, text="异常 (1)", font=('Arial', 14, 'bold'), bg='#e74c3c', fg='white', width=15, height=2, command=lambda: self.set_result(1)).pack(side=tk.LEFT, padx=30)
        
        aux_buttons_frame = tk.Frame(button_container, bg='#f0f0f0')
        aux_buttons_frame.pack(pady=15)
        
        tk.Button(aux_buttons_frame, text="跳过", font=('Arial', 12), bg='#f39c12', fg='white', width=12, command=lambda: self.set_result(-1)).pack(side=tk.LEFT, padx=15)
        tk.Button(aux_buttons_frame, text="退出", font=('Arial', 12), bg='#95a5a6', fg='white', width=12, command=lambda: self.set_result(-2)).pack(side=tk.LEFT, padx=15)
        
        tk.Label(button_container, text="快捷键: 0=正常, 1=异常, S=跳过, Q/Esc=退出", font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d').pack(pady=10)
        
        self.root.bind('<Key-0>', lambda e: self.set_result(0))
        self.root.bind('<Key-1>', lambda e: self.set_result(1))
        self.root.bind('<KeyPress-s>', lambda e: self.set_result(-1))
        self.root.bind('<KeyPress-S>', lambda e: self.set_result(-1))
        self.root.bind('<KeyPress-q>', lambda e: self.set_result(-2))
        self.root.bind('<KeyPress-Q>', lambda e: self.set_result(-2))
        self.root.bind('<Escape>', lambda e: self.set_result(-2))
        self.root.focus_set()

    def set_result(self, result):
        """设置标注结果并关闭窗口"""
        self.result = result
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def on_close(self):
        """窗口关闭事件"""
        self.result = -2 # 默认退出
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def get_annotation(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """获取用户标注（主接口）"""
        try:
            root = self.create_gui(window_data, window_idx, original_data_segment, auto_predicted_label)
            if root is None:
                return self.get_annotation_fallback(auto_predicted_label)
            root.mainloop()
            return self.result if self.result is not None else -2
        except Exception as e:
            print(f"❌ GUI标注失败: {e}, 回退到命令行模式")
            return self.get_annotation_fallback(auto_predicted_label)
    
    def get_annotation_fallback(self, auto_predicted_label=None):
        """回退到命令行标注模式"""
        if auto_predicted_label is not None:
            print(f"🤖 AI预测: {'异常' if auto_predicted_label == 1 else '正常'}")
        while True:
            choice = input("请输入标注 (0=正常, 1=异常, s=跳过, q=退出): ").strip().lower()
            if choice == 'q': return -2
            if choice == 's': return -1
            if choice in ['0', '1']: return int(choice)
            print("❌ 无效输入，请重新输入。")

# =================================
# 3. 交互式人工标注系统
# =================================

class HumanAnnotationSystem:
    """人工标注系统，集成GUI界面"""
    
    def __init__(self, output_dir: str, window_size: int = 288, use_gui: bool = True):
        self.output_dir = output_dir
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
                print(f"✅ 已加载 {len(self.annotation_history)} 条历史标注记录")
            except Exception as e:
                print(f"⚠️ 加载历史标注记录时出错: {e}")
    
    def save_annotations(self):
        """保存标注历史到文件"""
        try:
            with open(self.manual_labels_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotation_history, f, ensure_ascii=False, indent=4, default=convert_to_serializable)
        except Exception as e:
            print(f"❌ 保存标注记录时出错: {e}")
    
    def get_human_annotation(self, window_data: np.ndarray, window_idx: int, 
                           original_data_segment: np.ndarray = None, 
                           auto_predicted_label: int = None) -> int:
        """获取用户对单个窗口的标注"""
        # 检查是否已标注
        for record in self.annotation_history:
            if record.get('window_idx') == window_idx:
                print(f"↪️ 窗口 #{window_idx} 已被标注为: {record['label']}")
                return record['label']
        
        # 获取标注
        if self.use_gui and self.gui:
            label = self.gui.get_annotation(window_data, window_idx, original_data_segment, auto_predicted_label)
        else:
            print(f"\n{'='*50}\n请对窗口 #{window_idx} 进行标注 (命令行模式)")
            label = self.gui.get_annotation_fallback(auto_predicted_label)
        
        # 保存有效标注
        if label in [0, 1]:
            annotation_record = {
                'window_idx': window_idx,
                'label': label,
                'timestamp': datetime.now(),
                'auto_predicted_label': auto_predicted_label,
            }
            self.annotation_history.append(annotation_record)
            self.save_annotations()
            print(f"✅ 已标注窗口 #{window_idx} 为: {'异常' if label == 1 else '正常'}")
        
        return label

# =================================
# 4. 数据集与数据加载
# =================================

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
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

def load_hydraulic_data_with_stl_lof(data_path, window_size=288, stride=12, specific_feature_column=None,
                                     stl_period=24, lof_contamination=0.02, unlabeled_fraction=0.1):
    """
    使用STL+LOF进行初始标注的数据加载函数，并为交互式学习准备数据
    """
    print(f"📥 正在加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 选择特征列
    if specific_feature_column and specific_feature_column in df.columns:
        selected_cols = [specific_feature_column]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols: raise ValueError("未找到合适的数值列")
        selected_cols = [numeric_cols[0]]
    print(f"➡️ 选择的特征列: {selected_cols[0]}")
    
    data_values = df[selected_cols].fillna(method='ffill').fillna(method='bfill').fillna(0).values
    
    # STL+LOF异常检测
    stl_lof_detector = STLLOFAnomalyDetector(period=stl_period, contamination=lof_contamination)
    try:
        point_anomaly_labels = stl_lof_detector.detect_anomalies(data_values)
    except Exception as e:
        print(f"❌ STL+LOF检测失败: {e}. 将使用3-sigma作为回退方案。")
        mean, std = np.mean(data_values), np.std(data_values)
        point_anomaly_labels = (data_values > mean + 3 * std).astype(int).flatten()

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # 创建滑动窗口
    print("🔄 创建滑动窗口...")
    windows_scaled, windows_raw, window_anomaly_labels = [], [], []
    for i in range(0, len(data_scaled) - window_size + 1, stride):
        windows_scaled.append(data_scaled[i:i + window_size])
        windows_raw.append(data_values[i:i + window_size])
        window_anomaly_labels.append(point_anomaly_labels[i:i + window_size])
    
    if not windows_scaled: raise ValueError("未能创建滑动窗口")
    
    X = np.array(windows_scaled)
    windows_raw_data = np.array(windows_raw)
    N = len(X)
    
    # 基于STL+LOF结果生成窗口标签 (异常点比例 > 10% 则标记为异常)
    y = np.array([1 if np.mean(labels) > 0.1 else 0 for labels in window_anomaly_labels])
    
    print(f"📊 初步标签分布 (基于STL+LOF): 正常={np.sum(y==0)}, 异常={np.sum(y==1)}")
    
    # 设置一部分为未标注状态(-1)，供人工标注
    unlabeled_mask = np.random.random(N) < unlabeled_fraction
    y[unlabeled_mask] = -1 
    print(f"📊 最终标签分布: 正常={np.sum(y==0)}, 异常={np.sum(y==1)}, 未标注={np.sum(y==-1)}")
    
    # 数据集划分
    indices = np.arange(N)
    np.random.shuffle(indices)
    train_size, val_size = int(0.7 * N), int(0.15 * N)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_train, y_train, raw_train = X[train_indices], y[train_indices], windows_raw_data[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"✅ 数据加载完成: 训练集{X_train.shape}, 验证集{X_val.shape}, 测试集{X_test.shape}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler, 
            raw_train, selected_cols[0])

# =================================
# 5. 模型、经验回放与奖励函数
# =================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class EnhancedRLADAgent(nn.Module):
    """增强版RLAD智能体网络架构 (BiLSTM + Attention)"""
    def __init__(self, input_dim, seq_len=288, hidden_size=128, num_layers=2):
        super(EnhancedRLADAgent, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=0.3, bidirectional=True)
        
        lstm_out_size = hidden_size * 2
        self.attention = nn.MultiheadAttention(embed_dim=lstm_out_size, num_heads=8, dropout=0.2, batch_first=True)
        self.ln_attention = nn.LayerNorm(lstm_out_size)
        
        self.fc_block = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.q_head = nn.Linear(hidden_size // 2, 2) # 输出Q值
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.ln_attention(lstm_out + attn_out)
        
        # 使用平均池化和最大池化结合
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        pooled = avg_pool + max_pool
        
        features = self.fc_block(pooled)
        q_values = self.q_head(features)
        return q_values

    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                if state.ndim == 2: state = state.unsqueeze(0)
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
            if was_training: self.train()
            return action

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
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
        if len(self.buffer) == 0: return None
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
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
    """计算奖励，人工标注的样本给予更高的权重"""
    if true_label == -1: return 0.0
    
    weight = 3.0 if is_human_labeled else 1.0
    TP_REWARD, TN_REWARD, FN_PENALTY, FP_PENALTY = 5.0, 1.0, -3.0, -0.5
    
    if action == true_label:
        reward = TP_REWARD if true_label == 1 else TN_REWARD
    else:
        reward = FN_PENALTY if true_label == 1 else FP_PENALTY
    return reward * confidence * weight    
# =================================
# 6. 交互式训练与评估
# =================================

def enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, 
                           gamma=0.99, batch_size=64, beta=0.4):
    """使用优先经验回放进行一步DQN训练"""
    if len(replay_buffer) < batch_size: 
        return None, 0.0
    
    sample_result = replay_buffer.sample(batch_size, beta)
    if sample_result is None: 
        return None, 0.0
    
    states, actions, rewards, next_states, dones, indices, weights = sample_result
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    weights = torch.FloatTensor(weights).to(device)
    
    current_q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_q_values = target_agent(next_states).max(1)[0]
        next_q_values[dones] = 0.0
        target_q_values = rewards + gamma * next_q_values
    
    td_errors = target_q_values - current_q_values
    loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none', beta=1.0)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()
    
    priorities_np = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
    replay_buffer.update_priorities(indices, priorities_np)
    
    return loss.item(), td_errors.mean().item()

def enhanced_evaluate_model(agent, data_loader, device):
    """评估模型性能"""
    agent.eval()
    all_predictions, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            q_values = agent(X_batch)
            predictions = q_values.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(y_batch.numpy())
    
    agent.train()
    
    if len(all_predictions) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "f1_per_class": [0.0, 0.0]}
    
    all_predictions_np, all_labels_np = np.array(all_predictions), np.array(all_labels)
    
    f1 = f1_score(all_labels_np, all_predictions_np, average='weighted', zero_division=0)
    precision = precision_score(all_labels_np, all_predictions_np, average='weighted', zero_division=0)
    recall = recall_score(all_labels_np, all_predictions_np, average='weighted', zero_division=0)
    f1_pc = f1_score(all_labels_np, all_predictions_np, average=None, zero_division=0)
    
    return {
        "f1": float(f1), 
        "precision": float(precision), 
        "recall": float(recall), 
        "f1_per_class": [float(x) for x in f1_pc] if isinstance(f1_pc, np.ndarray) else [float(f1_pc), 0.0]
    }

def interactive_train_rlad_gui(agent, target_agent, optimizer, scheduler, replay_buffer, 
                              X_train, y_train, windows_raw_train, X_val, y_val, device, 
                              annotation_system, num_episodes=150, target_update_freq=15,
                              epsilon_start=0.95, epsilon_end=0.02, epsilon_decay_rate=0.995,
                              batch_size_rl=64, output_dir="./output", annotation_frequency=10):
    """交互式RLAD训练，包含GUI人工标注系统"""
    os.makedirs(output_dir, exist_ok=True)
    history = {k: [] for k in ['ep_loss', 'ep_td_error', 'val_f1', 'val_precision', 'val_recall', 
                               'epsilon', 'lr', 'anomaly_f1', 'normal_f1', 'human_annotations']}
    
    best_val_f1, best_anomaly_f1 = 0.0, 0.0
    human_labeled_indices = set()
    
    unlabeled_indices = np.where(y_train == -1)[0]
    np.random.shuffle(unlabeled_indices)
    unlabeled_idx_pool = deque(unlabeled_indices)
    
    epsilon = epsilon_start
    beta = 0.4
    beta_increment = (1.0 - beta) / num_episodes

    val_dataset = TimeSeriesDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_rl * 2, shuffle=False)

    print("\n🚀 开始GUI交互式RLAD训练...")
    
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        agent.train()
        ep_losses, ep_td_errors = [], []
        
        # --- 交互式标注环节 ---
        if annotation_system.use_gui and episode > 0 and episode % annotation_frequency == 0 and unlabeled_idx_pool:
            print(f"\n--- 触发第 {episode} 轮人工标注 ---")
            annotation_idx = unlabeled_idx_pool.popleft()
            
            state_to_label = torch.FloatTensor(X_train[annotation_idx]).unsqueeze(0).to(device)
            pred_label = agent.get_action(state_to_label)
            
            human_label = annotation_system.get_human_annotation(
                window_data=X_train[annotation_idx],
                window_idx=annotation_idx,
                original_data_segment=windows_raw_train[annotation_idx],
                auto_predicted_label=pred_label
            )
            
            if human_label in [0, 1]:
                y_train[annotation_idx] = human_label
                human_labeled_indices.add(annotation_idx)
                print(f"💡 样本 {annotation_idx} 已由人工更新标签为 {human_label}")
            elif human_label == -2:
                print("🛑 用户请求退出训练。")
                break
            else: # 跳过
                unlabeled_idx_pool.append(annotation_idx) # 放回池子末尾

        # --- 强化学习训练环节 ---
        num_samples = len(X_train)
        shuffled_indices = np.random.permutation(num_samples)
        
        for i in range(num_samples):
            idx = shuffled_indices[i]
            state = torch.FloatTensor(X_train[idx]).to(device)
            true_label = y_train[idx]
            
            if true_label == -1: continue

            action = agent.get_action(state, epsilon)
            is_human = idx in human_labeled_indices
            reward = enhanced_compute_reward(action, true_label, is_human_labeled=is_human)
            
            next_idx = (idx + 1) % num_samples
            next_state = torch.FloatTensor(X_train[next_idx]).to(device)
            done = (i == num_samples - 1)
            
            replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), done)
            
            loss, td_error = enhanced_train_dqn_step(
                agent, target_agent, replay_buffer, optimizer, device, 
                batch_size=batch_size_rl, beta=beta
            )
            if loss is not None:
                ep_losses.append(loss)
                ep_td_errors.append(td_error)


        # --- 周期性任务 ---
        if episode % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())

        val_metrics = enhanced_evaluate_model(agent, val_loader, device)
        scheduler.step(val_metrics['f1'])
        
        history['ep_loss'].append(np.mean(ep_losses) if ep_losses else 0)
        history['ep_td_error'].append(np.mean(ep_td_errors) if ep_td_errors else 0)
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['normal_f1'].append(val_metrics['f1_per_class'][0])
        history['anomaly_f1'].append(val_metrics['f1_per_class'][1])
        history['epsilon'].append(epsilon)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['human_annotations'].append(len(human_labeled_indices))

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_anomaly_f1 = val_metrics['f1_per_class'][1]
            torch.save(agent.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"\n⭐ 新的最佳模型! Val F1: {best_val_f1:.4f}, Anomaly F1: {best_anomaly_f1:.4f}")

        epsilon = max(epsilon_end, epsilon * epsilon_decay_rate)
        beta = min(1.0, beta + beta_increment)

    print(f"\n✅ 训练完成! 最佳验证集F1: {best_val_f1:.4f}")
    return history

# =================================
# 7. 结果可视化与主函数
# =================================

def plot_training_results(history, output_dir):
    """绘制训练过程图"""
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('RLAD 训练过程监控', fontsize=16)

    axs[0, 0].plot(history['ep_loss'], label='训练损失', color='tab:blue')
    axs[0, 0].plot(history['ep_td_error'], label='平均TD误差', color='tab:orange', alpha=0.7)
    axs[0, 0].set_title('损失与TD误差')
    axs[0, 0].set_xlabel('轮次')
    axs[0, 0].set_ylabel('值')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(history['val_f1'], label='综合 F1', color='tab:green', lw=2)
    axs[0, 1].plot(history['normal_f1'], label='正常类 F1', color='tab:cyan', ls='--')
    axs[0, 1].plot(history['anomaly_f1'], label='异常类 F1', color='tab:red', ls='--')
    axs[0, 1].set_title('验证集 F1 分数')
    axs[0, 1].set_xlabel('轮次')
    axs[0, 1].set_ylabel('F1 分数')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    ax2 = axs[1, 0].twinx()
    axs[1, 0].plot(history['epsilon'], label='Epsilon (探索率)', color='tab:purple')
    ax2.plot(history['human_annotations'], label='人工标注数 (右轴)', color='tab:brown', ls=':')
    axs[1, 0].set_title('探索率与人工标注')
    axs[1, 0].set_xlabel('轮次')
    axs[1, 0].set_ylabel('Epsilon')
    ax2.set_ylabel('累计标注数')
    fig.legend(loc='upper right', bbox_to_anchor=(0.48, 0.48))
    axs[1, 0].grid(True)

    axs[1, 1].plot(history['lr'], label='学习率', color='tab:pink')
    axs[1, 1].set_title('学习率变化')
    axs[1, 1].set_xlabel('轮次')
    axs[1, 1].set_ylabel('学习率')
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'training_summary.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='GUI交互式RLAD液压支架异常检测')
    parser.add_argument('--data_path', type=str, default="C:/Users/18104/Desktop/Python files/deeplearning/example/timeseries/examples/RLAD/clean_data.csv", help='数据文件路径')
    parser.add_argument('--feature_column_name', type=str, default=None, help='要处理的特定特征列名')
    parser.add_argument('--window_size', type=int, default=288, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=12, help='滑动窗口步长')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--num_episodes', type=int, default=100, help='训练轮数')
    parser.add_argument('--annotation_frequency', type=int, default=5, help='人工标注频率（每多少个episode）')
    parser.add_argument('--use_gui', action='store_true', default=True, help='是否使用GUI界面进行标注')
    parser.add_argument('--no_gui', action='store_true', help='禁用GUI，使用命令行标注')
    parser.add_argument('--output_dir', type=str, default="./output_rlad_v3", help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default="auto", help='设备选择 (auto, cpu, cuda)')
    parser.add_argument('--batch_size_rl', type=int, default=64, help='强化学习训练批次大小')
    parser.add_argument('--target_update_freq', type=int, default=10, help='目标网络更新频率')
    args = parser.parse_args()

    if args.no_gui:
        args.use_gui = False

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"GUI模式: {'启用' if args.use_gui else '禁用（使用命令行）'}")

    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, raw_train, feature_name = \
            load_hydraulic_data_with_stl_lof(
                data_path=args.data_path,
                window_size=args.window_size,
                stride=args.stride,
                specific_feature_column=args.feature_column_name
            )

        input_dim = X_train.shape[2]
        agent = EnhancedRLADAgent(input_dim, args.window_size, args.hidden_size, args.num_layers).to(device)
        target_agent = EnhancedRLADAgent(input_dim, args.window_size, args.hidden_size, args.num_layers).to(device)
        target_agent.load_state_dict(agent.state_dict())
        target_agent.eval()

        optimizer = optim.AdamW(agent.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)
        replay_buffer = PrioritizedReplayBuffer(capacity=20000)
        annotation_system = HumanAnnotationSystem(output_dir=args.output_dir, window_size=args.window_size, use_gui=args.use_gui)

        history = interactive_train_rlad_gui(
            agent, target_agent, optimizer, scheduler, replay_buffer,
            X_train, y_train, raw_train, X_val, y_val, device,
            annotation_system,
            num_episodes=args.num_episodes,
            target_update_freq=args.target_update_freq,
            batch_size_rl=args.batch_size_rl,
            output_dir=args.output_dir,
            annotation_frequency=args.annotation_frequency
        )

        plot_training_results(history, args.output_dir)
        print(f"训练结果图已保存至 {os.path.join(args.output_dir, 'training_summary.png')}")

    except Exception as e:
        print(f"\n❌ 主程序发生严重错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()