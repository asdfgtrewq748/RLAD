"""
修复版RLAD: 基于强化学习与GUI交互式人工标注的时间序列异常检测
专用于液压支架工作阻力异常检测 - 集成核心指标可视化功能
"""
import os
import json
import random
import warnings
import argparse
import traceback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import deque, namedtuple
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                           roc_curve, auc, precision_recall_curve, roc_auc_score,
                           average_precision_score, precision_recall_fscore_support)
from sklearn.manifold import TSNE
# GUI相关导入
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

# 配置matplotlib中文字体
plt.style.use('seaborn-v0_8-ticks') # A style suitable for papers
plt.rcParams['figure.dpi'] = 150 # Increase DPI for higher quality
plt.rcParams['savefig.dpi'] = 300 # High resolution for saved figures
plt.rcParams['font.family'] = 'serif' # Use a serif font like Times New Roman
plt.rcParams['axes.labelsize'] = 12 # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 10 # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 10 # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 10 # Font size for legend
plt.rcParams['axes.titlesize'] = 14 # Font size for title

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
            return str(obj)
        except Exception:
            return repr(obj)


# =================================
# 核心指标可视化类 (增强版)
# =================================

class CoreMetricsVisualizer:
    def __init__(self, output_dir="./output_visuals"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Define a professional color palette
        self.colors = {
            'primary': '#0072B2',    # Blue
            'secondary': '#D55E00',  # Vermillion
            'tertiary': '#009E73',   # Green
            'accent': '#CC79A7',     # Pink
            'neutral': '#56B4E9',    # Sky Blue
            'black': '#333333'
        }

    def _set_scientific_style(self, ax, title, xlabel, ylabel):
        """Set a common, clean style for axes, suitable for scientific papers."""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=10)

    def plot_f1_score_training(self, training_history, save_path=None):
        """Plot F1-Score, Precision, and Recall during training."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        episodes, val_f1 = training_history.get('episodes', []), training_history.get('val_f1', [])
        val_precision, val_recall = training_history.get('val_precision', []), training_history.get('val_recall', [])
        if not episodes or not val_f1: return
        ax.plot(episodes, val_f1, color=self.colors['black'], linestyle='-', linewidth=2, label='F1-Score')
        ax.plot(episodes, val_precision, color=self.colors['primary'], linestyle='--', linewidth=1.5, label='Precision')
        ax.plot(episodes, val_recall, color=self.colors['secondary'], linestyle=':', linewidth=1.5, label='Recall')
        self._set_scientific_style(ax, 'Validation Metrics During Training', 'Epoch', 'Score')
        ax.set_ylim(0, 1.05); ax.legend(loc='best', frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'training_metrics.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Training metrics plot saved to: {save_path}")

    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        """Plot the ROC curve."""
        if len(np.unique(y_true)) < 2: return None
        fpr, tpr, _ = roc_curve(y_true, y_scores); roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=self.colors['black'], lw=1.5, linestyle='--', label='Random Classifier')
        self._set_scientific_style(ax, 'Receiver Operating Characteristic (ROC)', 'False Positive Rate', 'True Positive Rate')
        ax.set_xlim([-0.05, 1.0]); ax.set_ylim([0.0, 1.05]); ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'roc_curve.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"ROC curve plot saved to: {save_path}"); return roc_auc

    def plot_final_metrics_bar(self, precision, recall, f1_score, auc_roc, save_path=None):
        """Plot the four core metrics as a horizontal bar chart."""
        metrics, values = ['AUC-ROC', 'F1-Score', 'Recall', 'Precision'], [auc_roc, f1_score, recall, precision]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(metrics, values, color=self.colors['primary'], height=0.6)
        self._set_scientific_style(ax, 'Final Model Performance', 'Score', 'Metric')
        ax.set_xlim(0, 1.0); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.grid(False) # Turn off grid for bar chart
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'final_metrics_summary.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Final metrics summary plot saved to: {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot a styled confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={"size": 14}, linecolor='white', linewidths=1)
        self._set_scientific_style(ax, 'Confusion Matrix', 'Predicted Label', 'True Label')
        ax.set_xticklabels(['Normal', 'Anomaly']); ax.set_yticklabels(['Normal', 'Anomaly'], va='center', rotation=90)
        ax.grid(False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'confusion_matrix.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Confusion matrix plot saved to: {save_path}")

    def plot_precision_recall_curve(self, y_true, y_scores, save_path=None):
        """Plot the Precision-Recall curve."""
        if len(np.unique(y_true)) < 2: return
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color=self.colors['tertiary'], lw=2, label=f'PR Curve (AP = {avg_precision:.3f})')
        self._set_scientific_style(ax, 'Precision-Recall Curve', 'Recall', 'Precision')
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.legend(loc="best", frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'precision_recall_curve.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Precision-Recall curve plot saved to: {save_path}")

    def plot_prediction_scores_distribution(self, y_true, y_scores, save_path=None):
        """Plot the distribution of prediction scores for each class."""
        fig, ax = plt.subplots(figsize=(8, 5))
        scores_normal = y_scores[y_true == 0]
        scores_anomaly = y_scores[y_true == 1]
        sns.kdeplot(scores_normal, ax=ax, color=self.colors['primary'], fill=True, label='Normal Scores')
        sns.kdeplot(scores_anomaly, ax=ax, color=self.colors['secondary'], fill=True, label='Anomaly Scores')
        self._set_scientific_style(ax, 'Prediction Score Distribution', 'Prediction Score (for Anomaly)', 'Density')
        ax.legend(frameon=False); plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'score_distribution.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Prediction score distribution plot saved to: {save_path}")

    def plot_tsne_features(self, features, y_true, save_path=None):
        """Plot t-SNE visualization of learned features."""
        print("Performing t-SNE... (this may take a moment)")
        if len(features) < 2: return
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1), n_jobs=-1)
        features_2d = tsne.fit_transform(np.array(features))
        df_tsne = pd.DataFrame({'t-SNE-1': features_2d[:, 0], 't-SNE-2': features_2d[:, 1], 'label': y_true})
        df_tsne['label'] = df_tsne['label'].map({0: 'Normal', 1: 'Anomaly'})
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.scatterplot(
            data=df_tsne, x='t-SNE-1', y='t-SNE-2', hue='label',
            palette={'Normal': self.colors['primary'], 'Anomaly': self.colors['secondary']},
            style='label', ax=ax, s=50,
            hue_order=['Normal', 'Anomaly']
        )
        ax.legend(title='Class', frameon=False)
        self._set_scientific_style(ax, 't-SNE Visualization of Learned Features', 't-SNE Dimension 1', 't-SNE Dimension 2')
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'tsne_features.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"t-SNE plot saved to: {save_path}")

    def generate_all_core_visualizations(self, training_history=None, y_true=None, y_scores=None,
                                   y_pred=None, features=None, final_metrics=None,
                                   original_data=None, window_indices=None, window_size=None,
                                   agent=None, sample_data=None):
        """Generate all core visualizations with a scientific paper style."""
        print("\nGenerating Core Metric Visualizations...")
        if training_history:
            self.plot_f1_score_training(training_history)
            self.plot_training_dashboard(training_history)
        has_labels = y_true is not None and len(y_true) > 0 and len(np.unique(y_true)) > 1
        auc_roc_value = None
        if has_labels and y_scores is not None:
            auc_roc_value = self.plot_roc_curve(y_true, y_scores)
            self.plot_precision_recall_curve(y_true, y_scores)
            self.plot_prediction_scores_distribution(y_true, y_scores)
        if has_labels and y_pred is not None:
            self.plot_confusion_matrix(y_true, y_pred)
        if has_labels and features is not None:
            self.plot_tsne_features(features, y_true)
        if final_metrics:
            auc_val = auc_roc_value if auc_roc_value is not None else final_metrics.get('auc_roc', 0)
            self.plot_final_metrics_bar(final_metrics.get('precision', 0), final_metrics.get('recall', 0),
                                        final_metrics.get('f1', 0), auc_val)
        if original_data is not None and window_indices is not None and window_size is not None:
            all_scores = final_metrics.get('all_probabilities')
            all_preds = final_metrics.get('all_predictions')
            if has_labels and all_scores is not None and all_preds is not None:
                self.plot_prediction_vs_actual(original_data, window_indices, y_true, all_preds, all_scores, window_size)
            scores_for_heatmap = all_scores if all_scores is not None else all_preds
            if scores_for_heatmap is not None:
                self.plot_anomaly_heatmap(original_data, scores_for_heatmap, window_indices, window_size)
            if all_scores is not None:
                self.plot_model_confidence_over_time(all_scores, window_indices, original_data, window_size)
        if agent is not None and sample_data is not None:
            self.plot_attention_weights(agent, sample_data, window_size)
        print("Core metric visualizations generated successfully!")

    def plot_prediction_vs_actual(self, original_data, window_indices, true_labels, predictions, scores, window_size, save_path=None):
        """Plot predicted scores against the original time series."""
        fig, ax = plt.subplots(figsize=(15, 6))
        timeline = np.arange(len(original_data))
        ax.plot(timeline, original_data, color=self.colors['black'], alpha=0.6, label='Original Signal', linewidth=1.0)
        
        # Plot anomaly scores as a scatter plot on the signal
        window_centers = [idx + window_size // 2 for idx in window_indices]
        scatter = ax.scatter(window_centers, scores, c=scores, cmap='coolwarm', s=15, label='Anomaly Score', zorder=3)
        
        # Highlight true anomaly regions
        for i, label in enumerate(true_labels):
            if i < len(window_indices) and label == 1:
                start_idx = window_indices[i]
                ax.axvspan(start_idx, start_idx + window_size, color=self.colors['secondary'], alpha=0.2, lw=0)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Anomaly Score', fontsize=10)
        self._set_scientific_style(ax, 'Predicted Anomaly Score vs. Actual Anomalies', 'Time Step', 'Value')
        ax.legend(loc='upper right', frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'prediction_vs_actual.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Prediction vs. actual plot saved to: {save_path}")

    def plot_anomaly_heatmap(self, original_data, predictions, window_indices, window_size, save_path=None):
        """Generate an anomaly detection heatmap."""
        heatmap_data = np.zeros(len(original_data)); count_map = np.zeros(len(original_data))
        for i, start_idx in enumerate(window_indices):
            score = predictions[i]
            for j in range(window_size):
                if start_idx + j < len(heatmap_data):
                    heatmap_data[start_idx + j] += score
                    count_map[start_idx + j] += 1
        mask = count_map > 0
        heatmap_data[mask] /= count_map[mask]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(original_data, color=self.colors['black'], alpha=0.7, linewidth=1.0)
        self._set_scientific_style(ax1, 'Original Time Series', '', 'Value')
        
        im = ax2.imshow(heatmap_data.reshape(1, -1), cmap='coolwarm', aspect='auto', interpolation='nearest', extent=[0, len(original_data), 0, 1])
        self._set_scientific_style(ax2, 'Anomaly Score Heatmap', 'Time Step', '')
        ax2.set_yticks([])
        
        cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.3)
        cbar.set_label('Anomaly Probability', fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'anomaly_heatmap.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Anomaly detection heatmap saved to: {save_path}")

    def plot_attention_weights(self, agent, sample_data, window_size, save_path=None):
        """Visualize the model's attention weights."""
        agent.eval()
        with torch.no_grad():
            sample_tensor = torch.FloatTensor(sample_data).unsqueeze(0)
            _, attn_weights = agent(sample_tensor, return_attention_weights=True)
        agent.train()
        attn_weights = attn_weights.squeeze(0).mean(axis=0).cpu().numpy() # Average over heads for a 1D view
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(attn_weights)), attn_weights, color=self.colors['primary'])
        self._set_scientific_style(ax, 'Average Attention Weights Across Sequence', 'Sequence Position (Time Step)', 'Attention Weight')
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'attention_weights.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Attention weights visualization saved to: {save_path}")

    def plot_model_confidence_over_time(self, probabilities, window_indices, original_data, window_size, save_path=None):
        """Plot model confidence over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        time_points = np.arange(len(original_data))
        ax1.plot(time_points, original_data, color=self.colors['black'], alpha=0.7, linewidth=1.0)
        self._set_scientific_style(ax1, 'Original Time Series', '', 'Value')
        
        window_centers = [idx + window_size//2 for idx in window_indices]
        ax2.plot(window_centers, probabilities, color=self.colors['primary'], linewidth=1.5)
        ax2.axhline(y=0.5, color=self.colors['secondary'], linestyle='--', alpha=0.8, label='Decision Threshold (0.5)')
        self._set_scientific_style(ax2, 'Model Confidence Over Time', 'Time Step', 'Anomaly Probability')
        ax2.set_ylim(0, 1)
        ax2.legend(frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'confidence_over_time.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Model confidence plot saved to: {save_path}")

    def plot_training_dashboard(self, training_history, save_path=None):
        """Create a comprehensive training dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 1. Training Loss
        ax1 = axes[0]
        ax1.plot(training_history['episodes'], training_history['train_loss'], color=self.colors['secondary'])
        self._set_scientific_style(ax1, 'Training Loss', 'Epoch', 'Loss')
        
        # 2. Validation Metrics
        ax2 = axes[1]
        ax2.plot(training_history['episodes'], training_history['val_f1'], color=self.colors['black'], label='F1-Score')
        ax2.plot(training_history['episodes'], training_history['val_precision'], color=self.colors['primary'], linestyle='--', label='Precision')
        ax2.plot(training_history['episodes'], training_history['val_recall'], color=self.colors['secondary'], linestyle=':', label='Recall')
        self._set_scientific_style(ax2, 'Validation Metrics', 'Epoch', 'Score')
        ax2.set_ylim(0, 1.05); ax2.legend(frameon=False)
        
        # 3. Epsilon and Learning Rate
        ax3 = axes[2]
        ax3.plot(training_history['episodes'], training_history['epsilon'], color=self.colors['primary'], label='Epsilon (Exploration Rate)')
        ax3.set_ylabel('Epsilon', color=self.colors['primary'])
        ax3.tick_params(axis='y', labelcolor=self.colors['primary'])
        ax3_2 = ax3.twinx()
        ax3_2.plot(training_history['episodes'], training_history['learning_rate'], color=self.colors['tertiary'], label='Learning Rate')
        ax3_2.set_ylabel('Learning Rate', color=self.colors['tertiary'])
        ax3_2.tick_params(axis='y', labelcolor=self.colors['tertiary'])
        self._set_scientific_style(ax3, 'Epsilon & Learning Rate', 'Epoch', '')
        
        # 4. Human Annotations
        ax4 = axes[3]
        ax4.plot(training_history['episodes'], training_history['human_annotations_count'], color=self.colors['tertiary'])
        self._set_scientific_style(ax4, 'Cumulative Human Annotations', 'Epoch', 'Count')
        
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'training_dashboard.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Training dashboard saved to: {save_path}")  
# =================================
# 修复的GUI标注界面
# =================================

class AnnotationGUI:
    def __init__(self, window_size=288):
        self.window_size = window_size
        self.result = None
        self.root = None
        self.current_window_data = None
        self.current_original_data = None
        self.window_idx = None
        self.auto_prediction = None
        # 添加键盘快捷键映射表和提示
        self.keyboard_shortcuts = {
            '0': '标记为正常',
            '1': '标记为异常',
            's': '跳过当前样本',
            'q': '退出标注',
            'n': '下一个样本',
            'p': '上一个样本',
            'r': '重置缩放',
            'z': '启用/禁用缩放模式'
        }
    def create_gui(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """创建GUI界面"""
        try:
            self.current_window_data = window_data
            self.current_original_data = original_data_segment
            self.window_idx = window_idx
            self.auto_prediction = auto_predicted_label
            
            self.root = tk.Tk()
            self.root.title(f"液压支架异常检测标注 - 窗口 #{window_idx}")
            self.root.geometry("1200x800")
            self.root.configure(bg='#f0f0f0')
            
            # 设置窗口关闭事件
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            # 绑定键盘快捷键
            self.root.bind('<Key-0>', lambda event: self.confirm_and_set_result(0, "正常"))
            self.root.bind('<Key-1>', lambda event: self.confirm_and_set_result(1, "异常"))
            self.root.bind('<Key-s>', lambda event: self.set_result(-1)) # s for skip
            self.root.bind('<Key-q>', lambda event: self.set_result(-2)) # q for quit
            
            self.create_widgets()
            return True
            
        except Exception as e:
            print(f"创建GUI界面失败: {e}")
            if self.root:
                try:
                    self.root.destroy()
                except:
                    pass
            return False
    
    def create_widgets(self):
        """创建界面组件"""
        try:
            # 主框架
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # 配置权重
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(1, weight=1)
            
            # 创建组件
            self.create_info_display(main_frame)
            self.create_charts(main_frame)
            self.create_buttons(main_frame)
            
        except Exception as e:
            print(f"创建界面组件失败: {e}")
            raise
    
    def create_info_display(self, parent):
        """创建信息显示区域"""
        try:
            info_frame = ttk.LabelFrame(parent, text="样本信息", padding="10")
            info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            info_text = f"窗口编号: #{self.window_idx}\n"
            info_text += f"窗口大小: {self.window_size}个数据点\n"
            
            if self.current_original_data is not None:
                max_val = np.max(self.current_original_data)
                min_val = np.min(self.current_original_data)
                mean_val = np.mean(self.current_original_data)
                info_text += f"原始数据范围: {min_val:.2f} ~ {max_val:.2f}\n"
                info_text += f"原始数据均值: {mean_val:.2f}\n"
            
            if self.auto_prediction is not None:
                pred_text = "异常" if self.auto_prediction == 1 else "正常"
                info_text += f"AI预测结果: {pred_text}"
            
            info_label = ttk.Label(info_frame, text=info_text, font=('Arial', 10))
            info_label.grid(row=0, column=0, sticky=(tk.W, tk.N))
                            
        except Exception as e:
            print(f"创建信息显示区域失败: {e}")
         # 添加快捷键提示
        shortcuts_frame = ttk.LabelFrame(parent, text="键盘快捷键", padding="10")
        shortcuts_frame.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.E), padx=(10, 0), pady=(0, 10))
        
        shortcuts_text = "\n".join([f"{key}: {desc}" for key, desc in self.keyboard_shortcuts.items()])
        shortcuts_label = ttk.Label(shortcuts_frame, text=shortcuts_text, font=('Arial', 9))
        shortcuts_label.pack(fill=tk.BOTH)
    def create_charts(self, parent):
        """创建数据可视化图表"""
        try:
            chart_frame = ttk.LabelFrame(parent, text="数据可视化", padding="10")
            chart_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
            
            # 创建matplotlib图表
            fig = Figure(figsize=(12, 6), dpi=80)
            
            # 标准化数据图
            ax1 = fig.add_subplot(211)
            x_indices = np.arange(len(self.current_window_data))
            ax1.plot(x_indices, self.current_window_data.flatten(), 'b-', linewidth=1.5, alpha=0.8)
            ax1.set_title('标准化数据', fontsize=12, fontweight='bold')
            ax1.set_xlabel('时间点')
            ax1.set_ylabel('标准化值')
            ax1.grid(True, alpha=0.3)
            
            # 原始数据图（如果有的话）
            if self.current_original_data is not None:
                ax2 = fig.add_subplot(212)
                ax2.plot(x_indices, self.current_original_data.flatten(), 'r-', linewidth=1.5, alpha=0.8)
                ax2.set_title('原始阻力数据', fontsize=12, fontweight='bold')
                ax2.set_xlabel('时间点')
                ax2.set_ylabel('阻力值')
                ax2.grid(True, alpha=0.3)
                
                # 添加异常阈值线（示例）
                threshold = np.percentile(self.current_original_data, 75) + 0.5 * (
                    np.percentile(self.current_original_data, 75) - np.percentile(self.current_original_data, 25))
                ax2.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='异常阈值')
                ax2.legend()
            
            fig.tight_layout()
            
            # 嵌入到tkinter
            canvas = FigureCanvasTkAgg(fig, chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"创建图表失败: {e}")
    
    def create_buttons(self, parent):
        """创建标注按钮 - 重点修复区域"""
        try:
            button_frame = ttk.Frame(parent)
            button_frame.grid(row=2, column=0, columnspan=2, pady=10)
            
            # 按钮样式配置
            button_style = {'width': 15, 'padding': 10}
            
            # 正常按钮
            normal_btn = ttk.Button(
                button_frame, 
                text="标记为正常 (0)",
                command=lambda: self.confirm_and_set_result(0, "正常"),
                **button_style
            )
            normal_btn.grid(row=0, column=0, padx=10)
            
            # 异常按钮
            anomaly_btn = ttk.Button(
                button_frame, 
                text="标记为异常 (1)",
                command=lambda: self.confirm_and_set_result(1, "异常"),
                **button_style
            )
            anomaly_btn.grid(row=0, column=1, padx=10)
            
            # 跳过按钮
            skip_btn = ttk.Button(
                button_frame, 
                text="跳过此样本 (s)",
                command=lambda: self.set_result(-1),
                **button_style
            )
            skip_btn.grid(row=0, column=2, padx=10)
            
            # 退出标注按钮
            exit_btn = ttk.Button(
                button_frame, 
                text="退出标注 (q)",
                command=lambda: self.set_result(-2),
                **button_style
            )
            exit_btn.grid(row=0, column=3, padx=10)
            
            # 确保按钮可见
            button_frame.columnconfigure(0, weight=1)
            button_frame.columnconfigure(1, weight=1)
            button_frame.columnconfigure(2, weight=1)
            button_frame.columnconfigure(3, weight=1)
            
        except Exception as e:
            print(f"创建按钮失败: {e}")
    
    def set_result(self, result):
        """设置标注结果并关闭窗口"""
        try:
            self.result = result
            if self.root:
                self.root.quit()
                self.root.destroy()
        except Exception as e:
            print(f"设置结果时出错: {e}")
    
    def on_close(self):
        """窗口关闭事件"""
        try:
            self.result = -2  # 用户关闭窗口，视为退出标注
            if self.root:
                self.root.quit()
                self.root.destroy()
        except:
            pass
    
    def get_annotation(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """获取用户标注（主要接口）"""
        try:
            if not self.create_gui(window_data, window_idx, original_data_segment, auto_predicted_label):
                print("GUI创建失败，切换到命令行模式")
                return self.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
            
            # 运行GUI
            self.root.mainloop()
            
            result = self.result if self.result is not None else -2
            
            # 清理
            self.result = None
            if self.root:
                try:
                    self.root.destroy()
                except:
                    pass
                self.root = None
            
            return result
            
        except Exception as e:
            print(f"GUI标注过程出错: {e}")
            return self.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
    
    def get_annotation_fallback(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """回退到命令行标注模式"""
        print(f"\n{'='*60}")
        print(f"请对窗口 #{window_idx} 进行标注")
        print(f"{'='*60}")
        
        if original_data_segment is not None:
            max_val = np.max(original_data_segment)
            min_val = np.min(original_data_segment)
            print(f"原始数据范围: {min_val:.2f} ~ {max_val:.2f}")
        
        if auto_predicted_label is not None:
            pred_text = "异常" if auto_predicted_label == 1 else "正常"
            print(f"AI预测结果: {pred_text}")
        
        while True:
            choice = input("\n请选择 [0:正常, 1:异常, -1:跳过, -2:退出]: ").strip()
            if choice in ['0', '1', '-1', '-2']:
                return int(choice)
            print("输入无效，请重新输入")
# =================================
# 人工标注系统
# =================================

class HumanAnnotationSystem:
    def __init__(self, output_dir: str, window_size: int = 288, use_gui: bool = True):
        self.output_dir = output_dir
        self.window_size = window_size
        self.use_gui = use_gui
        self.annotations_file = os.path.join(output_dir, "human_annotations.json")
        self.annotations = {}
        os.makedirs(output_dir, exist_ok=True)
        self.load_existing_annotations()
        if use_gui:
            self.gui_annotator = AnnotationGUI(window_size)
        else:
            self.gui_annotator = None
    
    def load_existing_annotations(self):
        """加载已有的人工标注"""
        try:
            if os.path.exists(self.annotations_file):
                with open(self.annotations_file, 'r', encoding='utf-8') as f:
                    # 确保加载的键是字符串，值是整数
                    loaded_annotations = json.load(f)
                    self.annotations = {str(k): int(v) for k, v in loaded_annotations.items()}
                print(f"已加载 {len(self.annotations)} 个历史标注")
            else:
                self.annotations = {}
        except Exception as e:
            print(f"加载历史标注失败: {e}")
            self.annotations = {}
    
    def save_annotations(self):
        """保存标注到文件"""
        try:
            # 转换为可序列化格式
            serializable_annotations = convert_to_serializable(self.annotations)
            with open(self.annotations_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_annotations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存标注失败: {e}")
    
    def get_human_annotation(self, window_data: np.ndarray, window_idx: int, 
                           original_data_segment: np.ndarray = None, 
                           auto_predicted_label: int = None) -> int:
        """获取人工标注"""
        # 使用字符串格式的键进行查询和存储
        window_key = str(window_idx)
        
        # 如果已经标注过，直接返回标注值
        if window_key in self.annotations:
            return self.annotations[window_key]
        
        # 获取新的标注
        if self.use_gui and self.gui_annotator:
            try:
                annotation = self.gui_annotator.get_annotation(
                    window_data, window_idx, original_data_segment, auto_predicted_label
                )
            except Exception as e:
                print(f"GUI标注失败，切换到命令行: {e}")
                annotation = self.get_annotation_cmdline(
                    window_data, window_idx, original_data_segment, auto_predicted_label
                )
        else:
            annotation = self.get_annotation_cmdline(
                window_data, window_idx, original_data_segment, auto_predicted_label
            )
        
        # 保存标注
        if annotation in [0, 1]:  # 只保存有效的标注（0或1）
            self.annotations[window_key] = annotation
            self.save_annotations()
        
        return annotation
    
    def get_annotation_cmdline(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        """命令行标注接口"""
        print(f"\n{'='*60}")
        print(f"请对窗口 #{window_idx} 进行标注")
        print(f"{'='*60}")
        
        if original_data_segment is not None:
            max_val = np.max(original_data_segment)
            min_val = np.min(original_data_segment)
            mean_val = np.mean(original_data_segment)
            print(f"原始数据统计: 最大值={max_val:.2f}, 最小值={min_val:.2f}, 均值={mean_val:.2f}")
        
        if auto_predicted_label is not None:
            pred_text = "异常" if auto_predicted_label == 1 else "正常"
            print(f"AI预测结果: {pred_text}")
        
        while True:
            choice = input("\n请选择 [0:正常, 1:异常, -1:跳过, -2:退出标注]: ").strip()
            if choice in ['0', '1', '-1', '-2']:
                return int(choice)
            print("输入无效，请重新输入")

# =================================
# 数据集和数据加载
# =================================

class HydraulicDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = {'data': self.X[idx], 'label': self.y[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_hydraulic_data_improved(data_path, window_size=288, stride=12, specific_feature_column=None):
    """改进的数据加载函数"""
    print(f"正在加载数据: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")
    
    actual_selected_column_name = None 

    if specific_feature_column:
        if specific_feature_column in df.columns:
            actual_selected_column_name = specific_feature_column
            print(f"使用指定列: {actual_selected_column_name}")
        else:
            print(f"指定列 '{specific_feature_column}' 不存在，将自动选择合适的列")

    if not actual_selected_column_name: 
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Date' in numeric_cols:
            numeric_cols.remove('Date')
        if numeric_cols:
            actual_selected_column_name = numeric_cols[0]
            print(f"自动选择第一个数值列: {actual_selected_column_name}")
        else:
            raise ValueError("没有找到合适的数值列用于异常检测")
    
    selected_cols = [actual_selected_column_name] 
    print(f"最终选择的列数: {len(selected_cols)}, 列名: {selected_cols}")

    if 'Date' not in df.columns:
        df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='5min')
        
    df_for_point_mapping = df[['Date'] + selected_cols].copy()
    
    data_values = df[selected_cols].fillna(method='ffill').fillna(method='bfill').fillna(0).values
    print(f"提取的数据形状: {data_values.shape}")
    
    # 基于选定列计算"来压判据"阈值
    all_points_for_thresholds = data_values.flatten()
    laiya_criterion_point = np.inf 

    if len(all_points_for_thresholds) > 0:
        Q1 = np.percentile(all_points_for_thresholds, 25)
        Q3 = np.percentile(all_points_for_thresholds, 75)
        IQR = Q3 - Q1
        laiya_criterion_point = Q3 + 0.5 * IQR
        print(f"计算得到的来压判据阈值: {laiya_criterion_point:.2f}")
    else:
        print("警告: 数据为空，无法计算阈值")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    print("创建滑动窗口...")
    windows_scaled_list = []
    windows_raw_data_list = []  # 保存原始数据用于人工标注
    windows_raw_max_list = [] 
    window_start_indices_all = []

    for i in range(0, len(data_scaled) - window_size + 1, stride):
        window_scaled = data_scaled[i:i+window_size]
        window_raw = data_values[i:i+window_size]
        window_raw_max = np.max(window_raw)
        
        windows_scaled_list.append(window_scaled)
        windows_raw_data_list.append(window_raw)
        windows_raw_max_list.append(window_raw_max)
        window_start_indices_all.append(i)
    
    if not windows_scaled_list:
        raise ValueError("无法创建滑动窗口，请检查数据长度和窗口参数")

    X = np.array(windows_scaled_list)
    windows_raw_data = np.array(windows_raw_data_list)  # 原始数据窗口
    windows_raw_max_np = np.array(windows_raw_max_list)
    window_start_indices_all_np = np.array(window_start_indices_all)
    
    N = len(X)
    
    # 基于"来压判据"的打标签逻辑
    y = np.zeros(N, dtype=int) 
    num_anomalies_found_by_laiya = 0 
    for i in range(N):
        if windows_raw_max_np[i] > laiya_criterion_point:
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
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler, 
            df_for_point_mapping, test_window_original_indices, actual_selected_column_name,
            windows_raw_train)  # 返回训练集的原始数据窗口

# =================================
# 网络结构
# =================================

class EnhancedRLADAgent(nn.Module):
    def __init__(self, input_dim, seq_len=288, hidden_size=128, num_layers=3):
        super(EnhancedRLADAgent, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.1, batch_first=True)
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.q_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 2)
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.constant_(param.data, 0)
        
    def forward(self, x, return_features=False, return_attention_weights=False):
        batch_size = x.size(0)
        
        # LSTM特征提取
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 自注意力
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        lstm_pooled = torch.mean(lstm_out, dim=1)
        attn_pooled = torch.mean(attn_out, dim=1)
        
        # 特征融合
        combined_features = torch.cat([lstm_pooled, attn_pooled], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # Q值预测
        q_values = self.q_network(fused_features)
        
        if return_features and return_attention_weights:
            return q_values, fused_features, attn_weights
        elif return_features:
            return q_values, fused_features
        elif return_attention_weights:
            return q_values, attn_weights
        else:
            return q_values
    
    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()

# =================================
# 经验回放和奖励系统
# =================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity=20000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = Experience(state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
        
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        experiences = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states = torch.stack([torch.FloatTensor(e.state) for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.stack([torch.FloatTensor(e.next_state) for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

def enhanced_compute_reward(action, true_label, confidence=1.0, is_human_labeled=False):
    """计算奖励，人工标注的样本给予更高的权重"""
    if true_label == -1: 
        return 0.0  # 未标注样本无奖励
    
    # 人工标注的样本权重更高
    weight_multiplier = 3.0 if is_human_labeled else 1.0
    
    base_reward = confidence * weight_multiplier
    TP_REWARD, TN_REWARD, FN_PENALTY, FP_PENALTY = 2.0, 1.0, -2.0, -1.5
    
    if action == true_label: 
        reward = base_reward * (TP_REWARD if true_label == 1 else TN_REWARD)
    else: 
        reward = base_reward * (FN_PENALTY if true_label == 1 else FP_PENALTY)
    
    return reward


def enhanced_warmup(replay_buffer, X_train, y_train, agent, device):
    """使用已标注数据预热经验回放池"""
    print("开始模型预热...")
    labeled_mask = (y_train != -1)
    X_labeled, y_labeled = X_train[labeled_mask], y_train[labeled_mask]
    
    if len(X_labeled) == 0:
        print("没有可用于预热的已标注数据。")
        return
        
    print(f"预热样本数量: {len(X_labeled)}")
    
    # 优先添加异常样本
    anomaly_indices = np.where(y_labeled == 1)[0]
    normal_indices = np.where(y_labeled == 0)[0]
    
    # 将所有异常样本和等量的正常样本加入回放池
    num_anomalies = len(anomaly_indices)
    num_normals_to_add = min(len(normal_indices), num_anomalies * 2) # 添加更多正常样本以平衡
    
    selected_indices = np.concatenate([
        anomaly_indices,
        np.random.choice(normal_indices, num_normals_to_add, replace=False)
    ])
    np.random.shuffle(selected_indices)
    
    print(f"预热将添加 {len(selected_indices)} 个样本 (异常: {num_anomalies}, 正常: {num_normals_to_add})")

    for idx in selected_indices:
        state = X_labeled[idx]
        true_label = y_labeled[idx]
        
        # 使用模型预测来模拟一个动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.get_action(state_tensor)
            
        reward = enhanced_compute_reward(action, true_label, is_human_labeled=False)
        
        # 简化处理，next_state设为自身，done为True
        replay_buffer.push(state, action, reward, state, True)
        
    print(f"模型预热完成，回放池大小: {len(replay_buffer)}")



# =================================
# 训练和评估函数
# =================================

def enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, 
                           gamma=0.99, batch_size=64, beta=0.4,
                           grad_clip=1.0, use_amp=True):
    """优化的训练步骤 - 添加混合精度训练"""
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
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # 使用自动混合精度
    if scaler:
        with torch.cuda.amp.autocast():
            current_q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_agent(next_states).max(1)[0]
                target_q_values = rewards + (gamma * next_q_values * ~dones)
            td_errors = target_q_values - current_q_values
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none', beta=1.0)).mean()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        # 原有的训练代码
        current_q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = target_agent(next_states).max(1)[0]
            target_q_values = rewards + (gamma * next_q_values * ~dones)
        td_errors = target_q_values - current_q_values
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none', beta=1.0)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=grad_clip)
        optimizer.step()
    
    priorities_np = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
    replay_buffer.update_priorities(indices, priorities_np)
    
    return loss.item(), np.mean(np.abs(td_errors.detach().cpu().numpy()))

def enhanced_evaluate_model(agent, data_loader, device):
    agent.eval()
    all_predictions = []
    all_labels = []
    all_probabilities_positive_class = []
    all_features = []
    
    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch['data'].to(device), batch['label'].to(device)
            q_values, features = agent(data, return_features=True)
            predictions = q_values.argmax(dim=1)
            probabilities = F.softmax(q_values, dim=1)
            prob_positive = probabilities[:, 1]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_features.extend(features.cpu().numpy())
            all_probabilities_positive_class.extend(prob_positive.cpu().numpy())
            
            labeled_mask = (labels != -1)
            if labeled_mask.any():
                all_labels.extend(labels[labeled_mask].cpu().numpy())
    
    agent.train()
    
    # Case: No labeled data in the batch
    if len(all_labels) == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc_roc': 0.0,
            'anomaly_f1': 0.0, 'normal_f1': 0.0, # Add default keys
            'anomaly_precision': 0.0, 'anomaly_recall': 0.0,
            'normal_precision': 0.0, 'normal_recall': 0.0,
            'predictions': [], 'labels': [], 'probabilities': [], 'features': [],
            'all_predictions': np.array(all_predictions)
        }
    
    full_labels_list = [item['label'].item() for item in data_loader.dataset]
    labeled_mask_for_metrics = np.array(full_labels_list) != -1
    
    # Create filtered data for metrics and visualization
    predictions_for_metrics = np.array(all_predictions)[labeled_mask_for_metrics]
    probabilities_for_metrics = np.array(all_probabilities_positive_class)[labeled_mask_for_metrics]
    features_for_metrics = np.array(all_features)[labeled_mask_for_metrics]
    labels_for_metrics = np.array(all_labels)

    # Calculate weighted metrics
    precision_w = precision_score(labels_for_metrics, predictions_for_metrics, zero_division=0, average='weighted')
    recall_w = recall_score(labels_for_metrics, predictions_for_metrics, zero_division=0, average='weighted')
    f1_w = f1_score(labels_for_metrics, predictions_for_metrics, zero_division=0, average='weighted')
    
    # Calculate per-class metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        labels_for_metrics, predictions_for_metrics, average=None, labels=[0, 1], zero_division=0
    )
    normal_precision, normal_recall, normal_f1 = p_class[0], r_class[0], f1_class[0]
    anomaly_precision, anomaly_recall, anomaly_f1 = p_class[1], r_class[1], f1_class[1]

    # Calculate AUC-ROC
    auc_roc_val = 0.0
    if len(np.unique(labels_for_metrics)) > 1:
        try:
            auc_roc_val = roc_auc_score(labels_for_metrics, probabilities_for_metrics)
        except ValueError:
            pass
    
    return {
        'precision': precision_w, 'recall': recall_w, 'f1': f1_w, 'auc_roc': auc_roc_val,
        # --- Per-class metrics ---
        'anomaly_precision': anomaly_precision, 'anomaly_recall': anomaly_recall, 'anomaly_f1': anomaly_f1,
        'normal_precision': normal_precision, 'normal_recall': normal_recall, 'normal_f1': normal_f1,
        # --- Data for visualization (consistent lengths) ---
        'predictions': predictions_for_metrics,
        'labels': labels_for_metrics,
        'probabilities': probabilities_for_metrics,
        'features': features_for_metrics,
        # --- Full data for point-wise marking ---
        'all_predictions': np.array(all_predictions),
        'all_probabilities': np.array(all_probabilities_positive_class) # 添加这一行
    }

# =================================
# 交互式训练函数
# =================================

def interactive_train_rlad_gui(agent, target_agent, optimizer, scheduler, replay_buffer, 
                              X_train, y_train, windows_raw_train, X_val, y_val, device, 
                              annotation_system, num_episodes=150, target_update_freq=15,
                              epsilon_start=0.95, epsilon_end=0.02, epsilon_decay_rate=0.995,
                              batch_size_rl=64, output_dir="./output", annotation_frequency=10):
    """交互式RLAD训练，包含GUI人工标注系统"""
    os.makedirs(output_dir, exist_ok=True)
    training_history = {
        'episodes': [], 'train_loss': [], 'avg_td_error': [], 'val_f1': [], 
        'val_precision': [], 'val_recall': [], 'epsilon': [], 
        'replay_buffer_size': [], 'learning_rate': [], 'anomaly_f1': [], 'normal_f1': [],
        'human_annotations_count': [], 'human_labeled_accuracy': []
    }
    
    best_val_f1 = 0.0
    best_anomaly_val_f1 = 0.0
    patience = 20
    patience_counter = 0
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
        # 人工标注阶段
        if episode > 0 and episode % annotation_frequency == 0 and len(unlabeled_indices) > 0:
            print(f"\n{'='*60}")
            print(f"第 {episode} 轮 - 开始人工标注阶段")
            print(f"{'='*60}")
            
            # 随机选择一些未标注样本进行标注
            num_to_annotate = min(3, len(unlabeled_indices))  # 每次最多标注3个
            selected_for_annotation = np.random.choice(unlabeled_indices, num_to_annotate, replace=False)
            
            newly_labeled_count = 0
            for idx in selected_for_annotation:
                # 获取模型预测作为参考
                with torch.no_grad():
                    sample_tensor = torch.FloatTensor(X_train[idx]).unsqueeze(0).to(device)
                    q_values = agent(sample_tensor)
                    auto_prediction = q_values.argmax().item()
                
                # 获取人工标注
                annotation = annotation_system.get_human_annotation(
                    X_train[idx], idx, windows_raw_train[idx], auto_prediction
                )
                
                if annotation == -2:  # 用户选择退出标注
                    print("用户退出标注，继续训练...")
                    break
                elif annotation != -1:  # 不是跳过
                    y_train[idx] = annotation
                    human_labeled_indices.add(idx)
                    newly_labeled_count += 1
                    
                    # 将新标注的样本加入经验回放
                    is_human_labeled = True
                    confidence = 1.0
                    reward = enhanced_compute_reward(annotation, annotation, confidence, is_human_labeled)
                    
                    # 创建虚拟的next_state（这里简化处理）
                    next_state = X_train[idx] if idx + 1 >= len(X_train) else X_train[idx + 1]
                    replay_buffer.push(X_train[idx], annotation, reward, next_state, False)
            
            # 更新未标注索引列表
            unlabeled_indices = np.array([idx for idx in unlabeled_indices 
                                        if idx not in human_labeled_indices])
            
            print(f"本轮新增人工标注: {newly_labeled_count} 个")
            print(f"累计人工标注: {len(human_labeled_indices)} 个")
            print(f"剩余未标注样本: {len(unlabeled_indices)} 个")
        
        # DQN训练步骤
        train_loss, avg_td_error = enhanced_train_dqn_step(
            agent, target_agent, replay_buffer, optimizer, device, batch_size=batch_size_rl
        )
        
        # 更新目标网络
        if episode % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # 验证
        if episode % 5 == 0:
            val_dataset = HydraulicDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            val_metrics = enhanced_evaluate_model(agent, val_loader, device)
            
            # 记录训练历史
            training_history['episodes'].append(episode)
            training_history['train_loss'].append(train_loss)
            training_history['avg_td_error'].append(avg_td_error)
            training_history['val_f1'].append(val_metrics['f1'])
            training_history['val_precision'].append(val_metrics['precision'])
            training_history['val_recall'].append(val_metrics['recall'])
            training_history['epsilon'].append(epsilon)
            training_history['replay_buffer_size'].append(len(replay_buffer))
            training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            training_history['anomaly_f1'].append(val_metrics['anomaly_f1'])
            training_history['normal_f1'].append(val_metrics['normal_f1'])
            training_history['human_annotations_count'].append(len(human_labeled_indices))
            
            # 计算人工标注样本的准确率
            if len(human_labeled_indices) > 0:
                human_labeled_accuracy = 1.0  # 简化计算，认为人工标注是准确的
            else:
                human_labeled_accuracy = 0.0
            training_history['human_labeled_accuracy'].append(human_labeled_accuracy)
            
            print(f"Episode {episode}: Loss={train_loss:.4f}, Val F1={val_metrics['f1']:.4f}, "
                  f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}, "
                  f"AUC-ROC={val_metrics['auc_roc']:.4f}, Human Labels={len(human_labeled_indices)}")
            
            # 早停检查
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                # 保存最佳模型
                torch.save(agent.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            if val_metrics['anomaly_f1'] > best_anomaly_val_f1:
                best_anomaly_val_f1 = val_metrics['anomaly_f1']
        
        # 学习率调度
        if scheduler and episode % 10 == 0:
            scheduler.step()
        
        # epsilon衰减
        epsilon = max(epsilon_end, epsilon * epsilon_decay_rate)
        
        # 早停
        if patience_counter >= patience:
            print(f"早停触发于episode {episode}")
            break
    
    print(f"\n训练完成！最佳验证集F1: {best_val_f1:.4f}, 最佳异常F1: {best_anomaly_val_f1:.4f}")
    print(f"总共进行了 {len(human_labeled_indices)} 个人工标注")
    
    return training_history
def _process_window_parallel(args: Tuple[int, int, int]) -> List[int]:
    """
    用于并行处理的顶层辅助函数。
    它接收窗口的起始索引、窗口大小和总数据长度，并返回该窗口覆盖的所有点索引。
    """
    start_idx, ws, data_length = args
    return [start_idx + i for i in range(ws) if start_idx + i < data_length]


def mark_anomalies_pointwise(df_original, test_window_indices, test_predictions, 
                           window_size, feature_column, output_path):
    """优化的逐点标记函数 - 使用并行处理"""
    print("开始将窗口预测映射到逐点标签...")
    
    from multiprocessing import Pool, cpu_count
    
    # 创建一个新列来存储逐点预测
    df_original['pointwise_prediction'] = 0 # 默认为正常
    
    # 找出被预测为异常的窗口的起始索引
    anomaly_window_indices = test_window_indices[test_predictions == 1]
    
    if len(anomaly_window_indices) == 0:
        print("测试集中未检测到异常窗口。")
    else:
        print(f"在测试集中检测到 {len(anomaly_window_indices)} 个异常窗口。")
        
        # 准备并行处理的参数
        data_length = len(df_original)
        pool_args = [(idx, window_size, data_length) for idx in anomaly_window_indices]
        
        # 使用多进程并行处理
        # 注意：在Windows上，并行代码必须在 if __name__ == "__main__": 块内执行
        try:
            with Pool(processes=min(cpu_count(), 8)) as pool:
                results = pool.map(_process_window_parallel, pool_args)
        except Exception as e:
            print(f"并行处理失败 ({e})，回退到串行处理...")
            results = [_process_window_parallel(arg) for arg in pool_args]

        # 合并结果
        point_indices_to_mark = set()
        for result in results:
            point_indices_to_mark.update(result)
            
        # 在DataFrame中标记这些点
        if point_indices_to_mark:
            df_original.loc[list(point_indices_to_mark), 'pointwise_prediction'] = 1
    
    # 保存带标记的CSV文件
    output_filename = os.path.join(output_path, f'predictions_{feature_column}.csv')
    df_original.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"带逐点预测标签的CSV文件已保存到: {output_filename}")





# =================================
# 主函数
# =================================

def parse_args():
    parser = argparse.ArgumentParser(description='RLAD v2.2 - 交互式异常检测')
    parser.add_argument('--data_path', type=str, required=False, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./output_rlad_v2.2', help='输出目录')
    parser.add_argument('--feature_column', type=str, default=None, help='指定特征列名')
    parser.add_argument('--window_size', type=int, default=288, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=12, help='滑动窗口步长')
    parser.add_argument('--num_episodes', type=int, default=150, help='训练轮数')
    parser.add_argument('--use_gui', action='store_true', help='使用GUI进行标注')
    parser.add_argument('--annotation_frequency', type=int, default=10, help='人工标注频率（每n轮进行一次）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()

def main():
    args = parse_args()
    # ---- 在此处直接指定文件路径 ----
    args.data_path = r"C:\Users\18104\Desktop\Python files\deeplearning\example\timeseries\examples\RLAD\clean_data.csv"
    # --- 核心修复：强制启用GUI模式 ---
    args.use_gui = True
    # 检查路径是否已设置
    if not args.data_path:
        print("错误: 未指定数据文件路径。请通过命令行参数 --data_path 或在代码中直接设置。")
        return
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ---- 交互式特征选择 ----
    actual_selected_column_name = args.feature_column
    if not actual_selected_column_name:
        try:
            print(f"正在读取数据文件 '{args.data_path}' 以选择特征列...")
            df_cols = pd.read_csv(args.data_path, nrows=0).columns.tolist()
            numeric_cols = pd.read_csv(args.data_path, nrows=1).select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols:
                print("错误：CSV文件中未找到任何数值列。")
                return
                
            print("\n请选择要用于异常检测的特征列:")
            for i, col in enumerate(numeric_cols):
                print(f"  [{i}] {col}")
            
            while True:
                try:
                    choice = int(input(f"请输入选项编号 [0-{len(numeric_cols)-1}]: "))
                    if 0 <= choice < len(numeric_cols):
                        actual_selected_column_name = numeric_cols[choice]
                        print(f"您已选择: '{actual_selected_column_name}'")
                        break
                    else:
                        print("无效选项，请重试。")
                except ValueError:
                    print("请输入数字。")
        except Exception as e:
            print(f"读取列时出错，将使用默认行为: {e}")
            # 保持 actual_selected_column_name 为 None，让加载函数自动选择
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 初始化可视化工具
    visualizer = CoreMetricsVisualizer(os.path.join(args.output_dir, 'visualizations'))
    try:
        # 加载数据
        (X_train, y_train, X_val, y_val, X_test, y_test, scaler, 
         df_for_point_mapping, test_window_original_indices, final_selected_col,
         windows_raw_train) = load_hydraulic_data_improved(
            args.data_path, args.window_size, args.stride, actual_selected_column_name
        )
        
        # 初始化人工标注系统
        annotation_system = HumanAnnotationSystem(
            output_dir=args.output_dir, 
            window_size=args.window_size, 
            use_gui=args.use_gui
        )
        
        # 初始化模型
        input_dim = X_train.shape[-1]
        agent = EnhancedRLADAgent(input_dim, args.window_size).to(device)
        target_agent = EnhancedRLADAgent(input_dim, args.window_size).to(device)
        target_agent.load_state_dict(agent.state_dict())
        
        optimizer = torch.optim.AdamW(agent.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
        replay_buffer = PrioritizedReplayBuffer(capacity=20000)
        
        # 模型预热
        enhanced_warmup(replay_buffer, X_train, y_train, agent, device)
        
        # 交互式训练
        training_history = interactive_train_rlad_gui(
            agent, target_agent, optimizer, scheduler, replay_buffer,
            X_train, y_train, windows_raw_train, X_val, y_val, device,
            annotation_system, args.num_episodes, annotation_frequency=args.annotation_frequency,
            output_dir=args.output_dir
        )
        
        # 加载最佳模型进行最终评估
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print("加载最佳模型进行最终评估...")
            agent.load_state_dict(torch.load(best_model_path))

        # 最终评估
        test_dataset = HydraulicDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        final_metrics = enhanced_evaluate_model(agent, test_loader, device)
        
        print(f"\n最终测试结果:")
        print(f"精确率: {final_metrics['precision']:.4f}")
        print(f"召回率: {final_metrics['recall']:.4f}")
        print(f"F1分数: {final_metrics['f1']:.4f}")
        print(f"AUC-ROC: {final_metrics['auc_roc']:.4f}")
        # 获取样本数据用于注意力可视化
        sample_idx = np.random.choice(len(X_test), 1)[0]
        sample_data = X_test[sample_idx]

        # 修改现有的可视化调用
        visualizer.generate_all_core_visualizations(
            training_history=training_history,
            y_true=final_metrics['labels'],
            y_scores=final_metrics['probabilities'], # 修复: 使用过滤后的 'probabilities'
            y_pred=final_metrics['predictions'],
            features=final_metrics['features'],
            final_metrics=final_metrics,
            original_data=df_for_point_mapping[final_selected_col].values,
            window_indices=test_window_original_indices,
            window_size=args.window_size,
            agent=agent,
            sample_data=sample_data
        )
        
        
        # 逐点标记并保存结果 (使用完整的预测列表)
        mark_anomalies_pointwise(
            df_for_point_mapping,
            test_window_original_indices,
            final_metrics['all_predictions'], # 使用 'all_predictions' 键
            args.window_size,
            final_selected_col,
            args.output_dir
        )
        
        # 保存结果
        results = {
            'training_history': convert_to_serializable(training_history),
            'final_metrics': convert_to_serializable(final_metrics),
            'model_config': {
                'input_dim': input_dim,
                'window_size': args.window_size,
                'stride': args.stride,
                'feature_column': final_selected_col
            }
        }
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {args.output_dir}")
        print("可视化图表已生成!")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()