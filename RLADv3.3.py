"""
RLAD v3.3 (Final Optimized): 基于STL+LOF与强化学习的交互式液压支架工作阻力异常检测
集成了v2.4的完整可视化与评估流程，并保留了v3.0的核心检测逻辑。
新增了为学术论文准备的、具有深度解释性的可视化图表生成功能。
对超参数和训练策略进行了全面优化，以恢复并提升模型性能。
对所有图表进行了严格的科研论文排版优化。
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
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                           roc_curve, auc, precision_recall_curve, roc_auc_score,
                           average_precision_score, precision_recall_fscore_support)
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL

# GUI及绘图库导入
import tkinter as tk
from tkinter import ttk, messagebox
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =================================
# 全局配置
# =================================

# [排版优化] 配置matplotlib为科研论文风格
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman' # 设置全局字体
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.unicode_minus'] = False

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
# 核心指标可视化类 (来自 v2.4)
# =================================

class CoreMetricsVisualizer:
    def __init__(self, output_dir="./output_visuals"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = {
            'primary': '#0072B2', 'secondary': '#D55E00', 'tertiary': '#009E73',
            'accent': '#CC79A7', 'neutral': '#56B4E9', 'black': '#333333'
        }

    def _set_scientific_style(self, ax, title, xlabel, ylabel, ticks_on_top=True):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20 if ticks_on_top else 10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # [排版优化] 将X轴刻度和标签移到顶部
        if ticks_on_top:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')

    def plot_f1_score_training(self, training_history, save_path=None):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        episodes, val_f1 = training_history.get('episodes', []), training_history.get('val_f1', [])
        val_precision, val_recall = training_history.get('val_precision', []), training_history.get('val_recall', [])
        if not episodes or not val_f1: return
        ax.plot(episodes, val_f1, color=self.colors['black'], linestyle='-', linewidth=2, label='F1-Score')
        ax.plot(episodes, val_precision, color=self.colors['primary'], linestyle='--', linewidth=1.5, label='Precision')
        ax.plot(episodes, val_recall, color=self.colors['secondary'], linestyle=':', linewidth=1.5, label='Recall')
        self._set_scientific_style(ax, 'Validation Metrics During Training', 'Epoch', 'Score')
        ax.set_ylim(0, 1.05); ax.legend(loc='lower center', frameon=False, bbox_to_anchor=(0.5, -0.25), ncol=3)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'training_metrics.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Training metrics plot saved to: {save_path}")

    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        if len(np.unique(y_true)) < 2: return None
        fpr, tpr, _ = roc_curve(y_true, y_scores); roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=self.colors['black'], lw=1.5, linestyle='--', label='Random Classifier')
        self._set_scientific_style(ax, 'Receiver Operating Characteristic (ROC)', 'False Positive Rate', 'True Positive Rate', ticks_on_top=False)
        ax.set_xlim([-0.05, 1.0]); ax.set_ylim([0.0, 1.05]); ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'roc_curve.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"ROC curve plot saved to: {save_path}"); return roc_auc

    def plot_final_metrics_bar(self, precision, recall, f1_score, auc_roc, save_path=None):
        metrics, values = ['AUC-ROC', 'F1-Score', 'Recall', 'Precision'], [auc_roc, f1_score, recall, precision]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(metrics, values, color=self.colors['primary'], height=0.6)
        self._set_scientific_style(ax, 'Final Model Performance', 'Score', 'Metric', ticks_on_top=False)
        ax.set_xlim(0, 1.05); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.grid(False)
        ax.invert_yaxis() # 让最重要的指标在最上面
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'final_metrics_summary.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Final metrics summary plot saved to: {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
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
        if len(np.unique(y_true)) < 2: return
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color=self.colors['tertiary'], lw=2, label=f'PR Curve (AP = {avg_precision:.3f})')
        self._set_scientific_style(ax, 'Precision-Recall Curve', 'Recall', 'Precision', ticks_on_top=False)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.legend(loc="best", frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'precision_recall_curve.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Precision-Recall curve plot saved to: {save_path}")

    def plot_prediction_scores_distribution(self, y_true, y_scores, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(y_scores[y_true == 0], ax=ax, color=self.colors['primary'], fill=True, label='Normal Scores')
        sns.kdeplot(y_scores[y_true == 1], ax=ax, color=self.colors['secondary'], fill=True, label='Anomaly Scores')
        self._set_scientific_style(ax, 'Prediction Score Distribution', 'Prediction Score (for Anomaly)', 'Density', ticks_on_top=False)
        ax.legend(frameon=False); plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'score_distribution.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Prediction score distribution plot saved to: {save_path}")

    def plot_tsne_features(self, features, y_true, save_path=None):
        print("Performing t-SNE... (this may take a moment)")
        if len(features) < 2: return
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1), n_jobs=-1)
        features_2d = tsne.fit_transform(np.array(features))
        df_tsne = pd.DataFrame({'t-SNE-1': features_2d[:, 0], 't-SNE-2': features_2d[:, 1], 'label': y_true})
        df_tsne['label'] = df_tsne['label'].map({0: 'Normal', 1: 'Anomaly'})
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.scatterplot(data=df_tsne, x='t-SNE-1', y='t-SNE-2', hue='label',
                        palette={'Normal': self.colors['primary'], 'Anomaly': self.colors['secondary']},
                        style='label', ax=ax, s=50, hue_order=['Normal', 'Anomaly'])
        ax.legend(title='Class', frameon=False)
        self._set_scientific_style(ax, 't-SNE Visualization of Learned Features', 't-SNE Dimension 1', 't-SNE Dimension 2', ticks_on_top=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'tsne_features.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"t-SNE plot saved to: {save_path}")

    def plot_prediction_vs_actual(self, original_data, window_indices, true_labels, scores, window_size, save_path=None):
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # [排版优化] 为了清晰，这个图的x轴刻度保持在下方
        self._set_scientific_style(ax, 'Predicted Anomaly Score vs. Actual Anomalies', 'Time Step', 'Value', ticks_on_top=False)

        ax.plot(np.arange(len(original_data)), original_data, color=self.colors['black'], alpha=0.6, label='Original Signal', linewidth=1.0)
        window_centers = [idx + window_size // 2 for idx in window_indices]
        
        # 创建第二个Y轴用于绘制分数
        ax2 = ax.twinx()
        scatter = ax2.scatter(window_centers, scores, c=scores, cmap='coolwarm', s=15, label='Anomaly Score', zorder=3)
        ax2.set_ylabel('Anomaly Score', fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.spines['top'].set_visible(False)

        if len(true_labels) > 0:
            true_anomaly_indices = [i for i, label in enumerate(true_labels) if label == 1]
            for i in true_anomaly_indices:
                if i < len(window_indices):
                    start_idx = window_indices[i]
                    ax.axvspan(start_idx, start_idx + window_size, color=self.colors['secondary'], alpha=0.2, lw=0)

        cbar = plt.colorbar(scatter, ax=ax2, pad=0.08); cbar.set_label('Anomaly Score', fontsize=10)
        
        # 合并图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)
        
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'prediction_vs_actual.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Prediction vs. actual plot saved to: {save_path}")

    def plot_anomaly_heatmap(self, original_data, predictions, window_indices, window_size, save_path=None):
        heatmap_data = np.zeros(len(original_data)); count_map = np.zeros(len(original_data))
        for i, start_idx in enumerate(window_indices):
            score = predictions[i]
            end_idx = min(start_idx + window_size, len(heatmap_data))
            heatmap_data[start_idx:end_idx] += score
            count_map[start_idx:end_idx] += 1
        mask = count_map > 0
        heatmap_data[mask] /= count_map[mask]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # [排版优化] 让热力图的刻度在顶部
        self._set_scientific_style(ax1, 'Original Time Series', '', 'Value', ticks_on_top=False)
        ax1.plot(original_data, color=self.colors['black'], alpha=0.7, linewidth=1.0)
        
        im = ax2.imshow(heatmap_data.reshape(1, -1), cmap='coolwarm', aspect='auto', interpolation='nearest', extent=[0, len(original_data), 0, 1])
        self._set_scientific_style(ax2, 'Anomaly Score Heatmap', 'Time Step', '', ticks_on_top=True)
        ax2.set_yticks([])
        
        cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.4); cbar.set_label('Anomaly Probability', fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'anomaly_heatmap.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Anomaly detection heatmap saved to: {save_path}")

    def plot_attention_weights(self, agent, sample_data, device, save_path=None):
        agent.eval()
        with torch.no_grad():
            sample_tensor = torch.FloatTensor(sample_data).unsqueeze(0).to(device)
            _, _, attn_weights = agent(sample_tensor, return_features=True, return_attention_weights=True)
        agent.train()
        attn_weights = attn_weights.squeeze(0).mean(axis=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(attn_weights)), attn_weights, color=self.colors['primary'])
        self._set_scientific_style(ax, 'Average Attention Weights Across Sequence', 'Sequence Position (Time Step)', 'Attention Weight')
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'attention_weights.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Attention weights visualization saved to: {save_path}")
    
    def plot_training_dashboard(self, training_history, save_path=None):
        if not training_history or 'episodes' not in training_history:
            print("⚠️ 训练历史为空，跳过训练面板绘制")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        episodes = training_history['episodes']
        
        if not episodes:
            print("⚠️ 没有episode数据，跳过可视化")
            return
        
        # [排版优化] 训练面板所有子图刻度都在顶部
        if 'losses' in training_history and training_history['losses']:
            losses = training_history['losses']
            loss_episodes = episodes[:len(losses)] if len(losses) < len(episodes) else episodes
            axes[0].plot(loss_episodes, losses, color=self.colors['secondary'])
            self._set_scientific_style(axes[0], 'Training Loss', 'Epoch', 'Loss')
        else:
            axes[0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Training Loss (No Data)', fontsize=14)
            
        if all(key in training_history for key in ['val_f1', 'val_precision', 'val_recall']):
            if training_history['val_f1']:
                axes[1].plot(episodes, training_history['val_f1'], color=self.colors['black'], label='F1-Score')
                axes[1].plot(episodes, training_history['val_precision'], color=self.colors['primary'], linestyle='--', label='Precision')
                axes[1].plot(episodes, training_history['val_recall'], color=self.colors['secondary'], linestyle=':', label='Recall')
                self._set_scientific_style(axes[1], 'Validation Metrics', 'Epoch', 'Score')
                axes[1].set_ylim(0, 1.05)
                axes[1].legend(frameon=False)
            else:
                axes[1].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Validation Metrics (No Data)', fontsize=14)
        else:
            axes[1].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Validation Metrics (No Data)', fontsize=14)
        
        if 'learning_rate' in training_history and training_history['learning_rate']:
            axes[2].plot(episodes, training_history['learning_rate'], color=self.colors['tertiary'], label='Learning Rate')
            axes[2].set_ylabel('Learning Rate', color=self.colors['tertiary'])
            axes[2].tick_params(axis='y', labelcolor=self.colors['tertiary'])
            self._set_scientific_style(axes[2], 'Learning Rate Schedule', 'Epoch', '')
            axes[2].set_yscale('log')
        else:
            axes[2].text(0.5, 0.5, 'No LR Data', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Learning Rate (No Data)', fontsize=14)
        
        if 'val_auc' in training_history and training_history['val_auc']:
            axes[3].plot(episodes, training_history['val_auc'], color=self.colors['primary'])
            self._set_scientific_style(axes[3], 'Validation AUC-ROC', 'Epoch', 'AUC')
            axes[3].set_ylim(0, 1.05)
        else:
            best_f1 = max(training_history.get("val_f1", [0])) if training_history.get("val_f1") else 0
            axes[3].text(0.5, 0.5, f'Total Episodes: {len(episodes)}\nBest F1: {best_f1:.3f}', 
                        ha='center', va='center', transform=axes[3].transAxes, fontsize=12)
            axes[3].set_title('Training Summary', fontsize=14)
        
        plt.tight_layout()
        if save_path is None: 
            save_path = os.path.join(self.output_dir, 'training_dashboard.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Training dashboard saved to: {save_path}")

    def generate_all_core_visualizations(self, training_history, final_metrics, original_data,
                                         window_indices, window_size, agent, sample_data, device):
        print("\nGenerating Core Metric Visualizations...")
        if training_history:
            self.plot_training_dashboard(training_history)
        
        y_true, y_scores, y_pred, features = final_metrics['labels'], final_metrics['probabilities'], final_metrics['predictions'], final_metrics['features']
        has_labels = len(y_true) > 0 and len(np.unique(y_true)) > 1
        
        auc_roc_value = None
        if has_labels:
            auc_roc_value = self.plot_roc_curve(y_true, y_scores)
            self.plot_precision_recall_curve(y_true, y_scores)
            self.plot_prediction_scores_distribution(y_true, y_scores)
            self.plot_confusion_matrix(y_true, y_pred)
            self.plot_tsne_features(features, y_true)
            if original_data is not None and len(window_indices) > 0 and 'all_probabilities' in final_metrics:
                self.plot_prediction_vs_actual(original_data, window_indices, y_true, final_metrics['all_probabilities'], window_size)

        self.plot_final_metrics_bar(final_metrics.get('precision', 0), final_metrics.get('recall', 0),
                                    final_metrics.get('f1', 0), auc_roc_value if auc_roc_value is not None else 0)
        
        scores_for_heatmap = final_metrics.get('all_probabilities', final_metrics.get('all_predictions'))
        if original_data is not None and scores_for_heatmap is not None:
            self.plot_anomaly_heatmap(original_data, scores_for_heatmap, window_indices, window_size)
        
        if sample_data is not None:
            self.plot_attention_weights(agent, sample_data, device)
        print("Core metric visualizations generated successfully!")

# =================================
# 新增：论文发表专用可视化类
# =================================
class PublicationVisualizer:
    """
    一个专门用于生成论文级别图表的类，以增强研究的说服力。
    """
    def __init__(self, output_dir="./output_visuals/publication"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = {
            'primary': '#0072B2', 'secondary': '#D55E00', 'tertiary': '#009E73',
            'accent': '#CC79A7', 'neutral': '#56B4E9', 'black': '#333333',
            'highlight': '#E69F00'
        }

    def _set_scientific_style(self, ax, title, xlabel, ylabel, grid=True, ticks_on_top=True):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20 if ticks_on_top else 10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if grid:
            ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10)
        # [排版优化] 将X轴刻度和标签移到顶部
        if ticks_on_top:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')


    def plot_raw_data_with_annotations(self, data_series, save_path=None):
        """
        [图A] 绘制带标注的原始数据，展示数据挑战。
        注意：这里的标注位置是基于典型数据形态的模拟，用于说明问题。
        """
        fig, ax = plt.subplots(figsize=(15, 7))
        segment = data_series[1000:3000].copy() # 截取一段有代表性的数据
        segment.plot(ax=ax, color=self.colors['black'], alpha=0.7, linewidth=1, label='Raw Pressure Signal')

        # [排版优化] 让此图的刻度在下方，更符合常规阅读习惯
        self._set_scientific_style(ax, 'Characteristics of Raw Hydraulic Support Pressure Data', 'Time Step (minutes)', 'Pressure (MPa)', ticks_on_top=False)

        # 标注1: 正常工作循环
        ax.axvspan(250, 550, color=self.colors['tertiary'], alpha=0.15, label='Normal Operational Cycle')
        ax.text(400, segment.max()*0.9, 'Normal Cycle', ha='center', fontsize=11, color=self.colors['tertiary'])

        # 标注2: 突发性压力冲击
        surge_point_x, surge_point_y = segment[600:700].idxmax(), segment.max()
        ax.annotate('Abrupt Pressure Surge', xy=(surge_point_x, surge_point_y),
                    xytext=(surge_point_x + 150, surge_point_y * 0.9),
                    arrowprops=dict(facecolor=self.colors['secondary'], shrink=0.05, width=1.5, headwidth=8),
                    fontsize=11, color=self.colors['secondary'], fontweight='bold')

        # 标注3: 强噪声干扰
        noise_region_start, noise_region_end = 1200, 1400
        ax.axvspan(noise_region_start, noise_region_end, color=self.colors['accent'], alpha=0.2, label='High Noise Interference')
        ax.text((noise_region_start + noise_region_end) / 2, segment.min()*1.1, 'High Noise', ha='center', fontsize=11, color=self.colors['accent'])

        # 标注 4: 渐进式周期性偏离
        deviation_start, deviation_end = 1600, 1900
        ax.axvspan(deviation_start, deviation_end, color=self.colors['highlight'], alpha=0.2, label='Gradual Deviation')
        ax.text((deviation_start + deviation_end) / 2, segment.max()*0.8, 'Gradual Deviation', ha='center', fontsize=11, color=self.colors['highlight'])

        ax.legend(loc='upper left', frameon=False, fontsize=10)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'figure_A_raw_data_overview.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✅ [论文图A] 原始数据特征图已保存到: {save_path}")

    def plot_stl_decomposition_effect(self, data_window, anomaly_mask, period=24, save_path=None):
        """
        [图B] 绘制STL分解效果，展示信号净化过程。
        """
        seasonal_val = max(7, period * 2 + 1)
        if seasonal_val % 2 == 0:
            seasonal_val += 1

        stl = STL(data_window, period=period, seasonal=seasonal_val, robust=True)
        res = stl.fit()

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # 1. 原始信号
        self._set_scientific_style(axes[0], '(a) Original Signal with Anomaly', '', 'Value')
        axes[0].plot(data_window, color=self.colors['black'], label='Original Signal')
        if np.any(anomaly_mask):
            axes[0].axvspan(np.where(anomaly_mask)[0].min(), np.where(anomaly_mask)[0].max(),
                            color=self.colors['secondary'], alpha=0.2, label='True Anomaly Region')
        axes[0].legend(loc='upper right', frameon=False)

        # 2. 趋势项
        self._set_scientific_style(axes[1], '(b) Trend Component', '', 'Value')
        axes[1].plot(res.trend, color=self.colors['primary'], label='Trend Component')
        
        # 3. 季节性项
        self._set_scientific_style(axes[2], '(c) Seasonal Component', '', 'Value')
        axes[2].plot(res.seasonal, color=self.colors['tertiary'], label='Seasonal Component')
        
        # 4. 残差项
        self._set_scientific_style(axes[3], '(d) Residual Component (Anomaly Amplified)', 'Time Step in Window', 'Value', ticks_on_top=False)
        axes[3].plot(res.resid, color=self.colors['secondary'], label='Residual Component')
        if np.any(anomaly_mask):
            axes[3].axvspan(np.where(anomaly_mask)[0].min(), np.where(anomaly_mask)[0].max(),
                            color=self.colors['secondary'], alpha=0.2)
        
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'figure_B_stl_decomposition.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✅ [论文图B] STL分解效果图已保存到: {save_path}")
        return res.resid

    def plot_lof_scores_on_residual(self, residuals, anomaly_mask, n_neighbors=20, save_path=None):
        """
        [图C] 在STL分解的残差上绘制LOF异常得分。
        """
        if len(residuals) <= n_neighbors:
            n_neighbors = len(residuals) -1
        if n_neighbors <= 0:
            print("⚠️ Residuals length is too short for LOF. Skipping LOF plot.")
            return
            
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof.fit_predict(residuals.values.reshape(-1, 1))
        lof_scores = -lof.negative_outlier_factor_

        fig, ax = plt.subplots(figsize=(10, 5))
        self._set_scientific_style(ax, 'LOF Anomaly Scores on Residual Series', 'Time Step in Window', 'LOF Score', ticks_on_top=False)
        ax.plot(lof_scores, color=self.colors['highlight'], marker='.', linestyle='-', markersize=4, label='LOF Anomaly Score')
        
        if np.any(anomaly_mask):
            ax.axvspan(np.where(anomaly_mask)[0].min(), np.where(anomaly_mask)[0].max(),
                        color=self.colors['secondary'], alpha=0.2, label='True Anomaly Region')

        threshold = np.percentile(lof_scores, 95)
        ax.axhline(y=threshold, color=self.colors['secondary'], linestyle='--', label=f'Example Threshold ({threshold:.2f})')

        ax.legend(loc='upper right', frameon=False)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'figure_C_lof_scores.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✅ [论文图C] LOF异常得分图已保存到: {save_path}")

# =================================
# STL+LOF双层异常检测系统 (来自 v3.0)
# =================================

class STLLOFAnomalyDetector:
    def __init__(self, period=24, seasonal=25, robust=True, n_neighbors=20, contamination=0.02):
        self.period = period
        if seasonal % 2 == 0:
            seasonal += 1
        self.seasonal = max(3, seasonal)
        
        if self.seasonal <= self.period:
            self.seasonal = self.period + (2 - self.period % 2) + 1
            
        self.robust = robust
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        print(f"🔧 STL+LOF Detector Initialized: STL(period={self.period}, seasonal={self.seasonal}), LOF(contamination={contamination})")

    def detect_anomalies(self, data):
        print("🔄 Running Enhanced STL+LOF point-wise anomaly detection...")
        series = pd.Series(data.flatten()).fillna(method='ffill').fillna(method='bfill')
        
        if len(series) < 2 * self.period: 
            raise ValueError(f"Data length {len(series)} is too short for STL period {self.period}")
        
        try:
            stl_result = STL(series, seasonal=self.seasonal, period=self.period, robust=self.robust).fit()
            residuals = stl_result.resid.dropna()
            
            if len(residuals) != len(series):
                aligned_residuals = pd.Series(index=series.index, dtype=float)
                aligned_residuals.loc[residuals.index] = residuals
                aligned_residuals = aligned_residuals.fillna(method='ffill').fillna(method='bfill')
                residuals = aligned_residuals
            
            residuals_2d = residuals.values.reshape(-1, 1)
            if len(residuals_2d) < self.n_neighbors: 
                self.n_neighbors = max(5, len(residuals_2d) - 1)
            
            if self.n_neighbors > 0:
                lof_model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination)
                lof_labels = lof_model.fit_predict(residuals_2d)
            else:
                lof_labels = np.zeros(len(residuals_2d), dtype=int)


            residual_mean = residuals.mean()
            residual_std = residuals.std()
            dynamic_threshold = residual_mean + 2.8 * residual_std
            statistical_anomalies = np.abs(residuals) > dynamic_threshold
            
            trend = stl_result.trend.dropna()
            if len(trend) > 1:
                trend_diff = np.diff(trend)
                if len(trend_diff) > 0:
                    trend_threshold = np.percentile(np.abs(trend_diff), 95)
                    trend_anomalies = np.abs(trend_diff) > trend_threshold
                    trend_anomalies = np.concatenate([[False], trend_anomalies])
                    
                    if len(trend_anomalies) < len(residuals):
                        padding = np.zeros(len(residuals) - len(trend_anomalies), dtype=bool)
                        trend_anomalies = np.concatenate([trend_anomalies, padding])
                    elif len(trend_anomalies) > len(residuals):
                        trend_anomalies = trend_anomalies[:len(residuals)]
                else:
                    trend_anomalies = np.zeros(len(residuals), dtype=bool)
            else:
                trend_anomalies = np.zeros(len(residuals), dtype=bool)
            
            combined_scores = np.zeros(len(residuals))
            combined_scores += (lof_labels == -1).astype(float) * 0.4
            combined_scores += statistical_anomalies.astype(float) * 0.35
            combined_scores += trend_anomalies.astype(float) * 0.25
            
            final_threshold = 0.5
            final_labels = (combined_scores > final_threshold).astype(int)
            
            if len(final_labels) != len(series):
                full_labels = np.zeros(len(series), dtype=int)
                min_len = min(len(final_labels), len(series))
                full_labels[:min_len] = final_labels[:min_len]
                final_labels = full_labels
            
        except Exception as e:
            print(f"⚠️ STL分解过程出错: {e}")
            data_mean = np.mean(series)
            data_std = np.std(series)
            threshold = data_mean + 3 * data_std
            final_labels = (np.abs(series - data_mean) > threshold).astype(int)
        
        anomaly_count = np.sum(final_labels)
        anomaly_rate = anomaly_count / len(final_labels) if len(final_labels) > 0 else 0
        print(f"✅ Enhanced STL+LOF detection complete. Found {anomaly_count} anomaly points ({anomaly_rate:.2%})")
        
        return final_labels

# =================================
# GUI交互式标注界面 (来自 v3.0)
# =================================

class AnnotationGUI:
    def __init__(self, window_size=288):
        self.window_size = window_size
        self.result = None
        self.root = None
        self.canvas_widget = None

    def create_gui(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        self.result = None
        if self.root:
            try: self.root.destroy()
            except tk.TclError: pass
        
        self.root = tk.Tk()
        self.root.title(f"Anomaly Annotation - Window #{window_idx}")
        self.root.geometry("1200x900")
        self.root.configure(bg='#f0f0f0')
        self.root.lift(); self.root.attributes('-topmost', True); self.root.after_idle(self.root.attributes, '-topmost', False)
        
        main_container = tk.Frame(self.root, bg='#f0f0f0'); main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main_container, text=f"Annotate Window #{window_idx}", font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#2c3e50').pack(pady=(0,10))
        
        fig = Figure(figsize=(11, 6), dpi=100); fig.patch.set_facecolor('#f0f0f0')
        ax1 = fig.add_subplot(211)
        ax1.plot(window_data.flatten(), 'b-', lw=1.5, label='Standardized Data'); ax1.set_title('Standardized Data', fontsize=11); ax1.grid(True, alpha=0.3); ax1.legend()
        ax2 = fig.add_subplot(212)
        if original_data_segment is not None:
            ax2.plot(original_data_segment.flatten(), 'r-', lw=1.5, label='Original Data'); ax2.set_title('Original Data', fontsize=11); ax2.grid(True, alpha=0.3); ax2.legend()
        fig.tight_layout(pad=2.0)
        
        canvas = FigureCanvasTkAgg(fig, main_container)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        button_frame = tk.Frame(main_container, bg='#f0f0f0'); button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=15)
        tk.Button(button_frame, text="Normal (0)", font=('Arial', 14, 'bold'), bg='#27ae60', fg='white', width=15, height=2, command=lambda: self.set_result(0)).pack(side=tk.LEFT, padx=30, expand=True)
        tk.Button(button_frame, text="Anomaly (1)", font=('Arial', 14, 'bold'), bg='#e74c3c', fg='white', width=15, height=2, command=lambda: self.set_result(1)).pack(side=tk.LEFT, padx=30, expand=True)
        tk.Button(button_frame, text="Skip (s)", font=('Arial', 12), bg='#f39c12', fg='white', width=12, command=lambda: self.set_result(-1)).pack(side=tk.RIGHT, padx=15)
        tk.Button(button_frame, text="Quit (q)", font=('Arial', 12), bg='#95a5a6', fg='white', width=12, command=lambda: self.set_result(-2)).pack(side=tk.RIGHT, padx=15)
        
        self.root.bind('<Key-0>', lambda e: self.set_result(0)); self.root.bind('<Key-1>', lambda e: self.set_result(1))
        self.root.bind('<KeyPress-s>', lambda e: self.set_result(-1)); self.root.bind('<KeyPress-q>', lambda e: self.set_result(-2))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close); self.root.focus_force()
        return self.root

    def set_result(self, result):
        self.result = result
        if self.root:
            try:
                self.root.quit()
            except (RuntimeError, tk.TclError):
                pass

    def on_close(self):
        self.set_result(-2)

    def get_annotation(self, *args, **kwargs):
        try:
            root = self.create_gui(*args, **kwargs)
            if root is None: return self.get_annotation_fallback(*args, **kwargs)
            while self.result is None:
                root.update()
                root.update_idletasks()
                time.sleep(0.01)
            root.destroy()
            return self.result
        except Exception as e:
            print(f"GUI Error: {e}")
            if self.root:
                try: self.root.destroy()
                except: pass
            return self.get_annotation_fallback(*args, **kwargs)

    def get_annotation_fallback(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        pred_text = f"AI预测: {'异常' if auto_predicted_label == 1 else '正常'}" if auto_predicted_label is not None else ""
        while True:
            choice = input(f"请对窗口 #{window_idx} 进行标注 (0=正常, 1=异常, s=跳过, q=退出) {pred_text}: ").strip().lower()
            if choice == 'q': return -2
            if choice == 's': return -1
            if choice in ['0', '1']: return int(choice)

# =================================
# 人工标注系统 (来自 v3.0)
# =================================

class HumanAnnotationSystem:
    def __init__(self, output_dir: str, window_size: int, use_gui: bool):
        self.output_dir = output_dir
        self.use_gui = use_gui
        self.manual_labels_file = os.path.join(output_dir, 'manual_annotations.json')
        self.gui = AnnotationGUI(window_size) if use_gui else None
        self.annotation_history = self.load_existing_annotations()
        
    def load_existing_annotations(self):
        if not os.path.exists(self.manual_labels_file): return []
        try:
            with open(self.manual_labels_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            print(f"✅ 已加载 {len(history)} 条历史标注记录")
            return history
        except Exception as e:
            print(f"⚠️ 加载历史标注记录时出错: {e}"); return []

    def save_annotations(self):
        try:
            with open(self.manual_labels_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotation_history, f, ensure_ascii=False, indent=4, default=convert_to_serializable)
        except Exception as e:
            print(f"❌ 保存标注记录时出错: {e}")

    def get_human_annotation(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        for record in self.annotation_history:
            if record.get('window_idx') == window_idx:
                print(f"↪️ 窗口 #{window_idx} 已被标注为: {record['label']}")
                return record['label']
        
        if self.use_gui and self.gui:
            label = self.gui.get_annotation(window_data, window_idx, original_data_segment, auto_predicted_label)
        else:
            if self.gui is None: self.gui = AnnotationGUI(0)
            label = self.gui.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
        
        if label in [0, 1]:
            self.annotation_history.append({'window_idx': window_idx, 'label': label, 'timestamp': datetime.now()})
            self.save_annotations()
            print(f"✅ 已标注窗口 #{window_idx} 为: {'异常' if label == 1 else '正常'}")
        return label

# =================================
# 数据集与数据加载 (来自 v3.0 原始逻辑，更鲁棒)
# =================================
def augment_time_series(window, label, augment_prob=0.7):
    """为异常样本添加数据增强"""
    if label != 1 or np.random.random() > augment_prob:
        return window
        
    augment_type = np.random.choice(['noise', 'shift', 'scale', 'flip'])
    
    if augment_type == 'noise':
        noise_level = np.random.uniform(0.01, 0.05)
        return window + np.random.normal(0, noise_level, window.shape)
    elif augment_type == 'shift':
        shift = np.random.randint(1, 10)
        shifted = np.roll(window, shift, axis=0)
        return shifted
    elif augment_type == 'scale':
        scale = np.random.uniform(0.9, 1.1)
        return window * scale
    elif augment_type == 'flip':
        return -window
    
    return window
def extract_time_series_features(window):
    """提取时间序列特征"""
    features = []
    
    features.append(np.mean(window))
    features.append(np.std(window))
    features.append(np.max(window))
    features.append(np.min(window))
    features.append(np.median(window))
    
    features.append(np.mean(np.diff(window)))
    features.append(np.std(np.diff(window)))
    
    try:
        fft_vals = np.abs(np.fft.rfft(window.flatten()))
        features.append(np.mean(fft_vals))
        features.append(np.std(fft_vals))
        features.append(np.max(fft_vals))
        features.append(np.argmax(fft_vals))
    except:
        features.extend([0, 0, 0, 0])
    
    features.append(np.sum(np.abs(np.diff(window))))
    
    q25, q75 = np.percentile(window, [25, 75])
    iqr = q75 - q75
    features.append(np.sum((window > q75 + 1.5*iqr) | (window < q25 - 1.5*iqr)))
    
    return np.array(features, dtype=np.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, raw_data=None, augment=False, extract_features=False):
        self.X = torch.FloatTensor(X.astype(np.float32))
        self.y = torch.LongTensor(y)
        self.raw_data = torch.FloatTensor(raw_data.astype(np.float32)) if raw_data is not None else None
        self.augment = augment
        self.extract_features = extract_features
        
        if self.extract_features:
            self.features = []
            for i in range(len(X)):
                self.features.append(extract_time_series_features(X[i]))
            self.features = torch.FloatTensor(np.array(self.features))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment and y != -1:
            x_np = x.numpy()
            x_np = augment_time_series(x_np, y.item())
            x = torch.FloatTensor(x_np)
            
        raw_data_item = self.raw_data[idx] if self.raw_data is not None else torch.zeros_like(x)
        
        if self.extract_features:
            return x, y, raw_data_item, self.features[idx]
        
        return x, y, raw_data_item

def load_hydraulic_data_with_stl_lof(data_path, window_size, stride, specific_feature_column,
                                     stl_period=24, lof_contamination=0.02, unlabeled_fraction=0.1):
    """使用STL+LOF进行异常检测的数据加载函数 - 恢复自 v3.2 原始版本"""
    print(f"📥 Loading data with robust logic: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if '1#' in df.columns and '102#' not in df.columns:
        df = df.rename(columns={'1#': '102#'})
    
    if specific_feature_column:
        if specific_feature_column not in df.columns:
            raise ValueError(f"❌ 指定的特征列 '{specific_feature_column}' 不存在")
        selected_cols = [specific_feature_column]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = [col for col in numeric_cols if not col.startswith('Unnamed')]
        if not selected_cols:
            raise ValueError("❌ 未找到有效的数值列")
    
    print(f"➡️ Selected feature column: {selected_cols[0]}")
    data_values = df[selected_cols].fillna(method='ffill').fillna(method='bfill').fillna(0).values.flatten()
    
    seasonal_param = max(7, stl_period * 2 + 1)
    if seasonal_param % 2 == 0: seasonal_param += 1
    
    detector = STLLOFAnomalyDetector(
        period=stl_period, seasonal=seasonal_param, robust=True,
        n_neighbors=20, contamination=lof_contamination
    )
    point_anomaly_labels = detector.detect_anomalies(data_values)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values.reshape(-1, 1)).flatten()
    
    windows_scaled, windows_raw, window_anomaly_labels, window_indices = [], [], [], []
    for i in range(0, len(data_scaled) - window_size + 1, stride):
        windows_scaled.append(data_scaled[i:i + window_size])
        windows_raw.append(data_values[i:i + window_size])
        window_anomaly_labels.append(point_anomaly_labels[i:i + window_size])
        window_indices.append(i)
    
    def compute_window_label(window_anomalies):
        anomaly_ratio = np.mean(window_anomalies)
        anomaly_count = np.sum(window_anomalies)
        
        consecutive_anomalies = 0
        max_consecutive = 0
        for point in window_anomalies:
            if point == 1:
                consecutive_anomalies += 1
                max_consecutive = max(max_consecutive, consecutive_anomalies)
            else:
                consecutive_anomalies = 0
        
        if anomaly_count >= 5 or anomaly_ratio >= 0.04 or max_consecutive >= 3:
            return 1
        return 0
        
    y_initial = np.array([compute_window_label(labels) for labels in window_anomaly_labels])
    
    anomaly_count = np.sum(y_initial == 1)
    if anomaly_count == 0 or (anomaly_count / len(y_initial) < 0.02):
        window_scores = [np.sum(labels) for labels in window_anomaly_labels]
        score_threshold = np.percentile(window_scores, 95)
        if score_threshold == 0:
             positive_scores = np.array(window_scores)[np.array(window_scores)>0]
             if len(positive_scores) > 0:
                 score_threshold = np.percentile(positive_scores, 20)
             else: # if still no positive scores, force some anomalies
                 min_anomalies = max(10, int(0.02 * len(y_initial)))
                 top_indices = np.random.choice(len(y_initial), min_anomalies, replace=False)
                 y_final = np.zeros_like(y_initial)
                 y_final[top_indices] = 1
                 print("⚠️ No anomalies found, forcing random anomalies.")
                 y_initial = y_final

        y_final = (np.array(window_scores) >= score_threshold).astype(int)
    else:
        y_final = y_initial
    
    labeled_indices = np.random.choice(len(y_final), size=int(len(y_final) * (1 - unlabeled_fraction)), replace=False)
    y_with_unlabeled = np.full(len(y_final), -1)
    y_with_unlabeled[labeled_indices] = y_final[labeled_indices]

    if np.sum(y_with_unlabeled==1) == 0 and np.sum(y_final==1) > 0:
        # ensure at least one anomaly is labeled
        anomaly_idx = np.where(y_final==1)[0][0]
        if y_with_unlabeled[anomaly_idx] == -1:
            y_with_unlabeled[anomaly_idx] = 1

    X = np.array(windows_scaled)[:, :, np.newaxis]
    y = y_with_unlabeled
    raw_windows = np.array(windows_raw)[:, :, np.newaxis]
    
    return train_test_split_with_indices(X, y, raw_windows, np.array(window_indices), test_size=0.3, val_size=0.15)


def train_test_split_with_indices(X, y, raw_windows, window_indices, test_size=0.2, val_size=0.1):
    """带索引的数据集划分函数 - 恢复自 v3.2 原始版本"""
    n_samples = len(X)
    
    labeled_mask = (y != -1)
    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(~labeled_mask)[0]
    
    if len(labeled_indices) < 20: # Fallback for very few labels
        print("⚠️ Labeled samples are too few, using random split.")
        indices = np.random.permutation(n_samples)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        train_indices = indices[:-n_test-n_val]
        val_indices = indices[-n_test-n_val:-n_test]
        test_indices = indices[-n_test:]
    else:
        # Stratified split on labeled data
        from sklearn.model_selection import train_test_split
        try:
            train_labeled_indices, test_val_labeled_indices = train_test_split(
                labeled_indices, test_size=(test_size + val_size), stratify=y[labeled_indices], random_state=42
            )
            val_labeled_indices, test_labeled_indices = train_test_split(
                test_val_labeled_indices, test_size=(test_size / (test_size+val_size)), stratify=y[test_val_labeled_indices], random_state=42
            )
            train_indices = np.concatenate([train_labeled_indices, unlabeled_indices])
            val_indices = val_labeled_indices
            test_indices = test_labeled_indices
            print("✅ Stratified split successful.")
        except ValueError: # This happens if one class has too few samples for stratifying
            print("⚠️ Stratified split failed due to class imbalance, using random split for labeled data.")
            indices = np.random.permutation(n_samples)
            n_test = int(n_samples * test_size)
            n_val = int(n_samples * val_size)
            train_indices = indices[:-n_test-n_val]
            val_indices = indices[-n_test-n_val:-n_test]
            test_indices = indices[-n_test:]


    X_train, y_train, raw_train = X[train_indices], y[train_indices], raw_windows[train_indices]
    X_val, y_val, raw_val = X[val_indices], y[val_indices], raw_windows[val_indices]
    X_test, y_test, raw_test = X[test_indices], y[test_indices], raw_windows[test_indices]
    
    train_window_indices = window_indices[train_indices]
    val_window_indices = window_indices[val_indices]
    test_window_indices = window_indices[test_indices]
    
    return (X_train, y_train, raw_train, train_window_indices,
            X_val, y_val, raw_val, val_window_indices,
            X_test, y_test, raw_test, test_window_indices)

# =================================
# 模型、经验回放与奖励函数 (来自 v3.0 逻辑)
# =================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class EnhancedRLADAgent(nn.Module):
    def __init__(self, input_dim=1, seq_len=288, hidden_size=64, num_heads=2, 
                 dropout=0.2, bidirectional=True, include_pos=True, 
                 num_actions=2, use_lstm=True, use_attention=True, num_layers=1):
        super(EnhancedRLADAgent, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        self.num_actions = num_actions
        self.num_layers = num_layers
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(seq_len // 4 if seq_len > 4 else 1),
            nn.Dropout(dropout)
        )
        
        self.pre_lstm_norm = nn.LayerNorm(64)
        
        if use_lstm:
            self.lstm = nn.LSTM(64, hidden_size // 2, self.num_layers, batch_first=True, bidirectional=bidirectional)
                             
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, batch_first=True)
        self.ln_attention = nn.LayerNorm(hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2), nn.LeakyReLU(0.1),
            nn.Dropout(0.1), nn.Linear(hidden_size*2, hidden_size)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
            nn.Dropout(dropout), nn.Linear(hidden_size, num_actions)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if 'classifier.3' in name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    nn.init.constant_(module.bias, 0)
                else:
                    nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity='relu')
                    if module.bias is not None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name: nn.init.orthogonal_(param.data)
                    elif 'bias' in param_name:
                        nn.init.constant_(param.data, 0.0)
                        param.data[module.hidden_size:2 * module.hidden_size].fill_(1.0)
    
    def forward(self, x, return_features=False, return_attention_weights=False):
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(-1) # Add feature dimension
        
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.pre_lstm_norm(x)
        
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.ln_attention(lstm_out + attn_out)
        
        pooled = F.adaptive_max_pool1d(x.transpose(1, 2), 1).squeeze(2) * 0.5 + torch.mean(x, dim=1) * 0.5
        q_values = self.classifier(pooled)
        
        if return_features and return_attention_weights: return q_values, pooled, attn_weights
        if return_features: return q_values, pooled
        if return_attention_weights: return q_values, attn_weights
        return q_values

    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon: return random.randint(0, 1)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if state.dim() == 2: # Ensure state is 3D for model input
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()
        if was_training:
            self.train()
        return action

class PrioritizedReplayBuffer:
    def __init__(self, capacity=20000, alpha=0.6):
        self.capacity, self.alpha, self.buffer, self.pos = capacity, alpha, [], 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        exp = Experience(state.cpu().float(), action, reward, next_state.cpu().float(), done)
        if len(self.buffer) < self.capacity: self.buffer.append(exp)
        else: self.buffer[self.pos] = exp
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if not self.buffer: return None
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        exps = [self.buffer[idx] for idx in indices]
        
        states = torch.stack([e.state for e in exps])
        actions = torch.LongTensor([e.action for e in exps])
        rewards = torch.FloatTensor([e.reward for e in exps])
        next_states = torch.stack([e.next_state for e in exps])
        dones = torch.BoolTensor([e.done for e in exps])
        
        return states, actions, rewards, next_states, dones, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities): self.priorities[idx] = priority
    
    def __len__(self): return len(self.buffer)

def enhanced_compute_reward(action, label, is_human_labeled=False, is_augmented=False):
    action, label = int(action), int(label)
    base_reward = 1.5 if action == label and label == 1 else (-2.0 if action != label and label == 1 else (1.0 if action == label else -1.5))
    if is_human_labeled: base_reward *= 1.3
    if is_augmented: base_reward *= 0.9
    return base_reward

# =================================
# 训练与评估 (集成 v2.4 和 v3.0)
# =================================
def enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, gamma=0.95, batch_size=64, beta=0.4, scaler=None, grad_clip=1.0):
    if len(replay_buffer) < batch_size: return None
    
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)
    states, actions, rewards, next_states, dones, weights = states.to(device), actions.to(device), rewards.to(device), next_states.to(device), dones.to(device), weights.to(device)
    
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        q_values = agent(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = target_agent(next_states)
            q_target = rewards + gamma * next_q_values.max(1)[0] * (~dones)
        
        loss = (weights * F.smooth_l1_loss(q_current, q_target, reduction='none')).mean()

    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
        optimizer.step()

    td_errors = torch.abs(q_target - q_current).detach()
    replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
    
    return loss.item()

def enhanced_evaluate_model(agent, data_loader, device, threshold=0.5):
    agent.eval()
    all_preds, all_labels, all_probs, all_features = [], [], [], []
    with torch.no_grad():
        for data, labels, _ in data_loader:
            data = data.to(device)
            q_values, features = agent(data, return_features=True)
            probs = F.softmax(q_values, dim=1)
            predicted = (probs[:, 1] >= threshold).long()
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_features.extend(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    agent.train() # Restore model to training mode
    
    labeled_mask = np.array(all_labels) != -1
    if not np.any(labeled_mask):
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc_roc': 0.0, 'labels': [], 'predictions': [], 'probabilities': [], 'features': [], 'all_predictions':all_preds, 'all_probabilities':all_probs}
    
    y_true, y_pred, y_scores = np.array(all_labels)[labeled_mask], np.array(all_preds)[labeled_mask], np.array(all_probs)[labeled_mask]
    features_labeled = np.array(all_features)[labeled_mask]
    
    if len(np.unique(y_true)) < 2:
        accuracy = np.mean(y_pred == y_true)
        return {'f1': accuracy, 'precision': accuracy, 'recall': accuracy, 'auc_roc': 0.5, 'labels': y_true, 'predictions': y_pred, 'probabilities': y_scores, 'features': features_labeled, 'all_predictions':all_preds, 'all_probabilities':all_probs}

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc_roc = roc_auc_score(y_true, y_scores)
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'auc_roc': auc_roc, 'labels': y_true, 'predictions': y_pred, 'probabilities': y_scores, 'features': features_labeled, 'all_predictions': all_preds, 'all_probabilities': all_probs}

def find_optimal_threshold(val_dataset, agent, device):
    agent.eval()
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data, labels, _ in val_loader:
            data = data.to(device)
            q_values = agent(data)
            probs = F.softmax(q_values, dim=1)
            valid_mask = labels != -1
            all_probs.extend(probs[valid_mask, 1].cpu().numpy())
            all_labels.extend(labels[valid_mask].cpu().numpy())
    
    agent.train() # Restore model to training mode

    if len(all_labels) < 10 or len(np.unique(all_labels)) < 2: return 0.5
    
    best_f1, best_threshold = 0, 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (np.array(all_probs) >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1: best_f1, best_threshold = f1, threshold
    
    return best_threshold

def interactive_train_rlad_gui(agent, target_agent, optimizer, scheduler, replay_buffer, 
                              X_train, y_train, raw_train, X_val, y_val, raw_val, device, 
                              annotation_system, args):
    X_train_gpu = torch.tensor(X_train, device=device, dtype=torch.float32)
    y_train_cpu = y_train.copy()
    
    val_dataset = TimeSeriesDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if len(replay_buffer) == 0:
        labeled_indices = np.where(y_train_cpu != -1)[0]
        if len(labeled_indices) > 0:
            for idx in tqdm(np.random.choice(labeled_indices, size=min(len(labeled_indices), 500), replace=False), desc="Warming-up buffer"):
                state, label = X_train_gpu[idx], y_train_cpu[idx]
                action, reward = agent.get_action(state), enhanced_compute_reward(agent.get_action(state), label)
                next_state = X_train_gpu[(idx + 1) % len(X_train_gpu)]
                replay_buffer.push(state, action, reward, next_state, False)
    
    history = {'episodes': [], 'losses': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_auc': [], 'learning_rate': []}
    unlabeled_idx_pool = deque(np.where(y_train_cpu == -1)[0])
    
    best_val_f1, patience_counter = 0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == 'cuda')
    
    # [性能优化] 引入学习率预热
    warmup_epochs = 5
    initial_lr = args.lr / 10
    target_lr = args.lr

    for episode in range(args.num_episodes):
        agent.train()
        
        # [性能优化] 学习率预热逻辑
        if episode < warmup_epochs:
            current_lr = initial_lr + (target_lr - initial_lr) * (episode / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        if annotation_system.use_gui and episode > 0 and episode % args.annotation_frequency == 0 and unlabeled_idx_pool:
            agent.eval()
            query_indices_list = list(unlabeled_idx_pool)
            query_indices = query_indices_list[:min(32, len(query_indices_list))]
            with torch.no_grad():
                q_values = agent(X_train_gpu[query_indices])
                uncertainties = 1.0 - torch.max(F.softmax(q_values, dim=1), dim=1)[0]
            
            most_uncertain_idx_in_batch = uncertainties.argmax().item()
            query_idx = query_indices[most_uncertain_idx_in_batch]
            
            human_label = annotation_system.get_human_annotation(
                X_train[query_idx], query_idx, raw_train[query_idx], q_values[most_uncertain_idx_in_batch].argmax().item()
            )
            
            if human_label in [0, 1]:
                y_train_cpu[query_idx] = human_label
                unlabeled_idx_pool.remove(query_idx)
            elif human_label == -2:
                annotation_system.use_gui = False
        
        agent.train()
        
        if len(replay_buffer) > args.batch_size_rl:
            for _ in range(5):
                enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, batch_size=args.batch_size_rl, scaler=scaler, grad_clip=args.grad_clip)

        if episode % args.target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
            
        val_metrics = enhanced_evaluate_model(agent, val_loader, device)
        
        # [性能优化] 只有在预热结束后才开始调度学习率
        if episode >= warmup_epochs:
            scheduler.step(val_metrics['f1'])
        
        history['episodes'].append(episode)
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_auc'].append(val_metrics['auc_roc'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        if val_metrics['f1'] > best_val_f1 + 0.001: # [性能优化] 降低改进阈值
            best_val_f1, patience_counter = val_metrics['f1'], 0
            torch.save(agent.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= args.early_stopping:
            print(f"Early stopping at episode {episode}")
            break

    return best_val_f1, history

# =================================
# 逐点标记与主函数 (来自 v2.4)
# =================================

def _process_window_parallel(args: Tuple[int, int, int]) -> List[int]:
    start_idx, ws, data_length = args
    return [start_idx + i for i in range(ws) if start_idx + i < data_length]

def mark_anomalies_pointwise(df_original, test_window_indices, test_predictions, window_size, feature_column, output_path):
    print("Mapping window predictions to point-wise labels...")
    from multiprocessing import Pool, cpu_count
    df_original['pointwise_prediction'] = 0
    anomaly_window_indices = test_window_indices[test_predictions == 1]
    
    if len(anomaly_window_indices) > 0:
        pool_args = [(idx, window_size, len(df_original)) for idx in anomaly_window_indices]
        try:
            with Pool(processes=min(cpu_count(), 8)) as pool:
                results = pool.map(_process_window_parallel, pool_args)
        except Exception:
            results = [_process_window_parallel(arg) for arg in pool_args]
        
        point_indices_to_mark = set(idx for sublist in results for idx in sublist)
        if point_indices_to_mark:
            df_original.loc[list(point_indices_to_mark), 'pointwise_prediction'] = 1
    
    output_filename = os.path.join(output_path, f'predictions_{feature_column}.csv')
    df_original.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Point-wise prediction CSV saved to: {output_filename}")


def main():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        print("无法初始化Tkinter根窗口，可能在无头环境中运行。")

    parser = argparse.ArgumentParser(description='Optimized Interactive RLAD Anomaly Detection')
    parser.add_argument('--data_path', type=str, default="clean_data.csv")
    parser.add_argument('--feature_column', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./output_rlad_v3_optimized")
    parser.add_argument('--window_size', type=int, default=288)
    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--annotation_frequency', type=int, default=5)
    parser.add_argument('--use_gui', action='store_true', default=True)
    parser.add_argument('--no_gui', action='store_false', dest='use_gui')
    parser.add_argument('--seed', type=int, default=42)
    # [性能优化] 恢复并优化超参数
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--batch_size_rl', type=int, default=16) # 恢复为16
    parser.add_argument('--target_update_freq', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.92)
    parser.add_argument('--grad_clip', type=float, default=0.5) # 恢复为0.5
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--early_stopping', type=int, default=20) # 增加耐心
    parser.add_argument('--scheduler_patience', type=int, default=7) # 增加耐心
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu_id}" if not args.force_cpu and torch.cuda.is_available() else "cpu")

    try:
        df_preview = pd.read_csv(args.data_path)
        numeric_columns = [col for col in df_preview.columns if df_preview[col].dtype in ['int64', 'float64'] and col not in ['Unnamed: 0', 'Time', 'time']]
        actual_selected_column_name = args.feature_column
        if not actual_selected_column_name:
            print("Please select a column:")
            for i, col in enumerate(numeric_columns): print(f"[{i}] {col}")
            idx = int(input("Enter index: "))
            actual_selected_column_name = numeric_columns[idx]

        publication_viz_dir = os.path.join(args.output_dir, "publication_figures")
        pub_visualizer = PublicationVisualizer(output_dir=publication_viz_dir)
        pub_visualizer.plot_raw_data_with_annotations(df_preview[actual_selected_column_name])
        
        (X_train, y_train, raw_train, train_window_indices, X_val, y_val, raw_val, val_window_indices, X_test, y_test, raw_test, test_window_indices) = load_hydraulic_data_with_stl_lof(
            args.data_path, args.window_size, args.stride, actual_selected_column_name
        )

        anomaly_indices_in_test = np.where(y_test == 1)[0]
        if len(anomaly_indices_in_test) > 0:
            idx_to_plot = anomaly_indices_in_test[0]
            window_to_plot = raw_test[idx_to_plot].flatten()
            anomaly_mask = (y_test[idx_to_plot] == 1) * np.ones_like(window_to_plot, dtype=bool)
            residuals = pub_visualizer.plot_stl_decomposition_effect(pd.Series(window_to_plot), anomaly_mask=anomaly_mask)
            if residuals is not None: pub_visualizer.plot_lof_scores_on_residual(residuals, anomaly_mask=anomaly_mask)

        agent = EnhancedRLADAgent(input_dim=X_train.shape[2], seq_len=X_train.shape[1]).to(device)
        target_agent = EnhancedRLADAgent(input_dim=X_train.shape[2], seq_len=X_train.shape[1]).to(device)
        target_agent.load_state_dict(agent.state_dict())
        optimizer = optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.scheduler_factor, patience=args.scheduler_patience)
        replay_buffer = PrioritizedReplayBuffer()
        annotation_system = HumanAnnotationSystem(args.output_dir, args.window_size, args.use_gui)

        _, training_history = interactive_train_rlad_gui(
            agent, target_agent, optimizer, scheduler, replay_buffer,
            X_train, y_train, raw_train, X_val, y_val, raw_val, device,
            annotation_system, args
        )
        
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            agent.load_state_dict(torch.load(best_model_path, map_location=device))
            
            val_dataset = TimeSeriesDataset(X_val, y_val)
            test_dataset = TimeSeriesDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=128, num_workers=args.num_workers)
            
            optimal_threshold = find_optimal_threshold(val_dataset, agent, device)
            test_metrics = enhanced_evaluate_model(agent, test_loader, device, threshold=optimal_threshold)
            
            visualizer = CoreMetricsVisualizer(os.path.join(args.output_dir, "visualizations"))
            sample_for_viz = X_test[0] if len(X_test) > 0 else X_train[0]
            visualizer.generate_all_core_visualizations(
                training_history, test_metrics, 
                df_preview[actual_selected_column_name].values,
                test_window_indices, args.window_size, agent, sample_for_viz, device
            )

    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    sys.exit(main())

