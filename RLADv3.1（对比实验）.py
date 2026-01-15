"""
RLAD v3.2 (Comparison): 在v3.1基础上增加多模型对比实验框架
模型包括: STL+3σ, Autoencoder (AE), Isolation Forest, STL-LOF, and STL-LOF-RLAD (本模型)
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
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

# GUI及绘图库导入
import tkinter as tk
from tkinter import ttk, messagebox
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =================================
# 全局配置
# =================================

# 配置matplotlib为科研论文风格
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
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
# 核心指标可视化类 (增加对比图)
# =================================

class CoreMetricsVisualizer:
    def __init__(self, output_dir="./output_visuals"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = {
            'primary': '#0072B2', 'secondary': '#D55E00', 'tertiary': '#009E73',
            'accent': '#CC79A7', 'neutral': '#56B4E9', 'black': '#333333'
        }
        self.model_colors = sns.color_palette("viridis", 6)


    def _set_scientific_style(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
    def plot_multiple_roc_curves(self, all_results, y_true, save_path=None):
        """在同一张图上绘制多个模型的ROC曲线"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 筛选出有标签的数据
        labeled_mask = (y_true != -1)
        y_true_filtered = y_true[labeled_mask]
        
        # 如果没有足够的标签，则不绘图
        if len(np.unique(y_true_filtered)) < 2:
            print("⚠️ Not enough classes in y_true to plot ROC curves.")
            return

        for i, (model_name, results) in enumerate(all_results.items()):
            y_score = results.get('scores')
            if y_score is None: continue
            
            y_score_filtered = y_score[labeled_mask]
            fpr, tpr, _ = roc_curve(y_true_filtered, y_score_filtered)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=self.model_colors[i], lw=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], color=self.colors['black'], lw=1.5, linestyle='--', label='Random Classifier')
        self._set_scientific_style(ax, 'ROC Curve Comparison', 'False Positive Rate', 'True Positive Rate')
        ax.set_xlim([-0.05, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right", frameon=False, fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'comparison_roc_curves.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Comparison ROC curve plot saved to: {save_path}")
    def plot_multiple_pr_curves(self, all_results, y_true, save_path=None):
        """在同一张图上绘制多个模型的Precision-Recall曲线"""
        fig, ax = plt.subplots(figsize=(8, 8))

        labeled_mask = (y_true != -1)
        y_true_filtered = y_true[labeled_mask]

        if len(np.unique(y_true_filtered)) < 2:
            print("⚠️ Not enough classes in y_true to plot PR curves.")
            return

        for i, (model_name, results) in enumerate(all_results.items()):
            y_score = results.get('scores')
            if y_score is None: continue

            y_score_filtered = y_score[labeled_mask]
            precision, recall, _ = precision_recall_curve(y_true_filtered, y_score_filtered)
            ap = average_precision_score(y_true_filtered, y_score_filtered)
            ax.plot(recall, precision, color=self.model_colors[i], lw=2,
                    label=f'{model_name} (AP = {ap:.3f})')

        self._set_scientific_style(ax, 'Precision-Recall Curve Comparison', 'Recall', 'Precision')
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.legend(loc="best", frameon=False, fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'comparison_pr_curves.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Comparison PR curve plot saved to: {save_path}")

    def plot_model_comparison_bar(self, comparison_results, save_path=None):
        """绘制多个模型性能指标的对比柱状图"""
        df = pd.DataFrame(comparison_results).T  # Transpose to have models as rows
        df = df.reset_index().rename(columns={'index': 'Model'})
        df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', ax=ax, palette=self.model_colors)
        
        self._set_scientific_style(ax, 'Model Performance Comparison', 'Metrics', 'Score')
        ax.set_ylim(0, 1.05)
        ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        
        # Add score labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=8,
                        rotation=0)

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
        if save_path is None: save_path = os.path.join(self.output_dir, 'model_comparison.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Model comparison plot saved to: {save_path}")

    # ... (保留所有来自 v3.1 的绘图函数) ...
    def plot_f1_score_training(self, training_history, save_path=None):
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
        metrics, values = ['AUC-ROC', 'F1-Score', 'Recall', 'Precision'], [auc_roc, f1_score, recall, precision]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(metrics, values, color=self.colors['primary'], height=0.6)
        self._set_scientific_style(ax, 'Final Model Performance', 'Score', 'Metric')
        ax.set_xlim(0, 1.0); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.grid(False)
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
        self._set_scientific_style(ax, 'Precision-Recall Curve', 'Recall', 'Precision')
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.legend(loc="best", frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'precision_recall_curve.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Precision-Recall curve plot saved to: {save_path}")

    def plot_prediction_scores_distribution(self, y_true, y_scores, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(y_scores[y_true == 0], ax=ax, color=self.colors['primary'], fill=True, label='Normal Scores')
        sns.kdeplot(y_scores[y_true == 1], ax=ax, color=self.colors['secondary'], fill=True, label='Anomaly Scores')
        self._set_scientific_style(ax, 'Prediction Score Distribution', 'Prediction Score (for Anomaly)', 'Density')
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
        self._set_scientific_style(ax, 't-SNE Visualization of Learned Features', 't-SNE Dimension 1', 't-SNE Dimension 2')
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'tsne_features.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"t-SNE plot saved to: {save_path}")

    def plot_prediction_vs_actual(self, original_data, window_indices, true_labels, scores, window_size, save_path=None):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(np.arange(len(original_data)), original_data, color=self.colors['black'], alpha=0.6, label='Original Signal', linewidth=1.0)
        window_centers = [idx + window_size // 2 for idx in window_indices]
        scatter = ax.scatter(window_centers, scores, c=scores, cmap='coolwarm', s=15, label='Anomaly Score', zorder=3)
        true_anomaly_indices = [i for i, label in enumerate(true_labels) if label == 1]
        for i in true_anomaly_indices:
            if i < len(window_indices):
                start_idx = window_indices[i]
                ax.axvspan(start_idx, start_idx + window_size, color=self.colors['secondary'], alpha=0.2, lw=0)
        cbar = plt.colorbar(scatter, ax=ax); cbar.set_label('Anomaly Score', fontsize=10)
        self._set_scientific_style(ax, 'Predicted Anomaly Score vs. Actual Anomalies', 'Time Step', 'Value')
        ax.legend(loc='upper right', frameon=False)
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
        ax1.plot(original_data, color=self.colors['black'], alpha=0.7, linewidth=1.0)
        self._set_scientific_style(ax1, 'Original Time Series', '', 'Value')
        im = ax2.imshow(heatmap_data.reshape(1, -1), cmap='coolwarm', aspect='auto', interpolation='nearest', extent=[0, len(original_data), 0, 1])
        self._set_scientific_style(ax2, 'Anomaly Score Heatmap', 'Time Step', '')
        ax2.set_yticks([])
        cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.3); cbar.set_label('Anomaly Probability', fontsize=10)
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
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        episodes = training_history['episodes']
        # 1. Training Loss
        axes[0].plot(episodes, training_history['train_loss'], color=self.colors['secondary'])
        self._set_scientific_style(axes[0], 'Training Loss', 'Epoch', 'Loss')
        # 2. Validation Metrics
        axes[1].plot(episodes, training_history['val_f1'], color=self.colors['black'], label='F1-Score')
        axes[1].plot(episodes, training_history['val_precision'], color=self.colors['primary'], linestyle='--', label='Precision')
        axes[1].plot(episodes, training_history['val_recall'], color=self.colors['secondary'], linestyle=':', label='Recall')
        self._set_scientific_style(axes[1], 'Validation Metrics', 'Epoch', 'Score')
        axes[1].set_ylim(0, 1.05); axes[1].legend(frameon=False)
        # 3. Epsilon and Learning Rate
        ax3_2 = axes[2].twinx()
        axes[2].plot(episodes, training_history['epsilon'], color=self.colors['primary'], label='Epsilon')
        ax3_2.plot(episodes, training_history['learning_rate'], color=self.colors['tertiary'], label='Learning Rate')
        axes[2].set_ylabel('Epsilon', color=self.colors['primary']); axes[2].tick_params(axis='y', labelcolor=self.colors['primary'])
        ax3_2.set_ylabel('Learning Rate', color=self.colors['tertiary']); ax3_2.tick_params(axis='y', labelcolor=self.colors['tertiary'])
        self._set_scientific_style(axes[2], 'Epsilon & Learning Rate', 'Epoch', '')
        # 4. Human Annotations
        axes[3].plot(episodes, training_history['human_annotations_count'], color=self.colors['tertiary'])
        self._set_scientific_style(axes[3], 'Cumulative Human Annotations', 'Epoch', 'Count')
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'training_dashboard.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Training dashboard saved to: {save_path}")

    def generate_all_core_visualizations(self, training_history, final_metrics, original_data,
                                         window_indices, window_size, agent, sample_data, device, comparison_results=None):
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
            self.plot_prediction_vs_actual(original_data, window_indices, y_true, final_metrics['all_probabilities'], window_size)

        self.plot_final_metrics_bar(final_metrics.get('precision', 0), final_metrics.get('recall', 0),
                                    final_metrics.get('f1', 0), auc_roc_value if auc_roc_value is not None else 0)
        
        scores_for_heatmap = final_metrics.get('all_probabilities', final_metrics.get('all_predictions'))
        if scores_for_heatmap is not None:
            self.plot_anomaly_heatmap(original_data, scores_for_heatmap, window_indices, window_size)
        
        if agent is not None:
            self.plot_attention_weights(agent, sample_data, device)
        
        # 新增：生成对比图
        if comparison_results:
            self.plot_model_comparison_bar(comparison_results)

        print("Core metric visualizations generated successfully!")

# =================================
# 基线模型实现
# =================================

def evaluate_baseline(y_true, y_pred, y_score=None):
    """计算基线模型的评估指标"""
    labeled_mask = (y_true != -1)
    y_true, y_pred = y_true[labeled_mask], y_pred[labeled_mask]
    
    if len(y_true) == 0:
        return {'Precision': 0, 'Recall': 0, 'F1-Score': 0, 'AUC-ROC': 0}

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    auc_roc = 0.0
    if y_score is not None and len(np.unique(y_true)) > 1:
        y_score = y_score[labeled_mask]
        auc_roc = roc_auc_score(y_true, y_score)
        
    return {'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC-ROC': auc_roc}

class STL3SigmaAnomalyDetector:
    """STL + 3-Sigma 检测器"""
    def __init__(self, period=24, seasonal=25, robust=True):
        self.period = period
        self.seasonal = seasonal if seasonal % 2 == 1 else seasonal + 1
        self.robust = robust
        if self.seasonal <= self.period: self.seasonal = self.period + (2 - self.period % 2)
        print(f"🔧 STL+3σ Detector Initialized: STL(period={period}, seasonal={self.seasonal})")

    def detect_anomalies(self, data):
        print("🔄 Running STL+3σ point-wise anomaly detection...")
        series = pd.Series(data.flatten()).fillna(method='ffill').fillna(method='bfill')
        if len(series) < 2 * self.period: raise ValueError(f"Data length is too short for STL period")
        
        stl_result = STL(series, seasonal=self.seasonal, period=self.period, robust=self.robust).fit()
        residuals = stl_result.resid.dropna()
        
        mean, std = residuals.mean(), residuals.std()
        threshold = 3 * std
        point_anomalies = (np.abs(residuals - mean) > threshold).astype(int)
        
        full_labels = np.zeros(len(series), dtype=int)
        np.put(full_labels, residuals.index, point_anomalies)
        
        anomaly_scores = np.zeros(len(series), dtype=float)
        np.put(anomaly_scores, residuals.index, np.abs(residuals - mean) / std)

        print(f"✅ STL+3σ detection complete. Found {np.sum(full_labels)} anomaly points.")
        return full_labels, anomaly_scores

class STLLOFAnomalyDetector:
    """STL + LOF 检测器 (原版)"""
    def __init__(self, period=24, seasonal=25, robust=True, n_neighbors=20, contamination='auto'):
        self.period = period
        self.seasonal = seasonal if seasonal % 2 == 1 else seasonal + 1
        self.robust = robust
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        if self.seasonal <= self.period: self.seasonal = self.period + (2 - self.period % 2)
        print(f"🔧 STL+LOF Detector Initialized: STL(period={period}, seasonal={self.seasonal}), LOF(contamination={contamination})")

    def detect_anomalies(self, data):
        print("🔄 Running STL+LOF point-wise anomaly detection...")
        series = pd.Series(data.flatten()).fillna(method='ffill').fillna(method='bfill')
        if len(series) < 2 * self.period: raise ValueError(f"Data length is too short for STL period")
        
        stl_result = STL(series, seasonal=self.seasonal, period=self.period, robust=self.robust).fit()
        residuals = stl_result.resid.dropna()
        residuals_2d = residuals.values.reshape(-1, 1)
        
        n_neighbors = self.n_neighbors
        if len(residuals_2d) < n_neighbors: n_neighbors = max(1, len(residuals_2d) - 1)
        if n_neighbors <= 0: raise ValueError("Not enough residual points for LOF.")

        lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination)
        lof_labels = lof_model.fit_predict(residuals_2d)
        
        full_labels = np.zeros(len(series), dtype=int)
        np.put(full_labels, residuals.index, (lof_labels == -1).astype(int))
        
        anomaly_scores = np.zeros(len(series), dtype=float)
        # LOF scores are negative, larger magnitude is more anomalous. We invert and scale them.
        lof_scores = lof_model.negative_outlier_factor_
        np.put(anomaly_scores, residuals.index, -lof_scores)

        print(f"✅ STL+LOF detection complete. Found {np.sum(full_labels)} anomaly points.")
        return full_labels, anomaly_scores

class AutoencoderAnomalyDetector(nn.Module):
    """基于LSTM的自编码器模型"""
    def __init__(self, input_dim, seq_len, hidden_dim=64, latent_dim=16, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.latent_fc = nn.Linear(hidden_dim * 2 * seq_len, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim * 2 * seq_len)
        self.decoder = nn.LSTM(hidden_dim * 2, input_dim, num_layers, batch_first=True)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, _, _ = x.shape
        x, (hidden, _) = self.encoder(x)
        x = x.reshape(batch_size, -1)
        latent = self.latent_fc(x)
        x_recon = self.decoder_fc(latent)
        x_recon = x_recon.reshape(batch_size, self.seq_len, self.hidden_dim * 2)
        x_recon, _ = self.decoder(x_recon)
        return x_recon

    def train_model(self, X_train, device, epochs=20, batch_size=64, lr=1e-3):
        print("🧠 Training Autoencoder...")
        self.to(device)
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_loader = DataLoader(torch.FloatTensor(X_train), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"AE Epoch {epoch+1}/{epochs}", leave=False):
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed = self(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"AE Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")
        print("✅ Autoencoder training complete.")

    def predict(self, X_test, device):
        print("🔍 Predicting with Autoencoder...")
        self.to(device)
        self.eval()
        test_loader = DataLoader(torch.FloatTensor(X_test), batch_size=256, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="AE Prediction", leave=False):
                batch = batch.to(device)
                reconstructed = self(batch)
                loss = torch.mean((batch - reconstructed) ** 2, dim=[1, 2])
                scores.extend(loss.cpu().numpy())
        
        scores = np.array(scores)
        # Heuristic thresholding: anomalies are top 5% of reconstruction errors
        threshold = np.percentile(scores, 95)
        preds = (scores > threshold).astype(int)
        return preds, scores

def run_isolation_forest(X_train, X_test, random_state=42):
    """运行孤立森林模型"""
    print("🌳 Training Isolation Forest...")
    # Reshape data from (n_samples, window_size, n_features) to (n_samples, window_size * n_features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=random_state, n_jobs=-1)
    model.fit(X_train_flat)
    
    print("🔍 Predicting with Isolation Forest...")
    preds_raw = model.predict(X_test_flat)
    preds = (preds_raw == -1).astype(int) # Convert from {-1, 1} to {1, 0}
    scores = -model.score_samples(X_test_flat) # score_samples returns the negative of the anomaly score
    
    print("✅ Isolation Forest complete.")
    return preds, scores

# =================================
# GUI交互式标注界面 (来自 v3.0)
# =================================

class AnnotationGUI:
    # ... (此处代码与 v3.1 完全相同，为简洁起见省略) ...
    # ... (在实际文件中，请将 v3.1 的 AnnotationGUI 类完整复制到此处) ...
    def __init__(self, window_size=288):
        self.window_size = window_size
        self.result = None
        self.root = None

    def create_gui(self, window_data, window_idx, original_data_segment=None, auto_predicted_label=None):
        self.result = None
        if self.root:
            try: self.root.destroy()
            except: pass
        
        self.root = tk.Tk()
        self.root.title(f"Anomaly Annotation - Window #{window_idx}")
        self.root.geometry("1200x900")
        self.root.configure(bg='#f0f0f0')
        self.root.lift(); self.root.attributes('-topmost', True); self.root.after_idle(self.root.attributes, '-topmost', False)
        
        main_container = tk.Frame(self.root, bg='#f0f0f0'); main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main_container, text=f"Annotate Window #{window_idx}", font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#2c3e50').pack(pady=(0,10))
        
        # Chart
        fig = Figure(figsize=(11, 6), dpi=100); fig.patch.set_facecolor('#f0f0f0')
        ax1 = fig.add_subplot(211)
        ax1.plot(window_data.flatten(), 'b-', lw=1.5, label='Standardized Data'); ax1.set_title('Standardized Data', fontsize=11); ax1.grid(True, alpha=0.3); ax1.legend()
        ax2 = fig.add_subplot(212)
        if original_data_segment is not None:
            ax2.plot(original_data_segment.flatten(), 'r-', lw=1.5, label='Original Data'); ax2.set_title('Original Data', fontsize=11); ax2.grid(True, alpha=0.3); ax2.legend()
        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, main_container); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Buttons
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
        if self.root: self.root.quit(); self.root.destroy()

    def on_close(self):
        self.set_result(-2)

    def get_annotation(self, *args, **kwargs):
        try:
            root = self.create_gui(*args, **kwargs)
            if root is None: return self.get_annotation_fallback(*args, **kwargs)
            root.mainloop()
            return self.result if self.result is not None else -2
        except Exception: return self.get_annotation_fallback(*args, **kwargs)

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
    # ... (此处代码与 v3.1 完全相同，为简洁起见省略) ...
    # ... (在实际文件中，请将 v3.1 的 HumanAnnotationSystem 类完整复制到此处) ...
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
            label = self.gui.get_annotation_fallback(window_data, window_idx, original_data_segment, auto_predicted_label)
        
        if label in [0, 1]:
            self.annotation_history.append({'window_idx': window_idx, 'label': label, 'timestamp': datetime.now()})
            self.save_annotations()
            print(f"✅ 已标注窗口 #{window_idx} 为: {'异常' if label == 1 else '正常'}")
        return label

# =================================
# 数据集与数据加载 (来自 v3.0 逻辑)
# =================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, raw_data=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        # 确保 raw_data 也是 torch.FloatTensor
        self.raw_data = torch.FloatTensor(raw_data) if raw_data is not None else torch.zeros_like(self.X)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 返回元组，更符合常规用法
        return self.X[idx], self.y[idx], self.raw_data[idx]

def load_hydraulic_data_with_stl_lof(data_path, window_size, stride, specific_feature_column,
                                     stl_period=24, lof_contamination=0.02, unlabeled_fraction=0.1):
    print(f"📥 Loading data: {data_path}")
    df = pd.read_csv(data_path)
    
    if '1#' in df.columns:
        df.rename(columns={'1#': '102#'}, inplace=True)
    
    if specific_feature_column and specific_feature_column in df.columns:
        selected_cols = [specific_feature_column]
    else:
        # Fallback to the first numeric column if not specified or not found
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols: raise ValueError("No numeric columns found in the data.")
        selected_cols = [numeric_cols[0]]
    
    selected_cols = selected_cols[:1]
    print(f"➡️ Selected feature column: {selected_cols[0]} (shape will be: 1D)")
    
    df_for_point_mapping = df[['Date'] + selected_cols].copy() if 'Date' in df.columns else df[selected_cols].copy()
    data_values = df[selected_cols].fillna(method='ffill').fillna(method='bfill').fillna(0).values
    
    if data_values.ndim > 1 and data_values.shape[1] == 1:
        data_values = data_values.flatten()
    
    print(f"📊 Data shape after processing: {data_values.shape}")
    
    # This detector is used for initial pseudo-labeling for the RL agent
    stl_lof_detector = STLLOFAnomalyDetector(period=stl_period, contamination=lof_contamination)
    try:
        point_anomaly_labels, _ = stl_lof_detector.detect_anomalies(data_values)
    except Exception as e:
        print(f"❌ Error during initial STL+LOF labeling: {e}. Falling back to zeros.")
        point_anomaly_labels = np.zeros_like(data_values)

    if data_values.ndim == 1:
        data_values = data_values.reshape(-1, 1)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    print("🔄 Creating sliding windows...")
    windows_scaled, windows_raw, window_anomaly_labels, window_indices = [], [], [], []
    for i in range(0, len(data_scaled) - window_size + 1, stride):
        windows_scaled.append(data_scaled[i:i+window_size])
        windows_raw.append(data_values[i:i+window_size])
        window_anomaly_labels.append(point_anomaly_labels[i:i+window_size])
        window_indices.append(i)
    
    X, windows_raw_data = np.array(windows_scaled), np.array(windows_raw)
    window_start_indices_all = np.array(window_indices)
    
    y = np.array([1 if np.mean(labels) > 0.01 else 0 for labels in window_anomaly_labels])
    print(f"📊 Initial labels (STL+LOF): Normal={np.sum(y==0)}, Anomaly={np.sum(y==1)}")
    
    if np.sum(y == 1) == 0 and len(window_anomaly_labels) > 0:
        print("⚠️ No window-level anomalies detected, using a more aggressive strategy...")
        anomaly_scores = [np.sum(labels) for labels in window_anomaly_labels]
        threshold = np.percentile(anomaly_scores, 90)
        y = np.array([1 if score >= threshold and score > 0 else 0 for score in anomaly_scores])
        print(f"📊 Adjusted labels: Normal={np.sum(y==0)}, Anomaly={np.sum(y==1)}")
    
    unlabeled_mask = np.random.random(len(y)) < unlabeled_fraction
    y[unlabeled_mask] = -1 
    print(f"📊 Final labels: Normal={np.sum(y==0)}, Anomaly={np.sum(y==1)}, Unlabeled={np.sum(y==-1)}")
    
    indices = np.arange(len(X))
    # np.random.shuffle(indices) # <-- 避免对时间序列数据随机打乱
    train_size, val_size = int(0.7 * len(X)), int(0.15 * len(X))
    # 按时间顺序划分
    train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

    X_train, y_train, raw_train = X[train_indices], y[train_indices], windows_raw_data[train_indices]
    X_val, y_val, raw_val = X[val_indices], y[val_indices], windows_raw_data[val_indices]
    X_test, y_test, raw_test = X[test_indices], y[test_indices], windows_raw_data[test_indices]
    test_window_original_indices = window_start_indices_all[test_indices]
    
    print(f"✅ Data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return (X_train, y_train, raw_train, X_val, y_val, raw_val, X_test, y_test, raw_test, scaler, 
            df_for_point_mapping, test_window_original_indices, selected_cols[0], data_values.flatten())


# =================================
# 模型、经验回放与奖励函数 (来自 v3.0 逻辑)
# =================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class EnhancedRLADAgent(nn.Module):
    def __init__(self, input_dim, seq_len=288, hidden_size=128, num_layers=2):
        super(EnhancedRLADAgent, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        lstm_out_size = hidden_size * 2
        self.attention = nn.MultiheadAttention(lstm_out_size, num_heads=8, dropout=0.2, batch_first=True)
        self.ln_attention = nn.LayerNorm(lstm_out_size)
        self.fc_block = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2), nn.LayerNorm(hidden_size // 2), nn.GELU(), nn.Dropout(0.4)
        )
        self.q_head = nn.Linear(hidden_size // 2, 2)
        
    def forward(self, x, return_features=False, return_attention_weights=False):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.ln_attention(lstm_out + attn_out)
        pooled = torch.mean(x, dim=1) + torch.max(x, dim=1)[0]
        features = self.fc_block(pooled)
        q_values = self.q_head(features)
        
        if return_features and return_attention_weights: return q_values, features, attn_weights
        if return_features: return q_values, features
        if return_attention_weights: return q_values, attn_weights
        return q_values

    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon: return random.randint(0, 1)
        was_training = self.training; self.eval()
        with torch.no_grad():
            q_values = self(state.unsqueeze(0))
            action = q_values.argmax(dim=1).item()
        if was_training: self.train()
        return action

class PrioritizedReplayBuffer:
    def __init__(self, capacity=20000, alpha=0.6):
        self.capacity, self.alpha, self.buffer, self.pos = capacity, alpha, [], 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        exp = Experience(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity: self.buffer.append(exp)
        else: self.buffer[self.pos] = exp
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if not self.buffer: return None
        prios = self.priorities[:len(self.buffer)]; probs = prios ** self.alpha; probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta); weights /= weights.max()
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

def enhanced_compute_reward(action, true_label, is_human_labeled=False):
    if true_label == -1: return 0.0
    weight = 3.0 if is_human_labeled else 1.0
    if action == true_label: return (5.0 if true_label == 1 else 1.0) * weight
    else: return (-3.0 if true_label == 1 else -0.5) * weight

# =================================
# 训练与评估 (集成 v2.4 和 v3.0)
# =================================

def enhanced_warmup(replay_buffer, X_train, y_train, agent, device):
    """使用已标注数据预热经验回放池 (来自 v2.4)"""
    print("🔥 Pre-warming replay buffer...")
    labeled_mask = (y_train != -1)
    X_labeled, y_labeled = X_train[labeled_mask], y_train[labeled_mask]
    if len(X_labeled) == 0: print("No labeled data for warm-up."); return
    
    for idx in np.random.permutation(len(X_labeled))[:500]: # Warm up with up to 500 samples
        state = torch.FloatTensor(X_labeled[idx]).to(device)
        true_label = y_labeled[idx]
        action = agent.get_action(state)
        reward = enhanced_compute_reward(action, true_label, is_human_labeled=False)
        next_state = state # Simplified for warm-up
        replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), True)
    print(f"🔥 Warm-up complete. Buffer size: {len(replay_buffer)}")

def enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, gamma=0.99, batch_size=64, beta=0.4, scaler=None):
    """使用优先经验回放进行一步DQN训练 (GPU优化版本)"""
    if len(replay_buffer) < batch_size: return None
    sample = replay_buffer.sample(batch_size, beta)
    if not sample: return None
    states, actions, rewards, next_states, dones, indices, weights = sample
    
    states, actions, rewards, next_states, dones, weights = \
        states.to(device, non_blocking=True), actions.to(device, non_blocking=True), \
        rewards.to(device, non_blocking=True), next_states.to(device, non_blocking=True), \
        dones.to(device, non_blocking=True), weights.to(device, non_blocking=True)
    
    optimizer.zero_grad()
    
    use_amp = scaler is not None
    with torch.cuda.amp.autocast(enabled=use_amp):
        q_current = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = target_agent(next_states).max(1)[0]
            q_target = rewards + gamma * q_next * (~dones)
        td_errors = q_target - q_current
        loss = (weights * F.smooth_l1_loss(q_current, q_target, reduction='none')).mean()
    
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
    
    replay_buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy() + 1e-6)
    return loss.item()

def enhanced_evaluate_model(agent, data_loader, device):
    """详细的模型评估函数 (来自 v2.4)"""
    agent.eval()
    all_preds, all_labels, all_probs, all_features = [], [], [], []
    
    with torch.no_grad():
        # 修改此处：将字典索引改为元组解包
        for data, labels, _ in data_loader:
            data = data.to(device)
            q_values, features = agent(data, return_features=True)
            
            all_preds.extend(q_values.argmax(dim=1).cpu().numpy())
            all_probs.extend(F.softmax(q_values, dim=1)[:, 1].cpu().numpy())
            all_features.extend(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    agent.train()
    all_preds, all_labels, all_probs, all_features = np.array(all_preds), np.array(all_labels), np.array(all_probs), np.array(all_features)
    
    labeled_mask = (all_labels != -1)
    if not np.any(labeled_mask): return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc_roc': 0.0, 'labels': [], 'predictions': [], 'probabilities': [], 'features': [], 'all_predictions': all_preds, 'all_probabilities': all_probs}

    y_true, y_pred, y_scores, features_labeled = all_labels[labeled_mask], all_preds[labeled_mask], all_probs[labeled_mask], all_features[labeled_mask]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc_roc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    
    return {
        'precision': precision, 'recall': recall, 'f1': f1, 'auc_roc': auc_roc,
        'labels': y_true, 'predictions': y_pred, 'probabilities': y_scores, 'features': features_labeled,
        'all_predictions': all_preds, 'all_probabilities': all_probs
    }

def interactive_train_rlad_gui(agent, target_agent, optimizer, scheduler, replay_buffer, 
                              X_train, y_train, raw_train, X_val, y_val, raw_val, device, 
                              annotation_system, args):
    """交互式RLAD训练主循环 - 完全向量化和GPU优化版"""
    history = {k: [] for k in ['episodes', 'train_loss', 'val_f1', 'val_precision', 'val_recall', 'epsilon', 'learning_rate', 'human_annotations_count']}
    best_val_f1 = 0.0
    human_labeled_indices = set(np.where(y_train != -1)[0])
    unlabeled_idx_pool = deque(np.where(y_train == -1)[0])
    epsilon = args.epsilon_start
    
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)
    
    initial_labeled_count = torch.sum(y_train_gpu != -1).item()
    if args.batch_size_rl > initial_labeled_count and initial_labeled_count > 0:
        print(f"⚠️ 批处理大小 ({args.batch_size_rl}) 大于已标注样本数 ({initial_labeled_count})。")
        args.batch_size_rl = max(64, initial_labeled_count // 2)
        print(f"✅ 动态调整批处理大小为: {args.batch_size_rl}")

    val_dataset = TimeSeriesDataset(X_val, y_val, raw_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_rl, shuffle=False, num_workers=getattr(args, 'num_workers', 0), pin_memory=True)
    
    use_amp = args.mixed_precision and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print("\n🚀 Starting Fully Vectorized RLAD Training...")
    print(f"📊 批处理大小: {args.batch_size_rl}, 数据加载工作进程: {getattr(args, 'num_workers', 0)}")
    print(f"🔧 混合精度训练: {'启用' if use_amp else '禁用'}")
    
    steps_per_episode = max(50, len(X_train) // args.batch_size_rl)

    for episode in tqdm(range(args.num_episodes), desc="Training Progress"):
        agent.train()
        ep_losses = []
        
        # ---- 人工标注逻辑 ----
        if episode > 0 and episode % args.annotation_frequency == 0 and len(unlabeled_idx_pool) > 0:
            # ... (此处省略与v3.1相同的标注逻辑) ...
            pass

        # ---- 核心优化：向量化训练步骤 ----
        labeled_indices = torch.where(y_train_gpu != -1)[0]
        if len(labeled_indices) < args.batch_size_rl:
            continue

        for step in range(steps_per_episode):
            batch_indices = labeled_indices[torch.randint(len(labeled_indices), (args.batch_size_rl,))]
            states = X_train_gpu[batch_indices]
            true_labels = y_train_gpu[batch_indices]

            with torch.no_grad():
                q_values = agent(states)
                greedy_actions = q_values.argmax(dim=1)
                random_actions = torch.randint_like(greedy_actions, 0, 2)
                is_random = torch.rand(args.batch_size_rl, device=device) < epsilon
                actions = torch.where(is_random, random_actions, greedy_actions)

            is_human_labeled = torch.isin(batch_indices, torch.tensor(list(human_labeled_indices), device=device))
            weights = torch.where(is_human_labeled, 3.0, 1.0)
            rewards = torch.zeros_like(actions, dtype=torch.float)
            correct_mask = (actions == true_labels)
            rewards[correct_mask & (true_labels == 1)] = 5.0
            rewards[correct_mask & (true_labels == 0)] = 1.0
            rewards[~correct_mask & (true_labels == 1)] = -3.0
            rewards[~correct_mask & (true_labels == 0)] = -0.5
            rewards *= weights

            next_indices = (batch_indices + 1) % len(X_train_gpu)
            next_states = X_train_gpu[next_indices]
            
            for i in range(args.batch_size_rl):
                replay_buffer.push(states[i].cpu(), actions[i].item(), rewards[i].item(), next_states[i].cpu(), False)

            if len(replay_buffer) >= args.batch_size_rl:
                loss = enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, 
                                             batch_size=args.batch_size_rl, scaler=scaler)
                if loss: ep_losses.append(loss)

        if episode % args.target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())

        val_metrics = enhanced_evaluate_model(agent, val_loader, device)
        scheduler.step(val_metrics['f1'])
        
        history['episodes'].append(episode); history['train_loss'].append(np.mean(ep_losses) if ep_losses else 0)
        history['val_f1'].append(val_metrics['f1']); history['val_precision'].append(val_metrics['precision']); history['val_recall'].append(val_metrics['recall'])
        history['epsilon'].append(epsilon); history['learning_rate'].append(optimizer.param_groups[0]['lr']); history['human_annotations_count'].append(len(human_labeled_indices))
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(agent.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"\n⭐ New best model! Val F1: {best_val_f1:.4f} at epoch {episode}")

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay_rate)

    print(f"\n✅ Training complete! Best validation F1: {best_val_f1:.4f}")
    return history

# =================================
# 逐点标记与主函数 (集成对比实验)
# =================================

def mark_anomalies_pointwise(df_original, test_window_indices, test_predictions, window_size, feature_column, output_path):
    print("Mapping window predictions to point-wise labels...")
    df_original['pointwise_prediction'] = 0
    anomaly_window_indices = test_window_indices[test_predictions == 1]
    
    if len(anomaly_window_indices) > 0:
        point_indices_to_mark = set()
        for start_idx in tqdm(anomaly_window_indices, desc="Marking points"):
            for i in range(window_size):
                if start_idx + i < len(df_original):
                    point_indices_to_mark.add(start_idx + i)
        df_original.loc[list(point_indices_to_mark), 'pointwise_prediction'] = 1
    
    output_filename = os.path.join(output_path, f'predictions_{feature_column}.csv')
    df_original.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Point-wise prediction CSV saved to: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description='Optimized GUI-Interactive RLAD Anomaly Detection with Model Comparison')
    # ... (参数与v3.1相同) ...
    parser.add_argument('--data_path', type=str, default="clean_data.csv", help='Data file path')
    parser.add_argument('--feature_column', type=str, default=None, help='Feature column name')
    parser.add_argument('--output_dir', type=str, default="./output_rlad_v3_comparison", help='Output directory')
    parser.add_argument('--window_size', type=int, default=288, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=12, help='Sliding window stride')
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--annotation_frequency', type=int, default=5, help='Annotation frequency (every N episodes)')
    parser.add_argument('--use_gui', action='store_true', default=True, help='Enable GUI for annotation')
    parser.add_argument('--no_gui', action='store_false', dest='use_gui', help='Disable GUI, use command line')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch_size_rl', type=int, default=512, help='RL training batch size')
    parser.add_argument('--target_update_freq', type=int, default=10, help='Target network update frequency')
    parser.add_argument('--epsilon_start', type=float, default=0.95)
    parser.add_argument('--epsilon_end', type=float, default=0.02)
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.995)
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='指定GPU ID')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器工作进程数 (Windows建议为0)')
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='启用混合精度训练')
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    if args.force_cpu: device = torch.device("cpu")
    elif torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu_id}")
    else: device = torch.device("cpu")
    
    print(f"设备: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---- 数据加载 ----
    (X_train, y_train, raw_train, X_val, y_val, raw_val, X_test, y_test, raw_test, scaler, 
     df_for_point_mapping, test_window_original_indices, selected_col, full_raw_data) = \
        load_hydraulic_data_with_stl_lof(args.data_path, args.window_size, args.stride, args.feature_column)

    comparison_results = {}
    all_model_scores = {} # <--- 新增字典，用于存储所有模型的分数
    # ---- 1. STL + 3-Sigma (Baseline) ----
    print("\n--- Running Baseline: STL + 3-Sigma ---")
    stl3s_detector = STL3SigmaAnomalyDetector(period=args.window_size // 12)
    point_preds, point_scores = stl3s_detector.detect_anomalies(full_raw_data)
    # Map point-wise predictions to window-wise predictions for evaluation
    window_preds_3s = np.array([1 if np.mean(point_preds[i:i+args.window_size]) > 0.01 else 0 for i in test_window_original_indices])
    window_scores_3s = np.array([np.mean(point_scores[i:i+args.window_size]) for i in test_window_original_indices])
    comparison_results['STL + 3-Sigma'] = evaluate_baseline(y_test, window_preds_3s, window_scores_3s)
    all_model_scores['STL + 3-Sigma'] = {'scores': window_scores_3s} # <--- 存储分数
    # ---- 2. STL + LOF (Standalone Baseline) ----
    print("\n--- Running Baseline: STL + LOF (Standalone) ---")
    stlof_detector = STLLOFAnomalyDetector(period=args.window_size // 12, contamination=0.05) # Use a reasonable contamination
    point_preds_lof, point_scores_lof = stlof_detector.detect_anomalies(full_raw_data)
    window_preds_lof = np.array([1 if np.mean(point_preds_lof[i:i+args.window_size]) > 0.01 else 0 for i in test_window_original_indices])
    window_scores_lof = np.array([np.mean(point_scores_lof[i:i+args.window_size]) for i in test_window_original_indices])
    comparison_results['STL-LOF'] = evaluate_baseline(y_test, window_preds_lof, window_scores_lof)
    all_model_scores['STL-LOF'] = {'scores': window_scores_lof} # <--- 存储分数
    # ---- 3. Isolation Forest (Baseline) ----
    print("\n--- Running Baseline: Isolation Forest ---")
    if_preds, if_scores = run_isolation_forest(X_train, X_test, random_state=args.seed)
    comparison_results['Isolation Forest'] = evaluate_baseline(y_test, if_preds, if_scores)
    all_model_scores['Isolation Forest'] = {'scores': if_scores} # <--- 存储分数
    # ---- 4. Autoencoder (Baseline) ----
    print("\n--- Running Baseline: Autoencoder ---")
    ae_model = AutoencoderAnomalyDetector(input_dim=X_train.shape[2], seq_len=args.window_size)
    ae_model.train_model(X_train, device, epochs=20, batch_size=args.batch_size_rl)
    ae_preds, ae_scores = ae_model.predict(X_test, device)
    comparison_results['Autoencoder'] = evaluate_baseline(y_test, ae_preds, ae_scores)
    all_model_scores['Autoencoder'] = {'scores': ae_scores} # <--- 存储分数
    # ---- 5. STL-LOF-RLAD (Our Model) ----
    print("\n--- Running Main Model: STL-LOF-RLAD ---")
    agent = EnhancedRLADAgent(input_dim=X_train.shape[2], seq_len=args.window_size).to(device)
    target_agent = EnhancedRLADAgent(input_dim=X_train.shape[2], seq_len=args.window_size).to(device)
    target_agent.load_state_dict(agent.state_dict())
    optimizer = optim.AdamW(agent.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)
    replay_buffer = PrioritizedReplayBuffer()
    annotation_system = HumanAnnotationSystem(args.output_dir, args.window_size, args.use_gui)

    enhanced_warmup(replay_buffer, X_train, y_train, agent, device)
    
    training_history = interactive_train_rlad_gui(agent, target_agent, optimizer, scheduler, replay_buffer,
                                                  X_train, y_train, raw_train, X_val, y_val, raw_val,
                                                  device, annotation_system, args)
    
    # Final Evaluation on Test Set
    print("\n--- Final Evaluation of STL-LOF-RLAD on Test Set ---")
    agent.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_dataset = TimeSeriesDataset(X_test, y_test, raw_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_rl, shuffle=False)
    final_metrics = enhanced_evaluate_model(agent, test_loader, device)
    
    comparison_results['STL-LOF-RLAD (Ours)'] = {
        'Precision': final_metrics['precision'],
        'Recall': final_metrics['recall'],
        'F1-Score': final_metrics['f1'],
        'AUC-ROC': final_metrics['auc_roc']
    }
    all_model_scores['STL-LOF-RLAD (Ours)'] = {'scores': final_metrics['all_probabilities']} # <--- 存储分数
    # ---- Results and Visualization ----
    print("\n========================================")
    print("          Model Comparison Results")
    print("========================================")
    results_df = pd.DataFrame(comparison_results).T
    print(results_df)
    results_df.to_csv(os.path.join(args.output_dir, 'comparison_results.csv'))
    
    visualizer = CoreMetricsVisualizer(output_dir=os.path.join(args.output_dir, 'visuals'))
    # 调用新的对比绘图函数
    visualizer.plot_multiple_roc_curves(all_model_scores, y_test)
    visualizer.plot_multiple_pr_curves(all_model_scores, y_test)
    visualizer.generate_all_core_visualizations(
        training_history=training_history,
        final_metrics=final_metrics,
        original_data=full_raw_data,
        window_indices=test_window_original_indices,
        window_size=args.window_size,
        agent=agent,
        sample_data=X_test[0],
        device=device,
        comparison_results=comparison_results
    )
    
    mark_anomalies_pointwise(df_for_point_mapping, test_window_original_indices, final_metrics['all_predictions'],
                             args.window_size, selected_col, args.output_dir)

    print("\n✅ Experiment finished successfully!")

if __name__ == "__main__":
    main()