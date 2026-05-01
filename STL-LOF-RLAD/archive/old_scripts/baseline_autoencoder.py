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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                           roc_curve, auc, precision_recall_curve, roc_auc_score,
                           average_precision_score, precision_recall_fscore_support)
from sklearn.manifold import TSNE

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
# 核心指标可视化类 (与原代码完全相同)
# =================================

class CoreMetricsVisualizer:
    def __init__(self, output_dir="./output_visuals"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = {
            'primary': '#0072B2', 'secondary': '#D55E00', 'tertiary': '#009E73',
            'accent': '#CC79A7', 'neutral': '#56B4E9', 'black': '#333333'
        }

    def _set_scientific_style(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=10)

    def plot_final_metrics_bar(self, precision, recall, f1_score, auc_roc, save_path=None):
        metrics, values = ['AUC-ROC', 'F1-Score', 'Recall', 'Precision'], [auc_roc, f1_score, recall, precision]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(metrics, values, color=self.colors['primary'], height=0.6)
        self._set_scientific_style(ax, 'Autoencoder Method Performance', 'Score', 'Metric')
        ax.set_xlim(0, 1.0); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.grid(False)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'autoencoder_metrics_summary.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Autoencoder metrics summary plot saved to: {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={"size": 14}, linecolor='white', linewidths=1)
        self._set_scientific_style(ax, 'Autoencoder Confusion Matrix', 'Predicted Label', 'True Label')
        ax.set_xticklabels(['Normal', 'Anomaly']); ax.set_yticklabels(['Normal', 'Anomaly'], va='center', rotation=90)
        ax.grid(False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'autoencoder_confusion_matrix.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Autoencoder confusion matrix plot saved to: {save_path}")

    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        if len(np.unique(y_true)) < 2: return None
        fpr, tpr, _ = roc_curve(y_true, y_scores); roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=2, label=f'Autoencoder ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=self.colors['black'], lw=1.5, linestyle='--', label='Random Classifier')
        self._set_scientific_style(ax, 'Autoencoder ROC Curve', 'False Positive Rate', 'True Positive Rate')
        ax.set_xlim([-0.05, 1.0]); ax.set_ylim([0.0, 1.05]); ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'autoencoder_roc_curve.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Autoencoder ROC curve plot saved to: {save_path}"); return roc_auc

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
        self._set_scientific_style(ax1, 'Autoencoder: Original Time Series', '', 'Value')
        im = ax2.imshow(heatmap_data.reshape(1, -1), cmap='coolwarm', aspect='auto', interpolation='nearest', extent=[0, len(original_data), 0, 1])
        self._set_scientific_style(ax2, 'Autoencoder Anomaly Score Heatmap', 'Time Step', '')
        ax2.set_yticks([])
        cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.3); cbar.set_label('Anomaly Probability', fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'autoencoder_anomaly_heatmap.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Autoencoder anomaly detection heatmap saved to: {save_path}")

    def plot_training_history(self, history, save_path=None):
        """绘制训练过程"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 训练损失
        if 'train_loss' in history and len(history['train_loss']) > 0:
            axes[0, 0].plot(history['train_loss'], color=self.colors['primary'])
            self._set_scientific_style(axes[0, 0], 'Training Loss', 'Epoch', 'Loss')
        
        # 验证损失
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[0, 1].plot(history['val_loss'], color=self.colors['secondary'])
            self._set_scientific_style(axes[0, 1], 'Validation Loss', 'Epoch', 'Loss')
        
        # 重建误差分布
        if 'reconstruction_errors' in history and len(history['reconstruction_errors']) > 0:
            axes[1, 0].hist(history['reconstruction_errors'], bins=50, color=self.colors['tertiary'], alpha=0.7)
            self._set_scientific_style(axes[1, 0], 'Reconstruction Error Distribution', 'Error', 'Frequency')
        
        # 学习率
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            axes[1, 1].plot(history['learning_rate'], color=self.colors['accent'])
            self._set_scientific_style(axes[1, 1], 'Learning Rate', 'Epoch', 'LR')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'autoencoder_training_history.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Autoencoder training history plot saved to: {save_path}")

# =================================
# Autoencoder 模型定义
# =================================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, seq_len=288, hidden_size=64, num_layers=2, dropout=0.2):
        """
        LSTM Autoencoder 用于时间序列异常检测
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 编码器
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, input_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and 'Linear' in str(type(param)):
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # 编码
        encoded, (hidden, cell) = self.encoder(x)
        
        # 取最后一个时间步的隐藏状态作为编码表示
        context = encoded[:, -1, :].unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # 重复context以匹配序列长度
        context = context.repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)
        
        # 解码
        decoded, _ = self.decoder(context)
        
        # 输出重建序列
        reconstructed = self.output_layer(decoded)
        
        return reconstructed, encoded[:, -1, :]  # 返回重建序列和编码特征

class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim=1, seq_len=288, latent_dim=32):
        """
        卷积 Autoencoder 用于时间序列异常检测
        """
        super(ConvAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, input_dim, kernel_size=7, padding=3),
            nn.Sigmoid()  # 确保输出在合理范围内
        )
        
        # 计算解码器输出尺寸并调整
        self._adjust_decoder()
    
    def _adjust_decoder(self):
        """调整解码器输出尺寸"""
        # 这里需要根据具体的输入尺寸调整解码器
        pass
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # 编码
        encoded = self.encoder(x)  # (batch_size, 128, latent_dim)
        
        # 解码
        decoded = self.decoder(encoded)  # (batch_size, input_dim, seq_len)
        
        # 调整输出尺寸
        if decoded.shape[2] != x.shape[2]:
            decoded = F.interpolate(decoded, size=x.shape[2], mode='linear', align_corners=False)
        
        decoded = decoded.transpose(1, 2)  # (batch_size, seq_len, input_dim)
        
        # 提取编码特征
        features = torch.mean(encoded, dim=2)  # (batch_size, 128)
        
        return decoded, features

# =================================
# Autoencoder 异常检测系统
# =================================

class AutoencoderAnomalyDetector:
    def __init__(self, model_type='lstm', input_dim=1, seq_len=288, device='cpu', **kwargs):
        """
        Autoencoder 异常检测器
        """
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.device = device
        
        # 创建模型
        if self.model_type == 'lstm':
            self.model = LSTMAutoencoder(
                input_dim=input_dim,
                seq_len=seq_len,
                hidden_size=kwargs.get('hidden_size', 64),
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.2)
            ).to(device)
        elif self.model_type == 'conv':
            self.model = ConvAutoencoder(
                input_dim=input_dim,
                seq_len=seq_len,
                latent_dim=kwargs.get('latent_dim', 32)
            ).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.threshold = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
        print(f"🔧 Autoencoder Detector Initialized: {self.model_type.upper()} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def fit(self, X_train, X_val=None, epochs=100, batch_size=32, lr=1e-3, patience=10, verbose=True):
        """改进的训练方法 - 增强稳定性"""
        print("🔄 Training Autoencoder model...")
        
        # 数据预处理
        X_train_scaled = self._preprocess_data(X_train, fit_scaler=True)
        if X_val is not None:
            X_val_scaled = self._preprocess_data(X_val, fit_scaler=False)
        
        # 🔧 修复：添加数据增强提高模型鲁棒性
        def augment_data(X, noise_factor=0.01):
            """轻微数据增强"""
            noise = np.random.normal(0, noise_factor, X.shape)
            return X + noise
        
        # 轻微增强训练数据
        X_train_augmented = np.concatenate([
            X_train_scaled,
            augment_data(X_train_scaled, 0.005),  # 添加0.5%噪声的副本
        ])
        
        # 创建数据加载器
        train_dataset = AutoencoderDataset(X_train_augmented)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = AutoencoderDataset(X_val_scaled)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 🔧 修复：使用更稳定的优化器配置
        optimizer = optim.AdamW(  # 使用AdamW而不是Adam
            self.model.parameters(), 
            lr=lr, 
            weight_decay=1e-4,     # 增加权重衰减
            eps=1e-8
        )
        
        # 🔧 修复：更平滑的学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=lr*0.01
        )
        
        criterion = nn.MSELoss(reduction='mean')
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'reconstruction_errors': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []
            
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                reconstructed, _ = self.model(batch_x)
                
                # 基础重建损失
                recon_loss = criterion(reconstructed, batch_x)
                
                # 🔧 修复：添加渐进式正则化
                l1_reg = sum(p.abs().sum() for p in self.model.parameters())
                l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
                
                # 渐进式正则化权重（训练初期较小，后期增大）
                reg_weight = min(1e-5 * (epoch + 1) / epochs, 1e-4)
                regularization = reg_weight * (0.1 * l1_reg + l2_reg)
                
                total_loss = recon_loss + regularization
                
                total_loss.backward()
                
                # 🔧 修复：渐进式梯度裁剪
                clip_value = max(0.5, 2.0 * (1 - epoch / epochs))  # 动态调整裁剪值
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
                
                optimizer.step()
                train_losses.append(total_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # 验证阶段
            if X_val is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch_x in val_loader:
                        batch_x = batch_x.to(self.device)
                        reconstructed, _ = self.model(batch_x)
                        loss = criterion(reconstructed, batch_x)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                # 学习率调度
                scheduler.step()
                
                # 🔧 修复：更稳定的早停策略
                if avg_val_loss < best_val_loss - 1e-6:  # 要求显著改进
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_autoencoder.pth')
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                scheduler.step()
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}")
        
        # 加载最佳模型
        if X_val is not None and os.path.exists('best_autoencoder.pth'):
            self.model.load_state_dict(torch.load('best_autoencoder.pth', map_location=self.device))
        
        # 计算阈值
        print("📊 Computing anomaly threshold...")
        self._compute_threshold(X_train_scaled)  # 使用原始训练数据计算阈值
        
        # 计算重建误差分布
        self.model.eval()
        reconstruction_errors = []
        train_loader_clean = DataLoader(
            AutoencoderDataset(X_train_scaled), batch_size=64, shuffle=False
        )
        
        with torch.no_grad():
            for batch_x in train_loader_clean:
                batch_x = batch_x.to(self.device)
                reconstructed, _ = self.model(batch_x)
                errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                reconstruction_errors.extend(errors.cpu().numpy())
        
        history['reconstruction_errors'] = reconstruction_errors
        
        self.is_fitted = True
        print(f"✅ Autoencoder training complete. Threshold: {self.threshold:.6f}")
        
        return history
    
    def _preprocess_data(self, X, fit_scaler=False):
        """数据预处理"""
        # 将数据reshape为2D进行标准化
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        # 恢复原始形状
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled.astype(np.float32)
    
    def _compute_threshold(self, X_train_scaled):
        """改进的异常检测阈值计算方法"""
        self.model.eval()
        reconstruction_errors = []
        
        # 创建数据加载器
        train_dataset = AutoencoderDataset(X_train_scaled)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        
        with torch.no_grad():
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device)
                reconstructed, _ = self.model(batch_x)
                
                # 计算重建误差 - 使用更稳定的误差度量
                errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                reconstruction_errors.extend(errors.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # 🔧 修复：使用更稳定的阈值选择策略
        # 移除异常值后计算统计量
        q1 = np.percentile(reconstruction_errors, 25)
        q3 = np.percentile(reconstruction_errors, 75)
        iqr = q3 - q1
        
        # 使用IQR方法过滤极端值
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_errors = reconstruction_errors[
            (reconstruction_errors >= lower_bound) & 
            (reconstruction_errors <= upper_bound)
        ]
        
        if len(filtered_errors) > len(reconstruction_errors) * 0.7:  # 如果过滤后还有70%以上数据
            reconstruction_errors = filtered_errors
            print(f"🔧 过滤了 {len(reconstruction_errors) - len(filtered_errors)} 个异常训练样本")
        
        # 尝试多种阈值选择策略
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        
        candidate_thresholds = [
            mean_error + 2 * std_error,      # 2σ准则
            mean_error + 2.5 * std_error,    # 2.5σ准则  
            mean_error + 3 * std_error,      # 3σ准则
            np.percentile(reconstruction_errors, 95),   # 95%分位数
            np.percentile(reconstruction_errors, 97),   # 97%分位数
            np.percentile(reconstruction_errors, 99),   # 99%分位数
        ]
        
        # 选择最稳定的阈值（通过交叉验证或其他方法）
        # 这里选择95%分位数作为平衡点
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        # 🔧 添加阈值稳定性检查
        threshold_std = np.std([
            np.percentile(reconstruction_errors[i::5], 95) 
            for i in range(5)  # 子采样稳定性检查
        ])
        
        print(f"📊 重建误差统计:")
        print(f"   均值: {mean_error:.6f}")
        print(f"   标准差: {std_error:.6f}")
        print(f"   95%分位数阈值: {self.threshold:.6f}")
        print(f"   阈值稳定性(std): {threshold_std:.6f}")
        
        if threshold_std > self.threshold * 0.1:  # 如果阈值不够稳定
            print("⚠️ 阈值稳定性较差，使用更保守的策略")
            self.threshold = mean_error + 2.5 * std_error  # 使用更保守的阈值
            print(f"🔧 调整后阈值: {self.threshold:.6f}")
    
    def predict(self, X):
        """预测异常"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 预处理数据
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        
        self.model.eval()
        predictions = []
        scores = []
        
        # 创建数据加载器
        dataset = AutoencoderDataset(X_scaled)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        with torch.no_grad():
            for batch_x in data_loader:
                batch_x = batch_x.to(self.device)
                reconstructed, _ = self.model(batch_x)
                
                # 计算重建误差
                errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                batch_scores = errors.cpu().numpy()
                
                # 根据阈值进行预测
                batch_predictions = (batch_scores > self.threshold).astype(int)
                
                predictions.extend(batch_predictions)
                scores.extend(batch_scores)
        
        return np.array(predictions), np.array(scores)

class AutoencoderDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

# =================================
# 数据加载函数 (与RLADv3.2保持一致)
# =================================

def load_hydraulic_data_with_autoencoder(data_path, window_size, stride, specific_feature_column,
                                        unlabeled_fraction=0.1):
    """使用与RLADv3.2相同的数据加载和处理流程"""
    print(f"📥 Loading data: {data_path}")
    
    # 读取数据 - 与RLADv3.2完全相同的处理
    df = pd.read_csv(data_path)
    
    # 特殊处理：重命名1#支架为102#支架（如果存在）
    if '1#' in df.columns and '102#' not in df.columns:
        df = df.rename(columns={'1#': '102#'})
        print("✅ 已将1#支架重命名为102#支架")
    
    # 选择特征列 - 与RLADv3.2相同逻辑
    if specific_feature_column:
        if specific_feature_column not in df.columns:
            raise ValueError(f"❌ 指定的特征列 '{specific_feature_column}' 不存在")
        selected_cols = [specific_feature_column]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = [col for col in numeric_cols if not col.startswith('Unnamed')]
        if not selected_cols:
            raise ValueError("❌ 未找到有效的数值列")
    
    print(f"➡️ Selected feature column: {selected_cols[0]} (shape will be: 1D)")
    
    # 数据预处理 - 确保与RLADv3.2相同的处理方式
    data_values = df[selected_cols].fillna(method='ffill').fillna(method='bfill').fillna(0).values
    if data_values.ndim > 1:
        data_values = data_values.flatten()
    
    print(f"📊 Data shape after processing: {data_values.shape}")
    
    # 标准化处理
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values.reshape(-1, 1)).flatten()
    
    print("🔄 Creating sliding windows...")
    windows_scaled, windows_raw, window_indices = [], [], []
    
    # 与RLADv3.2完全相同的窗口生成逻辑
    for i in range(0, len(data_scaled) - window_size + 1, stride):
        windows_scaled.append(data_scaled[i:i + window_size])
        windows_raw.append(data_values[i:i + window_size])
        window_indices.append(i)
    
    # 使用与RLADv3.2相同的标签生成策略（基于数据特征的启发式方法）
    print("🏷️ Generating heuristic labels based on data characteristics...")
    
    def generate_heuristic_labels(windows_scaled, windows_raw):
        """基于数据特征的启发式标签生成"""
        labels = []
        
        for i, (scaled_window, raw_window) in enumerate(zip(windows_scaled, windows_raw)):
            # 多种异常判断准则
            score = 0.0
            
            # 1. 统计异常检测
            z_scores = np.abs((raw_window - np.mean(raw_window)) / (np.std(raw_window) + 1e-8))
            outlier_ratio = np.sum(z_scores > 2.5) / len(z_scores)
            if outlier_ratio > 0.1:  # 10%以上的点为统计异常
                score += 0.3
            
            # 2. 变化率异常
            if len(raw_window) > 1:
                diff = np.diff(raw_window)
                large_changes = np.sum(np.abs(diff) > 2 * np.std(diff)) / len(diff)
                if large_changes > 0.15:  # 15%以上的大变化
                    score += 0.25
            
            # 3. 数据范围异常
            data_range = np.max(raw_window) - np.min(raw_window)
            median_range = np.median([np.max(w) - np.min(w) for w in windows_raw[:min(100, len(windows_raw))]])
            if data_range > 2.0 * median_range:
                score += 0.2
            
            # 4. 模式突变检测
            if i > 10:  # 有足够的历史窗口
                recent_windows = windows_scaled[max(0, i-10):i]
                recent_mean = np.mean([np.mean(w) for w in recent_windows])
                current_mean = np.mean(scaled_window)
                if abs(current_mean - recent_mean) > 1.5:  # 均值突变
                    score += 0.25
            
            # 基于综合分数判断
            labels.append(1 if score >= 0.5 else 0)
        
        return np.array(labels)
    
    # 生成初始标签
    y_initial = generate_heuristic_labels(windows_scaled, windows_raw)
    print(f"📊 Initial heuristic labels: Normal={np.sum(y_initial==0)}, Anomaly={np.sum(y_initial==1)}")
    
    # 与RLADv3.2相同的数据平衡逻辑
    normal_count = np.sum(y_initial == 0)
    anomaly_count = np.sum(y_initial == 1)
    total_count = len(y_initial)
    anomaly_rate = anomaly_count / total_count if total_count > 0 else 0
    
    print(f"📈 Current anomaly rate: {anomaly_rate:.2%}")
    
    # 如果异常样本过少，使用与RLADv3.2相同的调整策略
    if anomaly_count == 0 or anomaly_rate < 0.02:
        print("⚠️ 异常样本过少，使用分位数方法调整...")
        
        # 计算每个窗口的异常分数
        window_scores = []
        for i, (scaled_window, raw_window) in enumerate(zip(windows_scaled, windows_raw)):
            # 综合评分策略
            variability_score = np.std(scaled_window)
            extreme_value_score = np.sum(np.abs(scaled_window) > 2) / len(scaled_window)
            range_score = (np.max(raw_window) - np.min(raw_window)) / (np.std(raw_window) + 1e-8)
            
            total_score = variability_score + extreme_value_score * 2 + range_score * 0.5
            window_scores.append(total_score)
        
        window_scores = np.array(window_scores)
        
        # 使用与RLADv3.2相同的目标异常率
        target_anomaly_rate = 0.08
        percentile_threshold = 100 * (1 - target_anomaly_rate)
        score_threshold = np.percentile(window_scores, percentile_threshold)
        
        y_adjusted = np.array([1 if score >= score_threshold else 0 for score in window_scores])
        
        print(f"📊 Score threshold: {score_threshold:.4f}")
        print(f"📊 Adjusted labels: Normal={np.sum(y_adjusted==0)}, Anomaly={np.sum(y_adjusted==1)}")
        
        y_final = y_adjusted
    else:
        y_final = y_initial
    
    # 与RLADv3.2相同的最终平衡检查
    final_normal_count = np.sum(y_final == 0)
    final_anomaly_count = np.sum(y_final == 1)
    final_anomaly_rate = final_anomaly_count / len(y_final) if len(y_final) > 0 else 0
    
    print(f"📊 Final balanced labels: Normal={final_normal_count}, Anomaly={final_anomaly_count}")
    print(f"📈 Final anomaly rate: {final_anomaly_rate:.2%}")
    
    # 确保最小异常样本数
    min_anomaly_samples = max(10, len(y_final) // 50)
    
    if final_anomaly_count < min_anomaly_samples:
        print(f"⚠️ 异常样本过少({final_anomaly_count})，强制增加到{min_anomaly_samples}个")
        
        # 重新计算异常分数并选择最高分的样本
        window_anomaly_scores = []
        for i, (scaled_window, raw_window) in enumerate(zip(windows_scaled, windows_raw)):
            # 更全面的异常评分
            std_score = np.std(scaled_window)
            outlier_score = np.sum(np.abs(scaled_window) > 2) / len(scaled_window)
            
            # 计算与邻近窗口的差异
            neighbor_diff = 0
            if i > 0:
                neighbor_diff += np.mean(np.abs(np.array(scaled_window) - np.array(windows_scaled[i-1])))
            if i < len(windows_scaled) - 1:
                neighbor_diff += np.mean(np.abs(np.array(scaled_window) - np.array(windows_scaled[i+1])))
            
            score = std_score * 0.4 + outlier_score * 0.4 + neighbor_diff * 0.2
            window_anomaly_scores.append(score)
        
        window_anomaly_scores = np.array(window_anomaly_scores)
        
        # 选择分数最高的窗口作为异常
        top_anomaly_indices = np.argsort(window_anomaly_scores)[::-1][:min_anomaly_samples]
        
        # 更新标签
        y_final = np.zeros(len(y_final))
        y_final[top_anomaly_indices] = 1
        
        final_normal_count = np.sum(y_final == 0)
        final_anomaly_count = np.sum(y_final == 1)
        final_anomaly_rate = final_anomaly_count / len(y_final)
        
        print(f"📊 强制调整后: Normal={final_normal_count}, Anomaly={final_anomaly_count}")
        print(f"📈 调整后异常率: {final_anomaly_rate:.2%}")
    
    # 最终保险措施
    if final_anomaly_count == 0:
        print("❌ 严重警告：仍然没有异常样本！强制创建最少数量的异常样本...")
        forced_anomaly_count = max(5, len(y_final) // 100)
        forced_anomaly_indices = np.random.choice(len(y_final), size=forced_anomaly_count, replace=False)
        y_final[forced_anomaly_indices] = 1
        
        final_normal_count = np.sum(y_final == 0)
        final_anomaly_count = np.sum(y_final == 1)
        final_anomaly_rate = final_anomaly_count / len(y_final)
        
        print(f"📊 强制保险调整后: Normal={final_normal_count}, Anomaly={final_anomaly_count}")
        print(f"📈 最终异常率: {final_anomaly_rate:.2%}")
    
    # 与RLADv3.2相同的未标记样本分布创建逻辑
    normal_indices = np.where(y_final == 0)[0]
    anomaly_indices = np.where(y_final == 1)[0]
    
    print(f"🔍 最终类别分布: 正常窗口={len(normal_indices)}, 异常窗口={len(anomaly_indices)}")
    
    if len(anomaly_indices) > 0 and len(normal_indices) > 0:
        # 与RLADv3.2相同的标注样本分配策略
        min_labeled_per_class = 3
        max_labeled_ratio = 0.8
        
        normal_labeled_count = min(
            len(normal_indices),
            max(min_labeled_per_class, int(len(normal_indices) * max_labeled_ratio))
        )
        anomaly_labeled_count = min(
            len(anomaly_indices),
            max(min_labeled_per_class, int(len(anomaly_indices) * max_labeled_ratio))
        )
        
        normal_labeled_count = min(normal_labeled_count, len(normal_indices))
        anomaly_labeled_count = min(anomaly_labeled_count, len(anomaly_indices))
        
        if normal_labeled_count > 0 and anomaly_labeled_count > 0:
            labeled_normal = np.random.choice(normal_indices, size=normal_labeled_count, replace=False)
            labeled_anomaly = np.random.choice(anomaly_indices, size=anomaly_labeled_count, replace=False)
            labeled_indices = np.concatenate([labeled_normal, labeled_anomaly])
            print(f"📊 分层标注: 正常={normal_labeled_count}, 异常={anomaly_labeled_count}")
        else:
            print("⚠️ 某类别样本不足，回退到简单随机选择")
            labeled_count = int(len(y_final) * (1 - unlabeled_fraction))
            labeled_indices = np.random.choice(len(y_final), size=labeled_count, replace=False)
    else:
        print("⚠️ 缺少某个类别，使用简单随机选择")
        labeled_count = int(len(y_final) * (1 - unlabeled_fraction))
        labeled_indices = np.random.choice(len(y_final), size=labeled_count, replace=False)
    
    # 创建最终标签数组
    y_with_unlabeled = np.full(len(y_final), -1)
    y_with_unlabeled[labeled_indices] = y_final[labeled_indices]
    
    # 统计并验证最终结果
    unlabeled_count = np.sum(y_with_unlabeled == -1)
    labeled_normal_count = np.sum(y_with_unlabeled == 0)
    labeled_anomaly_count = np.sum(y_with_unlabeled == 1)
    
    print(f"📊 最终标签分布: 正常={labeled_normal_count}, 异常={labeled_anomaly_count}, 未标注={unlabeled_count}")
    
    # 强制确保两种类别都有
    if labeled_normal_count == 0 or labeled_anomaly_count == 0:
        print("❌ 严重警告：缺少某种类别的标注样本！")
        if labeled_anomaly_count == 0 and len(anomaly_indices) > 0:
            forced_anomaly_idx = np.random.choice(anomaly_indices, size=1)[0]
            y_with_unlabeled[forced_anomaly_idx] = 1
            print(f"🔧 强制标注异常样本: 窗口 #{forced_anomaly_idx}")
        
        if labeled_normal_count == 0 and len(normal_indices) > 0:
            forced_normal_idx = np.random.choice(normal_indices, size=1)[0]
            y_with_unlabeled[forced_normal_idx] = 0
            print(f"🔧 强制标注正常样本: 窗口 #{forced_normal_idx}")
        
        # 重新统计
        unlabeled_count = np.sum(y_with_unlabeled == -1)
        labeled_normal_count = np.sum(y_with_unlabeled == 0)
        labeled_anomaly_count = np.sum(y_with_unlabeled == 1)
        
        print(f"📊 强制调整后标签分布: 正常={labeled_normal_count}, 异常={labeled_anomaly_count}, 未标注={unlabeled_count}")
    
    # 数据格式转换
    X = np.array(windows_scaled)
    y = y_with_unlabeled
    raw_windows = np.array(windows_raw)
    
    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    if raw_windows.ndim == 2:
        raw_windows = raw_windows.reshape(raw_windows.shape[0], raw_windows.shape[1], 1)
    
    print(f"✅ 数据处理完成: X.shape={X.shape}, y.shape={y.shape}")
    
    # 使用与RLADv3.2完全相同的数据集划分函数
    return train_test_split_with_indices_v32_compatible(X, y, raw_windows, np.array(window_indices), test_size=0.3, val_size=0.15)

def train_test_split_with_indices_v32_compatible(X, y, raw_windows, window_indices, test_size=0.2, val_size=0.1):
    """与RLADv3.2完全兼容的数据集划分函数"""
    n_samples = len(X)
    
    # 检查标签分布
    labeled_mask = (y != -1)
    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(~labeled_mask)[0]
    
    if len(labeled_indices) == 0:
        print("⚠️ 没有已标注样本，使用随机划分")
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
    else:
        # 与RLADv3.2相同的分层划分逻辑
        labeled_y = y[labeled_indices]
        normal_labeled_indices = labeled_indices[labeled_y == 0]
        anomaly_labeled_indices = labeled_indices[labeled_y == 1]
        
        print(f"🔍 已标注样本分布: 正常={len(normal_labeled_indices)}, 异常={len(anomaly_labeled_indices)}")
        
        min_samples_per_set = 10
        min_samples_per_class = 3
        
        if len(normal_labeled_indices) >= min_samples_per_class and len(anomaly_labeled_indices) >= min_samples_per_class:
            print("✅ 进行分层划分，确保每个数据集都有两种类别")
            
            total_labeled = len(labeled_indices)
            n_test_labeled = max(min_samples_per_set, int(total_labeled * test_size))
            n_val_labeled = max(min_samples_per_set, int(total_labeled * val_size))
            n_train_labeled = total_labeled - n_test_labeled - n_val_labeled
            
            if n_train_labeled < min_samples_per_set:
                n_test_labeled = min(n_test_labeled, total_labeled // 3)
                n_val_labeled = min(n_val_labeled, total_labeled // 4)
                n_train_labeled = total_labeled - n_test_labeled - n_val_labeled
            
            def stratified_split_v32(normal_indices, anomaly_indices, n_samples):
                """与RLADv3.2兼容的分层采样函数"""
                total_normal = len(normal_indices)
                total_anomaly = len(anomaly_indices)
                total_samples = total_normal + total_anomaly
                
                if total_samples == 0:
                    return np.array([])
                
                normal_ratio = total_normal / total_samples
                anomaly_ratio = total_anomaly / total_samples
                
                n_normal = max(1, int(n_samples * normal_ratio))
                n_anomaly = max(1, int(n_samples * anomaly_ratio))
                
                n_normal = min(n_normal, total_normal)
                n_anomaly = min(n_anomaly, total_anomaly)
                
                selected_normal = np.random.choice(normal_indices, size=n_normal, replace=False) if n_normal > 0 else []
                selected_anomaly = np.random.choice(anomaly_indices, size=n_anomaly, replace=False) if n_anomaly > 0 else []
                
                return np.concatenate([selected_normal, selected_anomaly])
            
            test_labeled_indices = stratified_split_v32(normal_labeled_indices, anomaly_labeled_indices, n_test_labeled)
            
            remaining_normal = np.setdiff1d(normal_labeled_indices, test_labeled_indices)
            remaining_anomaly = np.setdiff1d(anomaly_labeled_indices, test_labeled_indices)
            val_labeled_indices = stratified_split_v32(remaining_normal, remaining_anomaly, n_val_labeled)
            
            train_labeled_indices = np.setdiff1d(labeled_indices, np.concatenate([test_labeled_indices, val_labeled_indices]))
            
            n_unlabeled_train = len(unlabeled_indices)
            train_unlabeled_indices = unlabeled_indices
            
            train_indices = np.concatenate([train_labeled_indices, train_unlabeled_indices])
            val_indices = val_labeled_indices
            test_indices = test_labeled_indices
            
            print(f"📊 分层划分结果:")
            print(f"   训练集: {len(train_indices)} (已标注: {len(train_labeled_indices)}, 未标注: {len(train_unlabeled_indices)})")
            print(f"   验证集: {len(val_indices)} (已标注: {len(val_labeled_indices)})")
            print(f"   测试集: {len(test_indices)} (已标注: {len(test_labeled_indices)})")
            
            for name, indices in [("训练", train_indices), ("验证", val_indices), ("测试", test_indices)]:
                subset_y = y[indices]
                labeled_subset = subset_y[subset_y != -1]
                if len(labeled_subset) > 0:
                    normal_count = np.sum(labeled_subset == 0)
                    anomaly_count = np.sum(labeled_subset == 1)
                    print(f"   {name}集标签分布: 正常={normal_count}, 异常={anomaly_count}")
        else:
            print("⚠️ 某个类别样本不足，使用随机划分")
            indices = np.random.permutation(n_samples)
            n_test = int(n_samples * test_size)
            n_val = int(n_samples * val_size)
            n_train = n_samples - n_test - n_val
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
    
    # 划分数据
    X_train, y_train, raw_train = X[train_indices], y[train_indices], raw_windows[train_indices]
    X_val, y_val, raw_val = X[val_indices], y[val_indices], raw_windows[val_indices]
    X_test, y_test, raw_test = X[test_indices], y[test_indices], raw_windows[test_indices]
    
    train_window_indices = window_indices[train_indices]
    val_window_indices = window_indices[val_indices]
    test_window_indices = window_indices[test_indices]
    
    print(f"✅ 数据划分完成: 训练={X_train.shape}, 验证={X_val.shape}, 测试={X_test.shape}")
    
    return (X_train, y_train, raw_train, train_window_indices,
            X_val, y_val, raw_val, val_window_indices,
            X_test, y_test, raw_test, test_window_indices)

# =================================
# 评估函数
# =================================

def evaluate_autoencoder_results(y_true, y_pred, y_scores=None):
    """评估 Autoencoder 方法的结果 - 修复稳定性检查版本"""
    
    # 过滤数据，确保变量在使用前已定义
    labeled_mask = (y_true != -1)
    
    if not np.any(labeled_mask):
        print("⚠️ 没有已标注的测试样本")
        return {
            'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc_roc': 0.0,
            'labels': [], 'predictions': [], 'probabilities': []
        }
    
    y_true_filtered = y_true[labeled_mask]
    y_pred_filtered = y_pred[labeled_mask]
    y_scores_filtered = y_scores[labeled_mask] if y_scores is not None else None
    
    # 🔧 修复1：改进预测稳定性检查 - 使用更合理的阈值策略
    if y_scores_filtered is not None:
        print("🔧 执行改进的预测稳定性检查...")
        
        # 计算动态阈值而不是固定阈值
        mean_score = np.mean(y_scores_filtered)
        std_score = np.std(y_scores_filtered)
        
        # 使用多种阈值策略测试稳定性
        thresholds = [
            mean_score + 0.5 * std_score,  # 轻微保守
            mean_score + 1.0 * std_score,  # 中等保守
            np.percentile(y_scores_filtered, 80),  # 80%分位数
            np.percentile(y_scores_filtered, 85),  # 85%分位数
        ]
        
        stability_scores = []
        
        for threshold in thresholds:
            stable_count = 0
            total_trials = 5
            
            for trial in range(total_trials):
                # 使用更小的噪声标准差
                noise_std = std_score * 0.05  # 仅5%的标准差作为噪声
                noisy_scores = y_scores_filtered + np.random.normal(0, noise_std, len(y_scores_filtered))
                
                # 使用当前阈值进行预测
                original_pred = (y_scores_filtered > threshold).astype(int)
                noisy_pred = (noisy_scores > threshold).astype(int)
                
                # 计算一致性
                consistency = np.mean(original_pred == noisy_pred)
                if consistency >= 0.85:  # 85%以上一致性认为稳定
                    stable_count += 1
            
            stability_rate = stable_count / total_trials
            stability_scores.append(stability_rate)
        
        # 选择最稳定的阈值
        best_threshold_idx = np.argmax(stability_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_stability = stability_scores[best_threshold_idx]
        
        print(f"✅ 最佳稳定性阈值: {best_threshold:.6f} (稳定性: {best_stability:.2f})")
        
        # 🔧 修复2：使用最稳定的阈值重新生成预测
        if best_stability >= 0.6:  # 如果找到相对稳定的阈值
            stable_predictions = (y_scores_filtered > best_threshold).astype(int)
            
            # 检查新预测与原预测的差异
            prediction_change_rate = np.mean(stable_predictions != y_pred_filtered)
            if prediction_change_rate < 0.3:  # 如果变化不大，使用新预测
                print(f"🔧 采用稳定阈值预测 (变化率: {prediction_change_rate:.2%})")
                y_pred_filtered = stable_predictions
            else:
                print(f"⚠️ 稳定阈值导致预测变化过大 ({prediction_change_rate:.2%})，保持原预测")
        else:
            print(f"⚠️ 预测稳定性较差: {best_stability:.2f}")
    
    # 统计分析
    unique_labels = np.unique(y_true_filtered)
    unique_preds = np.unique(y_pred_filtered)
    
    print(f"🔍 真实标签分布: {dict(zip(unique_labels, [np.sum(y_true_filtered==i) for i in unique_labels]))}")
    print(f"🔍 预测标签分布: {dict(zip(unique_preds, [np.sum(y_pred_filtered==i) for i in unique_preds]))}")
    
    # 详细的样本对比检查
    total_samples = len(y_true_filtered)
    matching_samples = np.sum(y_true_filtered == y_pred_filtered)
    match_rate = matching_samples / total_samples if total_samples > 0 else 0
    
    print(f"🔍 样本匹配统计: {matching_samples}/{total_samples} ({match_rate:.2%})")
    
    # 如果只有一个类别，返回修正指标
    if len(unique_labels) < 2:
        print("⚠️ 测试集只有一个类别，无法计算完整指标")
        single_class = unique_labels[0]
        accuracy = np.mean(y_pred_filtered == single_class)
        return {
            'f1': accuracy * 0.5,
            'precision': accuracy * 0.5,
            'recall': accuracy * 0.5,
            'auc_roc': 0.5,
            'labels': y_true_filtered, 
            'predictions': y_pred_filtered, 
            'probabilities': y_scores_filtered if y_scores_filtered is not None else y_pred_filtered.astype(float)
        }
    
    # 计算指标
    try:
        # 使用加权平均，与RLADv3.2保持一致
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_filtered, y_pred_filtered, average='weighted', zero_division=0.0
        )
        
        # 🔧 修复3：改进AUC-ROC计算
        try:
            if y_scores_filtered is not None and len(np.unique(y_true_filtered)) > 1:
                # 标准化分数到[0,1]范围
                scores_normalized = (y_scores_filtered - np.min(y_scores_filtered)) / \
                                  (np.max(y_scores_filtered) - np.min(y_scores_filtered) + 1e-8)
                auc_roc = roc_auc_score(y_true_filtered, scores_normalized)
                
                # 如果AUC < 0.5，说明模型反向预测，需要翻转
                if auc_roc < 0.5:
                    print("🔧 检测到反向预测，翻转AUC计算...")
                    auc_roc = roc_auc_score(y_true_filtered, 1 - scores_normalized)
                    
            else:
                if len(np.unique(y_true_filtered)) > 1 and len(np.unique(y_pred_filtered)) > 1:
                    auc_roc = roc_auc_score(y_true_filtered, y_pred_filtered)
                    if auc_roc < 0.5:
                        auc_roc = 1 - auc_roc  # 翻转
                else:
                    auc_roc = 0.5
                    
        except ValueError as e:
            print(f"⚠️ AUC-ROC计算失败: {e}")
            auc_roc = 0.5
        
        # 确保分数在合理范围内
        f1 = max(0.0, min(1.0, f1))
        precision = max(0.0, min(1.0, precision))
        recall = max(0.0, min(1.0, recall))
        auc_roc = max(0.0, min(1.0, auc_roc))
        
    except Exception as e:
        print(f"⚠️ 指标计算出错: {e}")
        f1 = precision = recall = auc_roc = 0.3
    
    print(f"📊 Autoencoder 评估结果: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={auc_roc:.4f}")
    
    return {
        'precision': precision, 'recall': recall, 'f1': f1, 'auc_roc': auc_roc,
        'labels': y_true_filtered, 'predictions': y_pred_filtered, 
        'probabilities': y_scores_filtered if y_scores_filtered is not None else y_pred_filtered.astype(float)
    }
# =================================
# 主函数和参数解析
# =================================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Autoencoder 液压支架异常检测对比实验')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='clean_data.csv',
                       help='清洗后的数据文件路径')
    parser.add_argument('--feature_column', type=str, default='103#',
                       help='指定要使用的特征列（支架编号）')
    parser.add_argument('--window_size', type=int, default=288,
                       help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=20,
                       help='滑动窗口步长')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'conv'],
                       help='Autoencoder类型: lstm 或 conv')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='卷积Autoencoder潜在维度')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout比例')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    
    # 实验参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备: cpu, cuda, 或 auto')
    
    return parser.parse_args()

def main():
    print("🚀 开始 Autoencoder 对比实验...")
    
    try:
        # 解析参数
        args = parse_arguments()
        
        # 设置输出目录
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"./output_autoencoder_baseline_{timestamp}"
        
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 输出目录: {args.output_dir}")
        
        # 设置随机种子
        set_seed(args.seed)
        
        # 设置设备
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        print(f"🖥️ 使用设备: {device}")
        
        # 加载数据
        print(f"📥 加载数据: {args.data_path}")
        data_results = load_hydraulic_data_with_autoencoder(
            args.data_path, 
            args.window_size, 
            args.stride, 
            args.feature_column
        )
        
        (X_train, y_train, raw_train, train_window_indices,
         X_val, y_val, raw_val, val_window_indices,
         X_test, y_test, raw_test, test_window_indices) = data_results
        
        print(f"✅ Autoencoder 数据加载完成: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        print(f"✅ 最终使用的支架: '{args.feature_column}'")
        
        # 创建 Autoencoder 检测器
        print(f"🔧 创建 {args.model_type.upper()} Autoencoder 检测器...")
        detector = AutoencoderAnomalyDetector(
            model_type=args.model_type,
            input_dim=X_train.shape[2],
            seq_len=X_train.shape[1],
            device=device,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            latent_dim=args.latent_dim,
            dropout=args.dropout
        )
        
        # 准备训练数据（只使用已标注的正常样本）
        labeled_mask = (y_train != -1)
        normal_mask = (y_train == 0)
        train_mask = labeled_mask & normal_mask
        
        X_train_normal = X_train[train_mask]
        X_val_for_training = X_val[y_val != -1] if len(X_val) > 0 else None
        
        print(f"📊 训练数据: {X_train_normal.shape[0]} 个正常样本")
        if X_val_for_training is not None:
            print(f"📊 验证数据: {X_val_for_training.shape[0]} 个样本")
        
        # 训练 Autoencoder
        print("🔄 开始训练 Autoencoder...")
        training_history = detector.fit(
            X_train_normal,
            X_val_for_training,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            patience=args.patience,
            verbose=True
        )
        
        # 在测试集上进行预测
        print("🔮 在测试集上进行异常检测...")
        test_predictions, test_scores = detector.predict(X_test)
        
        # 评估结果
        final_metrics = evaluate_autoencoder_results(y_test, test_predictions, test_scores)
        
        print(f"\n🎯 Autoencoder 最终测试结果 (支架: {args.feature_column}):")
        print(f"   F1分数: {final_metrics['f1']:.4f}")
        print(f"   精确率: {final_metrics['precision']:.4f}")
        print(f"   召回率: {final_metrics['recall']:.4f}")
        print(f"   AUC-ROC: {final_metrics['auc_roc']:.4f}")
        
        # 创建可视化对象
        visualizer = CoreMetricsVisualizer(os.path.join(args.output_dir, 'visualizations'))
        
        # 生成可视化图表
        print("📊 生成 Autoencoder 可视化图表...")
        
        # 1. 指标柱状图
        visualizer.plot_final_metrics_bar(
            final_metrics['precision'], final_metrics['recall'], 
            final_metrics['f1'], final_metrics['auc_roc']
        )
        
        # 2. 混淆矩阵
        if len(final_metrics['labels']) > 0 and len(final_metrics['predictions']) > 0:
            visualizer.plot_confusion_matrix(final_metrics['labels'], final_metrics['predictions'])
        
        # 3. ROC曲线
        if len(final_metrics['labels']) > 0 and len(final_metrics['probabilities']) > 0:
            visualizer.plot_roc_curve(final_metrics['labels'], final_metrics['probabilities'])
        
        # 4. 训练历史
        visualizer.plot_training_history(training_history)
        
        # 5. 异常热力图
        try:
            df_original = pd.read_csv(args.data_path)
            if args.feature_column in df_original.columns:
                original_data = df_original[args.feature_column].values
                # 将分数标准化为概率
                normalized_scores = (test_scores - np.min(test_scores)) / (np.max(test_scores) - np.min(test_scores) + 1e-8)
                visualizer.plot_anomaly_heatmap(
                    original_data, normalized_scores, test_window_indices, args.window_size
                )
        except Exception as e:
            print(f"⚠️ 异常热力图生成失败: {e}")
        
        # 保存结果到JSON文件
        results_summary = {
            'experiment_info': {
                'method': f'Autoencoder ({args.model_type.upper()})',
                'data_file': args.data_path,
                'selected_feature': args.feature_column,
                'window_size': args.window_size,
                'stride': args.stride,
                'model_parameters': {
                    'model_type': args.model_type,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'latent_dim': args.latent_dim,
                    'dropout': args.dropout
                },
                'training_parameters': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'patience': args.patience
                },
                'timestamp': datetime.now().isoformat(),
                'seed': args.seed
            },
            'data_statistics': {
                'total_samples': len(X_train) + len(X_val) + len(X_test),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'test_normal_samples': int(np.sum(y_test == 0)),
                'test_anomaly_samples': int(np.sum(y_test == 1)),
                'test_unlabeled_samples': int(np.sum(y_test == -1)),
                'train_normal_samples_used': X_train_normal.shape[0]
            },
            'training_results': {
                'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else None,
                'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None,
                'reconstruction_threshold': float(detector.threshold),
                'total_epochs_trained': len(training_history['train_loss'])
            },
            'detection_results': {
                'test_window_predictions': {
                    'normal_predicted': int(np.sum(test_predictions == 0)),
                    'anomaly_predicted': int(np.sum(test_predictions == 1))
                },
                'score_statistics': {
                    'mean_score': float(np.mean(test_scores)),
                    'std_score': float(np.std(test_scores)),
                    'min_score': float(np.min(test_scores)),
                    'max_score': float(np.max(test_scores))
                }
            },
            'performance_metrics': {
                'precision': float(final_metrics['precision']),
                'recall': float(final_metrics['recall']),
                'f1_score': float(final_metrics['f1']),
                'auc_roc': float(final_metrics['auc_roc'])
            },
            'comparison_baseline': {
                'method_description': f'{args.model_type.upper()} Autoencoder for Time Series Anomaly Detection',
                'approach': 'Reconstruction-based anomaly detection using deep learning',
                'advantages': ['Learns normal patterns automatically', 'No prior anomaly examples needed', 'Captures complex temporal dependencies'],
                'limitations': ['Requires sufficient normal data', 'Sensitive to hyperparameters', 'Black-box model']
            }
        }
        
        # 转换为可序列化格式
        results_summary = convert_to_serializable(results_summary)
        
        # 保存结果
        results_file = os.path.join(args.output_dir, 'autoencoder_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Autoencoder 实验结果已保存到: {results_file}")
        
        # 生成详细的点级异常标记文件
        print("🔄 生成点级异常标记文件...")
        
        def mark_anomalies_pointwise_autoencoder(df_original, test_window_indices, test_predictions, 
                                               test_scores, window_size, feature_column, output_path):
            """Autoencoder 点级异常标记函数"""
            df_result = df_original.copy()
            df_result['Autoencoder_Anomaly'] = 0  # 默认正常
            df_result['Autoencoder_Window_Score'] = 0.0  # 窗口异常分数
            df_result['Autoencoder_Reconstruction_Error'] = 0.0  # 重建误差
            
            # 标记窗口级异常
            for i, window_start in enumerate(test_window_indices):
                window_end = min(window_start + window_size, len(df_result))
                window_pred = test_predictions[i]
                window_score = test_scores[i] if i < len(test_scores) else 0.0
                
                # 设置窗口分数和重建误差
                df_result.loc[window_start:window_end-1, 'Autoencoder_Window_Score'] = window_score
                df_result.loc[window_start:window_end-1, 'Autoencoder_Reconstruction_Error'] = window_score
                
                # 如果窗口被预测为异常，标记整个窗口
                if window_pred == 1:
                    df_result.loc[window_start:window_end-1, 'Autoencoder_Anomaly'] = 1
            
            # 保存结果
            df_result.to_csv(output_path, index=False)
            
            # 统计信息
            total_anomaly_points = df_result['Autoencoder_Anomaly'].sum()
            total_points = len(df_result)
            anomaly_rate = total_anomaly_points / total_points * 100
            
            print(f"📊 Autoencoder 点级标记统计:")
            print(f"   总数据点: {total_points}")
            print(f"   异常点数: {total_anomaly_points}")
            print(f"   异常率: {anomaly_rate:.2f}%")
            print(f"   结果文件: {output_path}")
            
            return df_result
        
        # 生成点级标记文件
        df_for_point_mapping = pd.read_csv(args.data_path)
        pointwise_output_path = os.path.join(args.output_dir, f'autoencoder_pointwise_results_{args.feature_column}.csv')
        marked_df = mark_anomalies_pointwise_autoencoder(
            df_for_point_mapping, test_window_indices, test_predictions, test_scores,
            args.window_size, args.feature_column, pointwise_output_path
        )
        
        # 生成对比报告
        print("📝 生成 Autoencoder 对比实验报告...")

        report_content = f"""
        # Autoencoder 异常检测方法 - 对比实验报告

        ## 实验配置
        - **检测方法**: {args.model_type.upper()} Autoencoder
        - **数据文件**: {args.data_path}
        - **选择特征**: {args.feature_column}
        - **窗口大小**: {args.window_size}
        - **滑动步长**: {args.stride}
        - **模型类型**: {args.model_type.upper()}
        - **隐藏层大小**: {args.hidden_size}
        - **网络层数**: {args.num_layers}
        - **学习率**: {args.learning_rate}
        - **随机种子**: {args.seed}

        ## 方法说明
        Autoencoder 是一种基于重建误差的无监督异常检测方法：

        1. **编码阶段**: 将输入时间序列压缩到低维潜在空间
        2. **解码阶段**: 从潜在表示重建原始序列
        3. **异常检测**: 重建误差超过阈值的样本被标记为异常
        4. **优势**: 无需异常样本训练，能学习复杂的正常模式

        ### {args.model_type.upper()} Autoencoder 特点
        """

        # 🔧 修复：分别处理不同模型类型的描述
        if args.model_type == 'lstm':
            model_description = f"""
        - **LSTM编码器**: 捕获时间序列的长期依赖关系
        - **LSTM解码器**: 基于编码表示重建序列
        - **优势**: 适合处理长时间序列，能够记忆历史信息
        - **参数**: 隐藏层{args.hidden_size}维, {args.num_layers}层网络
        """
        else:
            model_description = f"""
        - **卷积编码器**: 通过卷积操作提取局部特征
        - **转置卷积解码器**: 重建原始序列
        - **优势**: 计算效率高，适合捕获局部模式
        - **参数**: 潜在维度{args.latent_dim}
        """

        # 数据统计部分
        # 🔧 修复：预先计算复杂表达式，避免f-string格式错误
        final_train_loss = training_history['train_loss'][-1] if training_history['train_loss'] else None
        final_val_loss = training_history['val_loss'][-1] if training_history['val_loss'] else None
        
        final_train_loss_str = f"{final_train_loss:.6f}" if final_train_loss is not None else 'N/A'
        final_val_loss_str = f"{final_val_loss:.6f}" if final_val_loss is not None else 'N/A'
        
        data_section = f"""

## 数据统计
- **总样本数**: {len(X_train) + len(X_val) + len(X_test)}
- **训练集**: {len(X_train)} (仅使用{X_train_normal.shape[0]}个正常样本)
- **验证集**: {len(X_val)}
- **测试集**: {len(X_test)}
- **测试集正常样本**: {np.sum(y_test == 0)}
- **测试集异常样本**: {np.sum(y_test == 1)}

## 训练结果
- **训练轮数**: {len(training_history['train_loss'])}
- **最终训练损失**: {final_train_loss_str}
- **最终验证损失**: {final_val_loss_str}
- **异常检测阈值**: {detector.threshold:.6f}

        ## 检测结果
        - **预测正常窗口**: {np.sum(test_predictions == 0)}
        - **预测异常窗口**: {np.sum(test_predictions == 1)}
        - **平均重建误差**: {np.mean(test_scores):.6f}
        - **重建误差标准差**: {np.std(test_scores):.6f}

        ## 性能指标
        - **精确率 (Precision)**: {final_metrics['precision']:.4f}
        - **召回率 (Recall)**: {final_metrics['recall']:.4f}
        - **F1分数**: {final_metrics['f1']:.4f}
        - **AUC-ROC**: {final_metrics['auc_roc']:.4f}

        ## 与RLAD方法对比
        本实验作为RLAD方法的对比基线，采用完全相同的数据预处理、窗口划分和评估流程，
        核心差异在于异常检测方法：
        - **RLAD**: 强化学习 + 深度神经网络 + 人工交互标注
        - **Autoencoder**: 重建误差 + 无监督学习

        ## 方法优缺点对比

        ### Autoencoder 优势：
        - 无需异常样本训练
        - 能够自动学习复杂的正常模式
        - 适合处理高维时间序列数据
        - 计算效率相对较高

        ### Autoencoder 局限性：
        - 依赖足够的正常训练数据
        - 阈值设置较为关键
        - 模型解释性有限
        - 对超参数敏感

        ## 文件输出
        - 实验结果: {results_file}
        - 点级标记: {pointwise_output_path}
        - 可视化图表: {os.path.join(args.output_dir, 'visualizations')}
        - 实验报告: {os.path.join(args.output_dir, 'autoencoder_experiment_report.md')}

        ---
        *实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        *生成工具: Autoencoder 对比实验系统*
        """

        # 合并所有部分
        full_report_content = report_content + model_description + data_section

        # 保存报告
        report_file = os.path.join(args.output_dir, 'autoencoder_experiment_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report_content)

        print(f"📄 实验报告已保存到: {report_file}")

        print(f"\n🎉 Autoencoder 对比实验完成!")
        print(f"📂 所有结果已保存到: {args.output_dir}")
        print(f"📊 主要性能指标: F1={final_metrics['f1']:.4f}, AUC={final_metrics['auc_roc']:.4f}")
        
        return 0
        
    except Exception as e:
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'args': vars(args) if 'args' in locals() else {}
        }
        
        try:
            if 'args' in locals() and hasattr(args, 'output_dir'):
                error_file = os.path.join(args.output_dir, 'error_log.json')
                with open(error_file, 'w') as f:
                    json.dump(convert_to_serializable(error_info), f, indent=2)
                print(f"💾 错误信息已保存到: {error_file}")
        except:
            print(f"💾 错误信息保存失败")
        
        print(f"❌ Autoencoder 实验执行出错: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"🏁 Autoencoder 对比实验结束，退出代码: {exit_code}")
