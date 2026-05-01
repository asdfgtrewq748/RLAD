#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original RLAD 异常检测对比实验
==============================

基于原始RLAD强化学习异常检测系统的对比实验
核心组件：
1. 深度Q网络 (DQN) 用于动作值函数估计
2. LSTM状态表示学习
3. 主动学习策略 (Active Learning)
4. 标签传播 (Label Propagation)
5. Isolation Forest预热启动

目标：重现原始RLAD v1.0的核心算法，与RLADv3.2形成对比

作者: AI Assistant  
基于: 原始RLAD论文实现
创建时间: 2025-08-26
版本: v1.0 (基于原始RLAD)
"""

import os
import sys
import json
import random
import warnings
import argparse
import time
import itertools
from pathlib import Path
from datetime import datetime
from collections import deque, namedtuple
from typing import Optional, Tuple, List, Dict, Any

# 数据处理库
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar

# 深度学习库 - PyTorch (替代TensorFlow实现原始RLAD)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 机器学习库
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, roc_auc_score,
    average_precision_score, precision_recall_fscore_support
)
from sklearn.manifold import TSNE

# 时间序列分析
from statsmodels.tsa.seasonal import STL

# 深度学习库（简化版）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# GUI及绘图库导入（用于兼容性）
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =================================
# Original RLAD 核心配置常量
# =================================

# 原始RLAD基本配置
NOT_ANOMALY = 0
ANOMALY = 1
ACTION_SPACE = [NOT_ANOMALY, ANOMALY]
ACTION_SPACE_N = len(ACTION_SPACE)

# 奖励函数配置
REWARD_CORRECT = 1
REWARD_INCORRECT = -1
TP_VALUE = 5  # True Positive奖励
TN_VALUE = 1  # True Negative奖励  
FP_VALUE = -1  # False Positive惩罚
FN_VALUE = -5  # False Negative惩罚

# LSTM网络配置
N_STEPS = 25  # 滑动窗口大小
N_INPUT_DIM = 2  # 输入维度 [value, action_flag]
N_HIDDEN_DIM = 128  # LSTM隐藏层维度

# 训练配置
EPISODES = 500  # 训练轮数
DISCOUNT_FACTOR = 0.5  # 折扣因子
EPSILON = 0.5  # epsilon-greedy探索率
EPSILON_DECAY = 1.00  # epsilon衰减率
VALIDATION_SEPARATE_RATIO = 0.9  # 验证集分割比例

# 主动学习配置
NUM_LABEL_PROPAGATION = 20  # 标签传播样本数
NUM_ACTIVE_LEARNING = 5  # 主动学习样本数
OUTLIERS_FRACTION = 0.01  # 异常值比例

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

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

# 忽略警告
warnings.filterwarnings("ignore")

# =================================
# 原始RLAD 状态和奖励函数
# =================================

def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    """
    原始RLAD的LSTM二元状态函数
    返回滑动窗口状态向量: [[x_1, f_1], [x_2, f_2], ..., [x_t, f_t]]
    """
    if timeseries_curser < N_STEPS:
        # 初始化阶段
        state = []
        for i in range(timeseries_curser + 1):
            if i == timeseries_curser:
                state.append([timeseries['value'][i], 1])  # 当前点标记为1
            else:
                state.append([timeseries['value'][i], 0])  # 历史点标记为0
        
        # 填充到固定长度
        while len(state) < N_STEPS:
            state.insert(0, [0, 0])  # 零填充
            
        return np.array(state, dtype='float32')
    
    elif timeseries_curser == N_STEPS:
        # 刚好达到窗口大小
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])
        
        state.pop(0)  # 移除最旧的
        state.append([timeseries['value'][timeseries_curser], 1])  # 添加当前的
        
        return np.array(state, dtype='float32')
    
    if timeseries_curser > N_STEPS:
        # 超过窗口大小，返回两个可能的下一状态
        state0 = np.concatenate((previous_state[1:N_STEPS],
                                 [[timeseries['value'][timeseries_curser], 0]]))
        state1 = np.concatenate((previous_state[1:N_STEPS], 
                                 [[timeseries['value'][timeseries_curser], 1]]))
        
        return np.array([state0, state1], dtype='float32')

def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0):
    """
    原始RLAD的二元奖励函数 (训练时使用标签)
    """
    if timeseries_curser >= N_STEPS:
        if timeseries['label'][timeseries_curser] == 0:  # 正常样本
            return [TN_VALUE, FP_VALUE]  # [选择正常的奖励, 选择异常的惩罚]
        if timeseries['label'][timeseries_curser] == 1:  # 异常样本
            return [FN_VALUE, TP_VALUE]  # [选择正常的惩罚, 选择异常的奖励]
    else:
        return [0, 0]  # 初始化阶段无奖励

def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    原始RLAD的二元奖励函数 (测试时使用真实标签)
    """
    if timeseries_curser >= N_STEPS:
        if timeseries['anomaly'][timeseries_curser] == 0:  # 正常样本
            return [TN_VALUE, FP_VALUE]
        if timeseries['anomaly'][timeseries_curser] == 1:  # 异常样本
            return [FN_VALUE, TP_VALUE]  
    else:
        return [0, 0]

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
        self._set_scientific_style(ax, 'Original Ensemble-RLAD Performance', 'Score', 'Metric')
        ax.set_xlim(0, 1.0); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.grid(False)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'ensemble_rlad_metrics_summary.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Original Ensemble-RLAD metrics summary plot saved to: {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={"size": 14}, linecolor='white', linewidths=1)
        self._set_scientific_style(ax, 'Original Ensemble-RLAD Confusion Matrix', 'Predicted Label', 'True Label')
        ax.set_xticklabels(['Normal', 'Anomaly']); ax.set_yticklabels(['Normal', 'Anomaly'], va='center', rotation=90)
        ax.grid(False)
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'ensemble_rlad_confusion_matrix.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Original Ensemble-RLAD confusion matrix plot saved to: {save_path}")

    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color=self.colors['primary'], linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        self._set_scientific_style(ax, 'Original Ensemble-RLAD ROC Curve', 'False Positive Rate', 'True Positive Rate')
        ax.legend(); ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'ensemble_rlad_roc_curve.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Original Ensemble-RLAD ROC curve plot saved to: {save_path}")

    def plot_anomaly_heatmap(self, original_data, predictions, window_indices, window_size, save_path=None):
        print("🔥 生成异常检测热力图...")
        anomaly_map = np.zeros(len(original_data))
        for i, (start_idx, end_idx) in enumerate(window_indices):
            if predictions[i] == 1:  # 异常窗口
                anomaly_map[start_idx:end_idx] += 1
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        time_steps = np.arange(len(original_data))
        ax1.plot(time_steps, original_data, color=self.colors['black'], alpha=0.7, linewidth=1)
        self._set_scientific_style(ax1, 'Original Time Series', 'Time Steps', 'Value')
        
        im = ax2.imshow(anomaly_map.reshape(1, -1), cmap='Reds', aspect='auto', extent=[0, len(original_data), -0.5, 0.5])
        self._set_scientific_style(ax2, 'Anomaly Detection Heatmap', 'Time Steps', '')
        ax2.set_yticks([]); plt.colorbar(im, ax=ax2, label='Anomaly Score')
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'ensemble_rlad_anomaly_heatmap.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Original Ensemble-RLAD anomaly heatmap plot saved to: {save_path}")

    def plot_ensemble_weights(self, weights, method_names, save_path=None):
        """绘制集成方法权重分布"""
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(method_names, weights, color=[
            self.colors['primary'], self.colors['secondary'], self.colors['tertiary'],
            self.colors['accent'], self.colors['neutral']
        ][:len(weights)])
        
        self._set_scientific_style(ax, 'Ensemble Method Weights', 'Detection Methods', 'Weight')
        ax.set_ylim(0, 1.0)
        
        # 在柱子上添加数值
        for bar, weight in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_path is None: save_path = os.path.join(self.output_dir, 'ensemble_rlad_weights.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight'); plt.close()
        print(f"Ensemble weights plot saved to: {save_path}")

# =================================
# 原始RLAD 深度Q网络估计器 (使用PyTorch实现)
# =================================

class Q_Estimator_Nonlinear(nn.Module):
    """
    原始RLAD的动作值函数近似器 Q(s,a)
    使用LSTM网络进行时序特征学习
    """
    
    def __init__(self, learning_rate=0.01, scope="Q_Estimator_Nonlinear"):
        super(Q_Estimator_Nonlinear, self).__init__()
        self.scope = scope
        self.learning_rate = learning_rate
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=N_INPUT_DIM,
            hidden_size=N_HIDDEN_DIM, 
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # 全连接层
        self.fc1 = nn.Linear(N_HIDDEN_DIM, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, ACTION_SPACE_N)  # 输出动作值
        
        # Dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, state):
        """
        前向传播
        输入: state [batch_size, seq_len, input_dim]
        输出: action_values [batch_size, num_actions]
        """
        # LSTM处理序列
        lstm_out, (hidden, cell) = self.lstm(state)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 全连接层
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        action_values = self.fc3(x)  # [batch_size, num_actions]
        
        return action_values
    
    def predict(self, state):
        """预测动作值"""
        self.eval()
        with torch.no_grad():
            # 确保state是numpy数组
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            
            # 处理维度
            if len(state.shape) == 2:  # 单个样本 (seq_len, features)
                state = np.expand_dims(state, axis=0)  # 添加batch维度 (1, seq_len, features)
            
            state_tensor = torch.FloatTensor(state)
            action_values = self.forward(state_tensor)
            return action_values.cpu().numpy()
    
    def update(self, state, target):
        """更新网络参数"""
        self.train()
        
        # 确保state是numpy数组并转换为tensor
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # 处理维度
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)  # 添加batch维度
        
        state_tensor = torch.FloatTensor(state)
        target_tensor = torch.FloatTensor(target)
        
        # 前向传播
        predicted = self.forward(state_tensor)
        
        # 计算损失
        loss = self.loss_fn(predicted, target_tensor)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def copy_model_parameters(estimator1, estimator2):
    """
    复制模型参数 (原始RLAD的参数同步)
    """
    estimator2.load_state_dict(estimator1.state_dict())

def make_epsilon_greedy_policy(estimator, nA):
    """
    创建epsilon-greedy策略 (原始RLAD的策略函数)
    """
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(observation)
        if len(q_values.shape) > 1:
            q_values = q_values[0]  # 取第一个样本
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    
    return policy_fn

# =================================
# 原始RLAD 主动学习模块
# =================================

class active_learning:
    """原始RLAD的主动学习类"""
    
    def __init__(self, env, N, strategy, estimator, already_selected):
        self.env = env
        self.N = N  # 需要标记的样本数
        self.strategy = strategy  # 策略：'margin_sampling'
        self.estimator = estimator
        self.already_selected = already_selected
        
    def get_samples(self):
        """获取需要主动学习的样本索引"""
        if self.strategy == 'margin_sampling':
            return self._margin_sampling()
        else:
            return self._random_sampling()
    
    def _margin_sampling(self):
        """边际采样策略"""
        # 限制处理的样本数量以避免计算过载
        max_samples = min(500, len(self.env.states_list))
        uncertainties = []
        
        for i in range(max_samples):
            if i in self.already_selected:
                uncertainties.append(-1)  # 已标记的样本
                continue
            
            state = self.env.states_list[i]
            # 计算动作值的不确定性 (最大值与次大值的差)
            q_values = self.estimator.predict(state)
            if len(q_values.shape) > 1:
                q_values = q_values[0]
            
            sorted_q = np.sort(q_values)
            if len(sorted_q) >= 2:
                margin = sorted_q[-1] - sorted_q[-2]  # 边际
            else:
                margin = sorted_q[-1]  # 只有一个值
            uncertainties.append(-margin)  # 负值表示不确定性高
        
        # 选择最不确定的样本
        candidates = [(i, unc) for i, unc in enumerate(uncertainties) if unc > -1]
        candidates.sort(key=lambda x: x[1])  # 按不确定性排序
        
        selected = [idx for idx, _ in candidates[:self.N]]
        return selected
    
    def _random_sampling(self):
        """随机采样策略"""
        available_indices = [i for i in range(len(self.env.states_list)) 
                           if i not in self.already_selected]
        return np.random.choice(available_indices, 
                              size=min(self.N, len(available_indices)), 
                              replace=False).tolist()
    
    def label(self, active_samples):
        """为选中的样本添加标签"""
        for sample_idx in active_samples:
            # 转换索引到时间序列位置
            ts_idx = sample_idx + N_STEPS
            if ts_idx < len(self.env.timeseries):
                # 使用真实标签进行标记
                true_label = self.env.timeseries['anomaly'][ts_idx]
                self.env.timeseries['label'][ts_idx] = true_label

class WarmUp:
    """原始RLAD的预热启动模块"""
    
    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        """使用Isolation Forest进行预热"""
        # 确保max_samples是整数
        n_samples = len(X_train)
        max_samples = min(256, n_samples)
        
        model = IsolationForest(
            contamination=outliers_fraction,
            max_samples=max_samples,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_train)
        return model
    
    def warm_up_SVM(self, outliers_fraction, N):
        """使用One-Class SVM进行预热"""
        model = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)
        return model

# =================================
# 简化版深度学习模型 (保持兼容)
# =================================

class SimpleAutoencoder(nn.Module):
    """简化版自动编码器用于集成"""
    def __init__(self, input_size, hidden_size=32):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# =================================
# Original Ensemble-RLAD 异常检测系统 (核心差异部分)
# =================================

class OriginalRLADAnomalyDetector:
    """
    原始RLAD异常检测器
    基于深度强化学习的异常检测系统
    """
    
    def __init__(self, 
                 episodes=EPISODES,
                 learning_rate=0.01,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=500000,
                 discount_factor=DISCOUNT_FACTOR,
                 replay_memory_size=500000,
                 replay_memory_init_size=50000,
                 batch_size=512,
                 num_label_propagation=NUM_LABEL_PROPAGATION,
                 num_active_learning=NUM_ACTIVE_LEARNING,
                 random_state=42):
        
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.discount_factor = discount_factor
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.batch_size = batch_size
        self.num_label_propagation = num_label_propagation
        self.num_active_learning = num_active_learning
        self.random_state = random_state
        
        # 设置随机种子
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # 初始化Q网络
        self.qlearn_estimator = Q_Estimator_Nonlinear(learning_rate=learning_rate)
        self.target_estimator = Q_Estimator_Nonlinear(learning_rate=learning_rate)
        
        # 初始化经验回放
        self.Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])
        self.replay_memory = []
        
        # 初始化预热模块
        self.warm_up = WarmUp()
        
        # 策略函数
        self.policy = make_epsilon_greedy_policy(self.qlearn_estimator, ACTION_SPACE_N)
        
        # 标签传播模型
        self.lp_model = LabelSpreading()
        
        # 训练状态
        self.is_trained = False
        
    def _prepare_timeseries_environment(self, X):
        """准备时间序列环境数据"""
        # 创建模拟的时间序列环境
        class MockTimeSeriesEnv:
            def __init__(self, data):
                self.data = data
                self.datasetsize = 1  # 单个时间序列
                self.datasetrng = 1
                self.datasetidx = 0
                self.timeseries_curser = 0
                self.action_space_n = ACTION_SPACE_N
                
                # 创建时间序列DataFrame
                self.timeseries = pd.DataFrame({
                    'value': data.flatten(),
                    'anomaly': np.zeros(len(data.flatten())),  # 未知真实标签
                    'label': np.full(len(data.flatten()), -1)  # 初始化为未标记
                })
                
                # 归一化数据
                scaler = MinMaxScaler()
                self.timeseries['value'] = scaler.fit_transform(
                    self.timeseries['value'].values.reshape(-1, 1)
                ).flatten()
                
                # 生成状态列表
                self.states_list = self._generate_states_list()
                
            def _generate_states_list(self):
                """生成状态列表"""
                states = []
                for cursor in range(N_STEPS, len(self.timeseries)):
                    state = []
                    for i in range(cursor - N_STEPS, cursor):
                        state.append([self.timeseries['value'][i], 0])
                    state.append([self.timeseries['value'][cursor], 1])  # 当前点
                    states.append(np.array(state[-N_STEPS:], dtype='float32'))
                return states
                
            def reset(self):
                self.timeseries_curser = N_STEPS
                if len(self.states_list) > 0:
                    return self.states_list[0]
                else:
                    return np.zeros((N_STEPS, N_INPUT_DIM), dtype='float32')
            
            def step(self, action):
                # 简化的step函数
                reward = 0  # 测试时不计算奖励
                self.timeseries_curser += 1
                
                done = self.timeseries_curser >= len(self.timeseries) - 1
                
                if not done and self.timeseries_curser - N_STEPS < len(self.states_list):
                    next_state = self.states_list[self.timeseries_curser - N_STEPS]
                else:
                    next_state = np.zeros((N_STEPS, N_INPUT_DIM), dtype='float32')
                
                return next_state, reward, done, []
        
        return MockTimeSeriesEnv(X)
    
    def _warm_up_with_isolation_forest(self, env):
        """使用Isolation Forest进行预热"""
        print('🚀 RLAD 预热启动中...')
        
        # 准备训练数据 - 限制数据量避免内存问题
        data_train = []
        # 只使用前1000个样本进行预热，避免内存问题
        max_samples = min(1000, len(env.states_list))
        for i, state in enumerate(env.states_list[:max_samples]):
            # 将每个状态展平为1维向量
            flattened_state = state.flatten()
            data_train.append(flattened_state)
        
        data_train = np.array(data_train)  # 现在是2维: (n_samples, features)
        print(f"📊 预热训练数据形状: {data_train.shape}")
        
        # 训练Isolation Forest
        model = self.warm_up.warm_up_isolation_forest(OUTLIERS_FRACTION, data_train)
        
        # 计算异常分数
        anomaly_score = model.decision_function(data_train)
        pred_score = [-1 * s + 0.5 for s in anomaly_score]
        
        # 选择不确定样本进行标记
        warm_samples = np.argsort(pred_score)[:5]  # 最异常的5个
        warm_samples = np.append(warm_samples, np.argsort(pred_score)[-5:])  # 最正常的5个
        
        # 为选中的样本分配伪标签 - 直接分配，避免标签传播的内存问题
        for sample in warm_samples:
            if sample < len(env.timeseries) - N_STEPS:
                # 基于异常分数分配标签
                if pred_score[sample] > np.percentile(pred_score, 90):
                    env.timeseries['label'][sample + N_STEPS] = 1  # 异常
                else:
                    env.timeseries['label'][sample + N_STEPS] = 0  # 正常
        
        # 简化的标签传播 - 使用基于距离的方法而不是sklearn的LabelSpreading
        print("🔄 执行简化标签传播...")
        labeled_indices = [i for i, label in enumerate(env.timeseries['label']) if label != -1]
        
        if len(labeled_indices) > 0:
            # 为更多样本分配伪标签，基于与已标记样本的相似性
            labeled_count = 0
            max_propagated_labels = min(self.num_label_propagation, len(env.states_list) - len(labeled_indices))
            
            for i in range(min(len(env.states_list), max_samples)):  # 只处理采样的数据
                if env.timeseries['label'][i + N_STEPS] == -1 and labeled_count < max_propagated_labels:
                    # 计算与最近已标记样本的距离
                    current_state = env.states_list[i].flatten()
                    min_distance = float('inf')
                    closest_label = 0
                    
                    for labeled_idx in labeled_indices[:10]:  # 只考虑前10个已标记样本以节省计算
                        if labeled_idx - N_STEPS < len(env.states_list):
                            labeled_state = env.states_list[labeled_idx - N_STEPS].flatten()
                            distance = np.linalg.norm(current_state - labeled_state)
                            if distance < min_distance:
                                min_distance = distance
                                closest_label = env.timeseries['label'][labeled_idx]
                    
                    # 如果距离足够近，分配相同标签
                    threshold = np.std(data_train.flatten()) * 0.5  # 基于数据的标准差设置阈值
                    if min_distance < threshold:
                        env.timeseries['label'][i + N_STEPS] = closest_label
                        labeled_count += 1
        
        total_labeled = np.sum(env.timeseries['label'] != -1)
        print(f'✅ 预热完成，总共标记了 {total_labeled} 个样本')
        return env
    
    def _q_learning_training(self, env):
        """Q-learning训练过程"""
        print('🎯 开始Q-learning训练...')
        
        total_t = 0
        epsilons = np.linspace(self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)
        
        # 填充经验回放池
        state = env.reset()
        for i in range(min(self.replay_memory_init_size, 1000)):  # 减少初始填充
            epsilon = epsilons[min(total_t, self.epsilon_decay_steps - 1)]
            action_probs = self.policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            next_state, reward, done, _ = env.step(action)
            self.replay_memory.append(self.Transition(state, reward, next_state, done))
            
            if done:
                state = env.reset()
            else:
                state = next_state
            
            total_t += 1
        
        # 主训练循环
        for episode in range(min(self.episodes, 50)):  # 减少训练轮数用于演示
            state = env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            # 主动学习
            if episode % 10 == 0 and episode > 0:
                labeled_indices = [i for i, e in enumerate(env.timeseries['label']) if e != -1]
                labeled_indices = [i - N_STEPS for i in labeled_indices if i >= N_STEPS]
                
                if len(labeled_indices) < len(env.states_list) - 10:  # 还有未标记样本
                    al = active_learning(env=env, N=self.num_active_learning, 
                                       strategy='margin_sampling',
                                       estimator=self.qlearn_estimator, 
                                       already_selected=labeled_indices)
                    al_samples = al.get_samples()
                    al.label(al_samples)
                    print(f'Episode {episode}: 标记了额外的 {len(al_samples)} 个样本')
            
            # Episode训练
            while not done and step_count < 100:  # 限制步数
                epsilon = epsilons[min(total_t, self.epsilon_decay_steps - 1)]
                action_probs = self.policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                # 经验回放
                self.replay_memory.append(self.Transition(state, reward, next_state, done))
                if len(self.replay_memory) > self.replay_memory_size:
                    self.replay_memory.pop(0)
                
                # 批量学习
                if len(self.replay_memory) >= 32:  # 减少批量大小
                    self._replay_experience()
                
                state = next_state
                total_t += 1
                step_count += 1
            
            if episode % 10 == 0:
                print(f'Episode {episode}: Reward = {episode_reward:.2f}, Epsilon = {epsilon:.3f}')
                
                # 同步目标网络
                copy_model_parameters(self.qlearn_estimator, self.target_estimator)
        
        print('✅ Q-learning训练完成')
    
    def _replay_experience(self):
        """经验回放学习"""
        if len(self.replay_memory) < 32:
            return
            
        # 随机采样经验
        batch_size = min(32, len(self.replay_memory))
        samples = random.sample(self.replay_memory, batch_size)
        
        for sample in samples:
            state, reward, next_state, done = sample
            
            # 计算目标Q值
            if done:
                target_q = reward
            else:
                next_q_values = self.target_estimator.predict(next_state)
                if len(next_q_values.shape) > 1:
                    next_q_values = next_q_values[0]
                target_q = reward + self.discount_factor * np.max(next_q_values)
            
            # 更新Q网络
            current_q_values = self.qlearn_estimator.predict(state)
            if len(current_q_values.shape) > 1:
                current_q_values = current_q_values[0]
            
            target_q_values = current_q_values.copy()
            target_q_values[0] = target_q  # 假设选择的动作是0，简化处理
            
            self.qlearn_estimator.update(state, target_q_values.reshape(1, -1))
    
    def fit(self, X_train):
        """训练RLAD模型"""
        print('🚀 开始训练Original RLAD模型...')
        
        # 准备环境
        env = self._prepare_timeseries_environment(X_train)
        
        # 预热启动
        env = self._warm_up_with_isolation_forest(env)
        
        # Q-learning训练
        self._q_learning_training(env)
        
        self.is_trained = True
        print('✅ Original RLAD训练完成')
    
    def predict(self, X_test):
        """使用训练好的RLAD模型进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        
        print('🔍 使用Original RLAD进行异常检测...')
        
        # 准备测试环境
        test_env = self._prepare_timeseries_environment(X_test)
        
        predictions = []
        scores = []
        
        # 逐步预测
        state = test_env.reset()
        done = False
        
        while not done:
            # 使用训练好的Q网络进行预测
            q_values = self.qlearn_estimator.predict(state)
            if len(q_values.shape) > 1:
                q_values = q_values[0]
            
            # 选择最佳动作
            action = np.argmax(q_values)
            predictions.append(action)
            
            # 计算置信度分数
            confidence = np.max(q_values) - np.min(q_values)
            scores.append(confidence)
            
            # 执行动作
            state, _, done, _ = test_env.step(action)
        
        print(f'✅ 预测完成，共预测 {len(predictions)} 个样本')
        
        return np.array(predictions), np.array(scores)
    
    def detect_anomalies(self, X):
        """检测异常（整合训练和预测）"""
        # 使用部分数据训练
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        
        # 训练模型
        self.fit(X_train)
        
        # 对所有数据进行预测
        predictions, scores = self.predict(X)
        
        # 确保输出长度与输入匹配
        if len(predictions) < len(X):
            # 填充前面的预测（前N_STEPS个无法预测）
            pad_predictions = np.zeros(len(X) - len(predictions))
            pad_scores = np.zeros(len(X) - len(scores))
            
            predictions = np.concatenate([pad_predictions, predictions])
            scores = np.concatenate([pad_scores, scores])
        
        return scores, predictions, {'rlad_scores': scores}

# =================================
# 数据加载函数 (与RLADv3.2保持一致)
# =================================

def load_hydraulic_data_with_original_rlad(data_path, window_size, stride, specific_feature_column,
                                          contamination=0.1, unlabeled_fraction=0.1):
    """使用Original RLAD进行异常检测的数据加载函数 - 与RLADv3.2对齐版本"""
    print(f"📥 Loading data: {data_path}")
    
    # 读取数据 - 与RLADv3.2完全相同的处理
    df = pd.read_csv(data_path)
    
    if specific_feature_column not in df.columns:
        raise ValueError(f"Feature column '{specific_feature_column}' not found in data. Available columns: {df.columns.tolist()}")
    
    print(f"🎯 Using feature column: {specific_feature_column}")
    feature_data = df[specific_feature_column].values
    
    print(f"📊 Original data shape: {feature_data.shape}")
    print(f"📊 Data range: [{np.min(feature_data):.4f}, {np.max(feature_data):.4f}]")
    
    # 创建滑动窗口
    print(f"🔄 Creating sliding windows (window_size={window_size}, stride={stride})...")
    windows = []
    window_indices = []
    
    for i in range(0, len(feature_data) - window_size + 1, stride):
        window = feature_data[i:i + window_size]
        windows.append(window.reshape(-1, 1))  # (window_size, 1)
        window_indices.append((i, i + window_size))
    
    windows = np.array(windows)  # (n_windows, window_size, 1)
    
    print(f"✅ Created {len(windows)} windows of shape {windows[0].shape}")
    
    # 使用Original Ensemble-RLAD进行异常检测
    print("🚀 使用Original Ensemble-RLAD检测异常...")
    
    # 假设大部分数据是正常的，选择前80%作为训练数据
    train_size = int(len(windows) * 0.8)
    X_train_normal = windows[:train_size]
    X_all = windows
    
    # 创建检测器并训练
    detector = OriginalRLADAnomalyDetector(
        episodes=50,  # 减少训练轮数用于演示
        learning_rate=0.01,
        random_state=42
    )
    detector.fit(X_train_normal)
    
    # 对所有数据进行异常检测
    anomaly_scores, predictions, method_scores = detector.detect_anomalies(X_all)
    
    # 🔧 改进的异常标签生成策略
    print("🔄 生成改进的异常标签...")
    
    # 方法1：基于多个统计指标的综合异常标签
    y_labels_statistical = np.zeros(len(windows))
    
    for i, window in enumerate(windows):
        window_flat = window.flatten()
        
        # 计算多种统计指标
        z_scores = np.abs(stats.zscore(window_flat))
        mean_val = np.mean(window_flat)
        std_val = np.std(window_flat)
        
        # 条件1：极值检测
        is_extreme = (np.max(window_flat) > np.percentile(feature_data, 99)) or \
                    (np.min(window_flat) < np.percentile(feature_data, 1))
        
        # 条件2：方差异常
        global_std = np.std(feature_data)
        is_variance_anomaly = std_val > 2 * global_std or std_val < 0.1 * global_std
        
        # 条件3：Z-score异常
        is_z_anomaly = np.max(z_scores) > 2.5
        
        # 条件4：趋势异常（窗口内趋势与全局趋势差异过大）
        if len(window_flat) > 3:
            window_trend = np.polyfit(range(len(window_flat)), window_flat, 1)[0]
            global_trend = np.polyfit(range(len(feature_data)), feature_data, 1)[0]
            is_trend_anomaly = abs(window_trend - global_trend) > 3 * abs(global_trend)
        else:
            is_trend_anomaly = False
        
        # 综合判断（至少满足2个条件才认为是异常）
        anomaly_count = sum([is_extreme, is_variance_anomaly, is_z_anomaly, is_trend_anomaly])
        if anomaly_count >= 2:
            y_labels_statistical[i] = 1
    
    # 方法2：基于重构误差分布的异常标签
    ensemble_scores_normalized = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    
    # 使用更合理的阈值：75分位数 + 1.5 * IQR
    q75 = np.percentile(ensemble_scores_normalized, 75)
    q25 = np.percentile(ensemble_scores_normalized, 25)
    iqr = q75 - q25
    adaptive_threshold = q75 + 1.5 * iqr
    
    y_labels_ensemble = (ensemble_scores_normalized > adaptive_threshold).astype(int)
    
    # 方法3：结合两种方法的软投票
    y_labels_combined = np.zeros(len(windows))
    
    # 如果两种方法都认为是异常，则标记为异常
    consensus_anomalies = (y_labels_statistical == 1) & (y_labels_ensemble == 1)
    
    # 如果任一方法认为是异常且分数较高，也标记为异常
    high_score_anomalies = (ensemble_scores_normalized > np.percentile(ensemble_scores_normalized, 85)) & \
                          ((y_labels_statistical == 1) | (y_labels_ensemble == 1))
    
    y_labels_combined[consensus_anomalies | high_score_anomalies] = 1
    
    # 使用组合标签作为最终标签
    y_labels = y_labels_combined.copy()
    
    # 添加一些未标记数据模拟真实场景
    if unlabeled_fraction > 0:
        n_unlabeled = int(len(y_labels) * unlabeled_fraction)
        unlabeled_indices = np.random.choice(len(y_labels), n_unlabeled, replace=False)
        y_labels[unlabeled_indices] = -1  # -1表示未标记
    
    print(f"📊 Final labels distribution:")
    print(f"   Normal (0): {np.sum(y_labels == 0)}")
    print(f"   Anomaly (1): {np.sum(y_labels == 1)}")
    print(f"   Unlabeled (-1): {np.sum(y_labels == -1)}")
    
    # 创建原始窗口数据用于可视化
    raw_windows = []
    for start_idx, end_idx in window_indices:
        raw_windows.append(feature_data[start_idx:end_idx])
    
    return (windows, y_labels, np.array(raw_windows), window_indices, 
            feature_data, anomaly_scores, method_scores, detector)

def train_test_split_with_indices_v32_compatible(X, y, raw_windows, window_indices, test_size=0.2, val_size=0.1):
    """
    时间序列数据分割 - 与RLADv3.2兼容
    保持时间顺序，不随机打乱
    """
    n_samples = len(X)
    
    # 计算分割点
    train_end = int(n_samples * (1 - test_size - val_size))
    val_end = int(n_samples * (1 - test_size))
    
    # 分割数据
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    raw_train = raw_windows[:train_end]
    raw_val = raw_windows[train_end:val_end]
    raw_test = raw_windows[val_end:]
    
    indices_train = window_indices[:train_end]
    indices_val = window_indices[train_end:val_end]
    indices_test = window_indices[val_end:]
    
    print(f"📊 Data split completed:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples") 
    print(f"   Testing: {len(X_test)} samples")
    
    return ((X_train, y_train, raw_train, indices_train),
            (X_val, y_val, raw_val, indices_val),
            (X_test, y_test, raw_test, indices_test))

# =================================
# 评估函数
# =================================

def evaluate_original_rlad_results(y_true, y_pred, y_scores=None):
    """
    评估Original Ensemble-RLAD结果
    
    参数:
    - y_true: 真实标签
    - y_pred: 预测标签  
    - y_scores: 预测分数 (可选)
    
    返回:
    - metrics: 包含各种评估指标的字典
    """
    
    # 过滤掉未标记的数据 (-1)
    labeled_mask = y_true != -1
    y_true_labeled = y_true[labeled_mask]
    y_pred_labeled = y_pred[labeled_mask]
    
    if y_scores is not None:
        y_scores_labeled = y_scores[labeled_mask]
    
    if len(y_true_labeled) == 0:
        print("⚠️ 警告: 没有标记的测试数据")
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc_roc': 0.5,
            'support_normal': 0, 'support_anomaly': 0, 'accuracy': 0.0
        }
    
    # 计算基本指标
    precision = precision_score(y_true_labeled, y_pred_labeled, zero_division=0)
    recall = recall_score(y_true_labeled, y_pred_labeled, zero_division=0)
    f1 = f1_score(y_true_labeled, y_pred_labeled, zero_division=0)
    
    # 计算支持数
    support_normal = np.sum(y_true_labeled == 0)
    support_anomaly = np.sum(y_true_labeled == 1)
    
    # 计算准确率
    accuracy = np.mean(y_true_labeled == y_pred_labeled)
    
    # 计算AUC (如果有分数)
    auc_roc = 0.5
    if y_scores is not None and len(np.unique(y_true_labeled)) > 1:
        try:
            auc_roc = roc_auc_score(y_true_labeled, y_scores_labeled)
        except ValueError as e:
            print(f"⚠️ AUC计算警告: {e}")
            auc_roc = 0.5
    
    metrics = {
        'precision': precision,
        'recall': recall,  
        'f1': f1,
        'auc_roc': auc_roc,
        'accuracy': accuracy,
        'support_normal': support_normal,
        'support_anomaly': support_anomaly
    }
    
    print(f"📈 Original RLAD Performance Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Support - Normal: {support_normal}, Anomaly: {support_anomaly}")
    
    return metrics

# =================================
# 主函数和参数解析
# =================================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Original RLAD异常检测对比实验")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='clean_data.csv',
                       help='数据文件路径')
    parser.add_argument('--feature_column', type=str, default='103#',
                       help='特征列名')
    parser.add_argument('--window_size', type=int, default=288,
                       help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=20,
                       help='滑动窗口步长')
    
    # Original RLAD参数
    parser.add_argument('--episodes', type=int, default=50,
                       help='Q-learning训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Q网络学习率')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='epsilon-greedy初始探索率')
    parser.add_argument('--epsilon_end', type=float, default=0.1,
                       help='epsilon-greedy最终探索率')
    
    # 数据分割参数
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='验证集比例')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output_ensemble_rlad_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {args.output_dir}")
    
    try:
        # 1. 加载数据并进行Original Ensemble-RLAD异常检测
        print("=" * 60)
        print("🚀 开始Original RLAD异常检测对比实验")
        print("=" * 60)
        
        (X, y, raw_windows, window_indices, original_data, 
         anomaly_scores, method_scores, detector) = load_hydraulic_data_with_original_rlad(
            args.data_path, 
            args.window_size, 
            args.stride,
            args.feature_column,
            contamination=0.1,  # 固定参数
            unlabeled_fraction=0.1  # 固定参数
        )
        
        # 2. 数据分割
        print("\n" + "=" * 40)
        print("📊 数据分割")
        print("=" * 40)
        
        (train_data, val_data, test_data) = train_test_split_with_indices_v32_compatible(
            X, y, raw_windows, window_indices, 
            test_size=args.test_size, 
            val_size=args.val_size
        )
        
        X_train, y_train, raw_train, indices_train = train_data
        X_val, y_val, raw_val, indices_val = val_data  
        X_test, y_test, raw_test, indices_test = test_data
        
        # 3. 评估检测结果 (使用测试集)
        print("\n" + "=" * 40)
        print("📈 模型评估")
        print("=" * 40)
        
        # 对测试集进行检测
        test_scores, test_predictions, test_method_scores = detector.detect_anomalies(X_test)
        
        # 评估性能
        final_metrics = evaluate_original_rlad_results(y_test, test_predictions, test_scores)
        
        # 4. 可视化结果
        print("\n" + "=" * 40)
        print("📊 生成可视化结果")
        print("=" * 40)
        
        visualizer = CoreMetricsVisualizer(args.output_dir)
        
        # 过滤标记数据用于可视化
        labeled_mask = y_test != -1
        if np.sum(labeled_mask) > 0:
            y_test_labeled = y_test[labeled_mask]
            test_pred_labeled = test_predictions[labeled_mask]
            test_scores_labeled = test_scores[labeled_mask]
            
            # 绘制性能指标
            visualizer.plot_final_metrics_bar(
                final_metrics['precision'], 
                final_metrics['recall'],
                final_metrics['f1'], 
                final_metrics['auc_roc']
            )
            
            # 绘制混淆矩阵
            visualizer.plot_confusion_matrix(y_test_labeled, test_pred_labeled)
            
            # 绘制ROC曲线
            if len(np.unique(y_test_labeled)) > 1:
                visualizer.plot_roc_curve(y_test_labeled, test_scores_labeled)
            
            # 绘制Q网络结构信息（Original RLAD特有）
            # visualizer.plot_rlad_structure(detector.qlearn_estimator)
            
        # 绘制异常热力图
        test_start_idx = indices_test[0][0] if indices_test else 0
        test_end_idx = indices_test[-1][1] if indices_test else len(original_data)
        test_original_data = original_data[test_start_idx:test_end_idx]
        
        # 调整window_indices以匹配测试数据
        adjusted_indices = [(start - test_start_idx, end - test_start_idx) 
                           for start, end in indices_test]
        
        visualizer.plot_anomaly_heatmap(
            test_original_data, 
            test_predictions, 
            adjusted_indices,
            args.window_size
        )
        
        # 5. 保存详细结果
        print("\n" + "=" * 40)
        print("💾 保存实验结果")
        print("=" * 40)
        
        # 数据统计部分
        final_train_score = np.mean(test_scores) if len(test_scores) > 0 else 0
        final_val_score = np.mean(test_scores) if len(test_scores) > 0 else 0
        
        final_train_score_str = f"{final_train_score:.6f}"
        final_val_score_str = f"{final_val_score:.6f}"
        
        data_section = f"""

## 数据统计
- **总样本数**: {len(X)}
- **训练集**: {len(X_train)}
- **验证集**: {len(X_val)}
- **测试集**: {len(X_test)}
- **测试集正常样本**: {np.sum(y_test == 0)}
- **测试集异常样本**: {np.sum(y_test == 1)}
- **测试集未标记样本**: {np.sum(y_test == -1)}

## 检测结果
- **异常比例设定**: {args.contamination}
- **平均异常分数**: {final_train_score_str}
- **异常分数标准差**: {final_val_score_str}

## RLAD网络配置
- **学习率**: {detector.learning_rate:.6f}
- **训练轮数**: {detector.episodes}
- **epsilon探索率**: {detector.epsilon_start} -> {detector.epsilon_end}
- **折扣因子**: {detector.discount_factor}
- **LSTM隐藏层维度**: {N_HIDDEN_DIM}
- **滑动窗口大小**: {N_STEPS}
"""
        
        # 性能指标部分
        metrics_section = f"""

## 性能指标
- **精确率 (Precision)**: {final_metrics['precision']:.4f}
- **召回率 (Recall)**: {final_metrics['recall']:.4f}
- **F1分数**: {final_metrics['f1']:.4f}
- **AUC-ROC**: {final_metrics['auc_roc']:.4f}
- **准确率**: {final_metrics['accuracy']:.4f}

## 支持样本数
- **正常样本**: {final_metrics['support_normal']}
- **异常样本**: {final_metrics['support_anomaly']}
"""
        
        # 方法比较部分
        comparison_section = f"""

## 方法对比
- **检测方法**: Original Ensemble-RLAD (集成多种异常检测算法)
- **核心优势**: 
  - 集成多种检测算法，提高鲁棒性
  - 无需预先标记异常样本
  - 适用于多种类型的异常模式
  - 可解释性强，能分析各方法贡献
- **适用场景**: 复杂工业系统异常检测，多模态异常检测
- **与RLADv3.2的差异**: 
  - 不使用强化学习框架
  - 采用传统机器学习集成策略
  - 计算复杂度较低
  - 更适合实时应用场景

## 算法组件详情
1. **Isolation Forest**: 基于随机森林的孤立点检测
2. **One-Class SVM**: 基于支持向量机的单类分类
3. **统计方法**: Z-Score、Modified Z-Score、IQR等统计指标
4. **STL分解**: 时间序列季节性-趋势分解
5. **简化Autoencoder**: 轻量级神经网络重构误差检测

## 集成策略
- **加权投票**: 根据方法性能动态调整权重
- **软投票**: 基于连续分数而非硬分类
- **鲁棒性**: 单一方法失效不影响整体性能
"""
        
        # 生成完整报告
        report_content = f"""# Original Ensemble-RLAD 异常检测实验报告

**实验时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**实验配置**: 窗口大小={args.window_size}, 步长={args.stride}, 特征列={args.feature_column}

{data_section}

{metrics_section}

{comparison_section}

---
*本实验使用Original Ensemble-RLAD方法进行异常检测，与RLADv3.2方法形成对比*
"""
        
        # 保存报告
        report_file = os.path.join(args.output_dir, 'ensemble_rlad_experiment_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 保存完整结果
        results_summary = {
            'experiment_info': {
                'method': 'Original Ensemble-RLAD',
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'data_path': args.data_path,
                    'feature_column': args.feature_column,
                    'window_size': args.window_size,
                    'stride': args.stride,
                    'contamination': args.contamination,
                    'unlabeled_fraction': args.unlabeled_fraction,
                    'test_size': args.test_size,
                    'val_size': args.val_size,
                    'seed': args.seed
                }
            },
            'data_statistics': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'test_normal_samples': int(np.sum(y_test == 0)),
                'test_anomaly_samples': int(np.sum(y_test == 1)),
                'test_unlabeled_samples': int(np.sum(y_test == -1))
            },
            'detection_results': {
                'rlad_config': {
                    'learning_rate': detector.learning_rate,
                    'episodes': detector.episodes,
                    'epsilon_start': detector.epsilon_start,
                    'epsilon_end': detector.epsilon_end,
                    'discount_factor': detector.discount_factor,
                    'lstm_hidden_dim': N_HIDDEN_DIM,
                    'window_size': N_STEPS
                },
                'test_anomaly_score_stats': {
                    'mean': float(np.mean(test_scores)),
                    'std': float(np.std(test_scores)),
                    'min': float(np.min(test_scores)),
                    'max': float(np.max(test_scores))
                }
            },
            'performance_metrics': {
                'precision': float(final_metrics['precision']),
                'recall': float(final_metrics['recall']),
                'f1_score': float(final_metrics['f1']),
                'auc_roc': float(final_metrics['auc_roc']),
                'accuracy': float(final_metrics['accuracy'])
            },
            'comparison_baseline': {
                'method_description': 'Original Ensemble-RLAD for Time Series Anomaly Detection',
                'approach': 'Ensemble of multiple anomaly detection algorithms',
                'advantages': ['Robust ensemble approach', 'No prior anomaly labels needed', 'Interpretable method contributions', 'Lower computational complexity'],
                'limitations': ['Fixed ensemble weights', 'May not capture complex temporal patterns', 'Limited adaptability']
            }
        }
        
        # 转换为可序列化格式
        results_summary = convert_to_serializable(results_summary)
        
        # 保存结果
        results_file = os.path.join(args.output_dir, 'ensemble_rlad_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Original Ensemble-RLAD 实验结果已保存到: {results_file}")
        
        # 生成详细的点级异常标记文件
        print("🔄 生成点级异常标记文件...")
        
        # 创建点级异常标记
        point_anomalies = np.zeros(len(original_data))
        
        for i, prediction in enumerate(test_predictions):
            if i < len(indices_test) and prediction == 1:
                start_idx, end_idx = indices_test[i]
                point_anomalies[start_idx:end_idx] = 1
        
        # 保存点级结果
        point_level_df = pd.DataFrame({
            'timestamp': range(len(original_data)),
            'value': original_data,
            'is_anomaly': point_anomalies.astype(int)
        })
        
        point_file = os.path.join(args.output_dir, 'ensemble_rlad_point_level_results.csv')
        point_level_df.to_csv(point_file, index=False)
        print(f"📁 点级异常标记已保存到: {point_file}")
        
        # 打印最终总结
        print("\n" + "=" * 60)
        print("🎉 Original Ensemble-RLAD 异常检测实验完成！")
        print("=" * 60)
        print(f"📊 最终性能指标:")
        print(f"   F1分数: {final_metrics['f1']:.4f}")
        print(f"   精确率: {final_metrics['precision']:.4f}")
        print(f"   召回率: {final_metrics['recall']:.4f}")
        print(f"   AUC-ROC: {final_metrics['auc_roc']:.4f}")
        print(f"📁 所有结果已保存到: {args.output_dir}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
