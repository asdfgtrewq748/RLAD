"""
改进版RLAD: 基于强化学习与主动学习的时间序列异常检测
专用于液压支架工作阻力异常检测 - 修复与增强版本
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    """设置随机种子保证可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =================================
# 工具函数：修复JSON序列化问题
# =================================

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
    else:
        return obj

# =================================
# 1. 改进的数据集加载与预处理
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

def load_hydraulic_data_improved(data_path, window_size=288, stride=12):
    """
    改进的数据加载函数
    优化异常检测策略，增加数据多样性
    """
    print(f"正在加载数据: {data_path}")
    
    # 读取数据
    df = pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")
    
    # 选择支架压力列
    support_columns = [col for col in df.columns if '#' in col]
    if len(support_columns) == 0:
        # Fallback if no '#' in column names, select first few numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >=3:
            support_columns = numeric_cols[:3].tolist()
        elif len(numeric_cols) > 0:
            support_columns = numeric_cols.tolist() # Use whatever is available
        else:
            raise ValueError("No numeric columns found for support pressure.")

    print(f"发现支架列: {support_columns[:10]}...")
    
    # 选择前3个支架作为多通道输入
    if len(support_columns) >= 3:
        selected_cols = support_columns[:3]
    elif len(support_columns) > 0 : # if 1 or 2 columns
        selected_cols = support_columns + [support_columns[0]] * (3 - len(support_columns))
    else: # Should be caught by earlier ValueError
        raise ValueError("Not enough support columns to select 3 channels.")

    # 提取数值数据并处理缺失值
    data = df[selected_cols].fillna(method='ffill').fillna(method='bfill').fillna(0).values
    print(f"提取的数据形状: {data.shape}")
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 创建滑动窗口
    print("创建滑动窗口...")
    windows = []
    for i in range(0, len(data_scaled) - window_size + 1, stride):
        window = data_scaled[i:i + window_size]
        windows.append(window)
    
    X = np.array(windows)
    print(f"窗口数据形状: {X.shape}")
    
    # 改进的异常标签生成策略 - 增加异常样本多样性
    N = len(X)
    if N == 0:
        raise ValueError("No windows created. Check window_size, stride, and data length.")
    X_flat = X.reshape(N, -1)
    
    # 1. IsolationForest - 提高contamination以获得更多异常样本
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=300)
    iso_scores = iso_forest.fit_predict(X_flat)
    
    # 2. 基于统计的异常检测（Z-score）- 降低阈值
    window_means = np.mean(X_flat, axis=1)
    z_scores = np.abs((window_means - np.mean(window_means)) / (np.std(window_means) + 1e-8))
    stat_anomalies = z_scores > 1.8  # 降低到1.8以获得更多异常
    
    # 3. 基于方差的异常检测
    window_vars = np.var(X_flat, axis=1)
    var_threshold = np.percentile(window_vars, 85)  # 前15%认为是异常
    var_anomalies = window_vars > var_threshold
    
    # 4. 基于梯度变化的异常检测 (Ensure X_flat has more than 1 feature for diff)
    if X_flat.shape[1] > 1:
        gradients = np.abs(np.diff(X_flat, axis=1))
        gradient_means = np.mean(gradients, axis=1)
        grad_threshold = np.percentile(gradient_means, 85)
        grad_anomalies = gradient_means > grad_threshold
    else:
        grad_anomalies = np.zeros(N, dtype=bool) # No gradient if only one feature per window
    
    # 综合多种方法 - 至少一种方法认为是异常就标记为异常
    iso_anomalies = (iso_scores == -1)
    ensemble_anomalies = iso_anomalies | stat_anomalies | var_anomalies | grad_anomalies
    
    # 生成标签：1=异常, 0=正常, -1=未标注
    y = np.zeros(N, dtype=int)
    y[ensemble_anomalies] = 1
    
    # 进一步减少未标注样本比例到3%
    unlabeled_mask = np.random.random(N) < 0.03
    y[unlabeled_mask] = -1
    
    print(f"改进后标签分布: 正常={np.sum(y==0)}, 异常={np.sum(y==1)}, 未标注={np.sum(y==-1)}")
    
    # 数据集划分
    train_size = int(0.7 * N)
    val_size = int(0.15 * N)
    
    indices = np.arange(N)
    np.random.shuffle(indices) # Shuffle before splitting for more robust train/val/test sets

    X_train = X[indices[:train_size]]
    y_train = y[indices[:train_size]]
    X_val = X[indices[train_size:train_size + val_size]]
    y_val = y[indices[train_size:train_size + val_size]]
    X_test = X[indices[train_size + val_size:]]
    y_test = y[indices[train_size + val_size:]]
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# =================================
# 2. 优化的网络结构
# =================================

class EnhancedRLADAgent(nn.Module):
    """
    增强版RLAD智能体网络架构
    添加残差连接和注意力机制
    """
    
    def __init__(self, input_dim=3, seq_len=288, hidden_size=128, num_layers=3):
        super(EnhancedRLADAgent, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多层LSTM编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # 注意力机制
        lstm_output_size = hidden_size * 2  # 双向LSTM
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # 残差连接和层归一化
        self.ln_attention = nn.LayerNorm(lstm_output_size)
        
        # 全连接层
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 2)  # Q值输出
        
        # 正则化
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # 使用GELU激活函数
        
        # 层归一化
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name : # More specific weight names
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # 自注意力机制
        # MultiheadAttention expects (seq_len, batch, embed_dim) if batch_first=False
        # or (batch, seq_len, embed_dim) if batch_first=True. lstm_out is already (batch, seq_len, embed_dim)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out) # Query, Key, Value
        
        # 残差连接
        lstm_out = self.ln_attention(lstm_out + attn_out)
        
        # 全局平均池化和最大池化结合
        avg_pool = torch.mean(lstm_out, dim=1)  # (batch, hidden_size*2)
        max_pool, _ = torch.max(lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # 融合不同池化结果
        combined = avg_pool + max_pool  # 残差连接风格的融合
        
        # 全连接层
        x_fc = self.fc1(combined) # Renamed to avoid conflict with input x
        x_fc = self.ln1(x_fc)
        x_fc = self.gelu(x_fc)
        x_fc = self.dropout(x_fc)
        
        features = x_fc  # 保存特征用于可视化
        
        x_fc = self.fc2(x_fc)
        x_fc = self.ln2(x_fc)
        x_fc = self.gelu(x_fc)
        x_fc = self.dropout(x_fc)
        
        q_values = self.fc3(x_fc)
        
        if return_features:
            return q_values, features
        else:
            return q_values
    
    def get_action(self, state, epsilon=0.0):
        """获取动作，支持单样本推理"""
        if random.random() < epsilon:
            return random.randint(0, 1) # Action space size is 2 (0 or 1)
        else:
            was_training = self.training
            self.eval()
            
            with torch.no_grad():
                # Ensure state has batch dimension if it's a single sample
                if state.ndim == 2: # (seq_len, input_dim)
                    state = state.unsqueeze(0) # (1, seq_len, input_dim)
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
            
            if was_training:
                self.train()
            
            return action
# ... (第一部分结束) ...
# ... (第一部分代码继续) ...
# =================================
# 3. 优化的奖励函数和训练机制
# =================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """优先级经验回放"""
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
        if probs_sum == 0: # Avoid division by zero if all priorities are zero
            # Uniform sampling if all priorities are zero
            indices = np.random.choice(len(self.buffer), batch_size)
        else:
            probs /= probs_sum
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            
        experiences = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta) if probs_sum > 0 else np.ones_like(indices, dtype=np.float32)
        weights /= weights.max() # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        
        # Ensure states are correctly shaped (batch_size, seq_len, input_dim)
        # Assuming states in buffer are (seq_len, input_dim)
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

def enhanced_compute_reward(action, true_label, confidence=1.0, diversity_bonus=0.0):
    """
    增强的奖励函数
    添加多样性奖励和动态调整
    """
    if true_label == -1:  # 未标注样本
        return 0.0  # No reward for unlabeled samples
    
    base_reward = confidence * 1.0 # Base magnitude of reward
    
    # Define reward values
    TP_REWARD = 5.0  # True Positive (correctly identified anomaly)
    TN_REWARD = 1.0  # True Negative (correctly identified normal)
    FN_PENALTY = -3.0 # False Negative (missed anomaly) - Critical error
    FP_PENALTY = -0.5 # False Positive (normal classified as anomaly) - Less critical but still penalized

    if action == true_label:
        if true_label == 1:  # Correctly identified anomaly (TP)
            reward = base_reward * TP_REWARD
        else:  # Correctly identified normal (TN)
            reward = base_reward * TN_REWARD
    else: # Misclassification
        if true_label == 1 and action == 0:  # Missed anomaly (FN)
            reward = base_reward * FN_PENALTY
        elif true_label == 0 and action == 1: # Normal classified as anomaly (FP)
            reward = base_reward * FP_PENALTY
        else: # Should not happen if actions are 0 and 1
            reward = -base_reward 
    
    # 添加多样性奖励 (e.g., if the model explores less common states or actions)
    reward += diversity_bonus
    
    return reward

def enhanced_warmup_with_multiple_methods(X_train, y_train, replay_buffer, agent, device):
    """使用多种方法的增强预热"""
    print("开始增强版预热...")
    
    labeled_mask = (y_train != -1)
    X_labeled = X_train[labeled_mask]
    y_labeled = y_train[labeled_mask]
    
    if len(X_labeled) == 0:
        print("警告: 没有已标注样本用于预热")
        return
    
    print(f"预热样本数量: {len(X_labeled)}")
    
    # 1. IsolationForest
    iso_forest = IsolationForest(contamination=0.15, random_state=42, n_estimators=300)
    X_flat = X_labeled.reshape(len(X_labeled), -1)
    iso_forest.fit(X_flat)
    anomaly_scores = iso_forest.decision_function(X_flat) # Lower scores are more anomalous
    
    # 2. 基于聚类的异常检测 (Optional, ensure sklearn.cluster is imported if used)
    # from sklearn.cluster import KMeans
    # from sklearn.metrics import pairwise_distances
    
    # n_clusters = min(10, max(2, len(X_labeled) // 5)) # Ensure at least 2 clusters if using KMeans
    # if n_clusters > 1 and len(X_labeled) > n_clusters : # Check if enough samples for Kmeans
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    #     cluster_labels = kmeans.fit_predict(X_flat)
        
    #     distances = []
    #     for i, label in enumerate(cluster_labels):
    #         dist = np.linalg.norm(X_flat[i] - kmeans.cluster_centers_[label])
    #         distances.append(dist)
    #     distances = np.array(distances)
        
    #     # Combine IsolationForest (lower is more anomalous) and K-Means distance (higher is more anomalous)
    #     # Normalize distances to be comparable, e.g., (distances - min) / (max - min)
    #     norm_distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
    #     combined_scores = 0.7 * (-anomaly_scores) + 0.3 * norm_distances # Higher score = more anomalous
    # else:
    combined_scores = -anomaly_scores # Higher score = more anomalous (negating iso_forest scores)
    
    # 分层抽样
    n_samples_to_add = min(len(X_labeled), 5000) # Limit number of samples added during warmup
    
    # Sort by combined_scores (higher scores are more anomalous)
    sorted_indices = np.argsort(combined_scores)[::-1] # Descending order

    # Select top P% as anomalies, bottom Q% as normal, rest as uncertain for warmup
    # This gives higher confidence to samples at the extremes of the anomaly score distribution
    n_abnormal_warmup = int(0.2 * n_samples_to_add)
    n_normal_warmup = int(0.5 * n_samples_to_add)
    
    abnormal_indices_warmup = sorted_indices[:n_abnormal_warmup]
    normal_indices_warmup = sorted_indices[-n_normal_warmup:]
    # Middle indices can be sampled randomly or also based on scores
    middle_indices_candidates = sorted_indices[n_abnormal_warmup:-n_normal_warmup]
    n_middle_warmup = min(len(middle_indices_candidates), n_samples_to_add - n_abnormal_warmup - n_normal_warmup)
    middle_indices_warmup = np.random.choice(middle_indices_candidates, size=n_middle_warmup, replace=False) if n_middle_warmup > 0 else []


    # 添加到回放池
    for indices_group, confidence_level in [
        (abnormal_indices_warmup, 0.95), 
        (normal_indices_warmup, 0.85), 
        (middle_indices_warmup, 0.6)
    ]:
        for original_idx in indices_group:
            # original_idx is an index into X_labeled / y_labeled
            state_np = X_labeled[original_idx]
            true_label = y_labeled[original_idx]

            state = torch.FloatTensor(state_np).to(device) # (seq_len, input_dim)
            
            # For warmup, action can be based on pseudo-label or random
            if confidence_level > 0.8: # Higher confidence, use true_label (pseudo)
                action = true_label
            elif confidence_level > 0.6: # Medium confidence
                action = true_label if random.random() < 0.7 else 1 - true_label
            else: # Lower confidence, explore
                action = random.randint(0, 1)
            
            diversity_bonus = 0.1 if true_label == 1 else 0.0 # Small bonus for anomalies
            reward = enhanced_compute_reward(action, true_label, confidence_level, diversity_bonus)
            
            # For warmup, next_state can be the same state or a randomly chosen one
            # Using same state for simplicity, or a subsequent sample if available
            next_state_np_idx = (original_idx + 1) % len(X_labeled)
            next_state_np = X_labeled[next_state_np_idx]
            next_state = torch.FloatTensor(next_state_np).to(device)
            done = False # Typically, episodes don't end during warmup additions
            
            replay_buffer.push(
                state.cpu(), # Store as CPU tensors in buffer
                action,
                reward,
                next_state.cpu(),
                done
            )
    
    print(f"增强预热完成，回放池大小: {len(replay_buffer)}")

# =================================
# 4. 增强的训练函数
# =================================

def enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, 
                           gamma=0.99, batch_size=64, beta=0.4):
    """增强的DQN训练步骤，支持优先级经验回放"""
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0 # Return loss and avg TD error
    
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)
    
    states = states.to(device) # (batch_size, seq_len, input_dim)
    actions = actions.to(device) # (batch_size,)
    rewards = rewards.to(device) # (batch_size,)
    next_states = next_states.to(device) # (batch_size, seq_len, input_dim)
    dones = dones.to(device) # (batch_size,)
    weights = torch.FloatTensor(weights).to(device) # (batch_size,)
    
    # 当前Q值: Q(s,a)
    # agent(states) output shape: (batch_size, num_actions)
    # actions.unsqueeze(1) shape: (batch_size, 1)
    current_q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1) # Shape: (batch_size,)
    
    # 目标Q值 - 使用Double DQN
    with torch.no_grad():
        # 使用在线网络选择动作: a' = argmax_a Q_online(s', a)
        next_actions = agent(next_states).argmax(dim=1) # Shape: (batch_size,)
        # 使用目标网络计算Q值: Q_target(s', a')
        # target_agent(next_states) shape: (batch_size, num_actions)
        # next_actions.unsqueeze(1) shape: (batch_size, 1)
        next_q_values_target = target_agent(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1) # Shape: (batch_size,)
        
        # TD Target: r + gamma * Q_target(s', argmax_a Q_online(s',a)) * (1-done)
        target_q_values = rewards + (gamma * next_q_values_target * ~dones) # Shape: (batch_size,)
    
    # TD误差
    td_errors = target_q_values - current_q_values # Consistent with PER update: priority ~ |TD_error|
    
    # 加权Huber Loss (Smooth L1 Loss)
    # F.smooth_l1_loss computes 0.5 * x^2 if |x| < beta, and |x| - 0.5 * beta otherwise.
    # Here, beta for smooth_l1_loss is 1.0 by default.
    loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none', beta=1.0)).mean()
    
    # 优化
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0) # Gradient clipping
    optimizer.step()
    
    # 更新优先级
    priorities_np = np.abs(td_errors.detach().cpu().numpy()) + 1e-6 # Add epsilon to avoid zero priority
    replay_buffer.update_priorities(indices, priorities_np)
    
    return loss.item(), td_errors.abs().mean().item()


def enhanced_evaluate_model(agent, data_loader, device):
    """增强的模型评估，包含更多指标"""
    agent.eval()
    
    all_predictions = []
    all_labels = []
    all_q_values_logits = []
    all_probabilities_positive_class = [] # For ROC/PR curves
    all_features = []
    
    with torch.no_grad():
        for states, labels in data_loader: # labels from DataLoader are original labels (-1, 0, 1)
            states = states.to(device)
            
            # Filter out unlabeled data for metric calculation
            labeled_mask = (labels != -1)
            if labeled_mask.sum() == 0:
                continue # Skip batch if no labeled samples
                
            states_labeled = states[labeled_mask]
            labels_labeled = labels[labeled_mask] # These are 0s and 1s
            
            if states_labeled.size(0) == 0: # Double check after filtering
                continue

            q_values_logits, features = agent(states_labeled, return_features=True)
            predictions = q_values_logits.argmax(dim=1)
            
            probabilities = F.softmax(q_values_logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_labeled.cpu().numpy())
            all_q_values_logits.extend(q_values_logits.cpu().numpy())
            all_features.extend(features.cpu().numpy())
            all_probabilities_positive_class.extend(probabilities[:, 1].cpu().numpy()) # Prob for anomaly class (class 1)
    
    agent.train() # Set back to train mode

    if len(all_predictions) == 0:
        # Return default values if no labeled samples were evaluated
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "precision_per_class": [0.0, 0.0], 
            "recall_per_class": [0.0, 0.0], 
            "f1_per_class": [0.0, 0.0],
            "predictions": [], "labels": [], 
            "q_values_logits": [], "probabilities_positive_class": [], "features": []
        }
    
    all_predictions_np = np.array(all_predictions)
    all_labels_np = np.array(all_labels)
    
    # Ensure there are both classes for per-class metrics, or handle appropriately
    unique_labels = np.unique(all_labels_np)
    num_classes = 2 # Assuming 0 and 1

    # Weighted metrics (handles imbalance)
    precision_w = precision_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    recall_w = recall_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    f1_w = f1_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    
    # Per-class metrics
    # Ensure labels=[0, 1] for consistent output length if one class is missing in predictions/labels
    precision_pc = precision_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
    recall_pc = recall_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
    f1_pc = f1_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
        
    return {
        "precision": float(precision_w),
        "recall": float(recall_w),
        "f1": float(f1_w),
        "precision_per_class": [float(x) for x in precision_pc],
        "recall_per_class": [float(x) for x in recall_pc],
        "f1_per_class": [float(x) for x in f1_pc],
        "predictions": all_predictions_np.tolist(),
        "labels": all_labels_np.tolist(),
        "q_values_logits": np.array(all_q_values_logits).tolist(),
        "probabilities_positive_class": np.array(all_probabilities_positive_class).tolist(),
        "features": np.array(all_features).tolist()
    }
# ... (第二部分结束) ...
# ... (第二部分代码继续) ...
# =================================
# 5. 增强的主训练函数
# =================================

def enhanced_train_rlad(agent, target_agent, optimizer, scheduler, replay_buffer, 
                       X_train, y_train, X_val, y_val, device, 
                       num_episodes=150, target_update_freq=15,
                       epsilon_start=0.95, epsilon_end=0.02, epsilon_decay_rate=0.995, # Changed to decay_rate
                       batch_size_rl=64, # RL batch size
                       output_dir="./output"):
    """增强的RLAD主训练函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    training_history = {
        'episodes': [], 'train_loss': [], 'avg_td_error': [],
        'val_f1': [], 'val_precision': [], 'val_recall': [],
        'epsilon': [], 'replay_buffer_size': [], 'learning_rate': [],
        'anomaly_f1': [], 'normal_f1': [] 
    }
    
    best_val_f1 = 0.0 # Overall F1 on validation
    best_anomaly_val_f1 = 0.0 # Anomaly F1 on validation
    patience = 40 # Early stopping patience
    patience_counter = 0
    
    print("开始增强的RLAD训练...")
    
    # Filter out unlabeled samples for RL training interaction
    labeled_train_mask = (y_train != -1)
    X_train_labeled = X_train[labeled_train_mask]
    y_train_labeled = y_train[labeled_train_mask] # Contains 0s and 1s
    
    if len(X_train_labeled) == 0:
        print("错误：训练集中没有已标记的样本。无法继续训练。")
        return training_history # Or raise an error

    print(f"用于RL交互的已标记训练样本数量: {len(X_train_labeled)}")
    print(f"异常样本数量 (已标记): {np.sum(y_train_labeled == 1)}")
    print(f"正常样本数量 (已标记): {np.sum(y_train_labeled == 0)}")
    
    epsilon = epsilon_start

    for episode in tqdm(range(num_episodes), desc="训练进度"):
        episode_losses = []
        episode_td_errors = []
        
        # Epsilon decay (exponential)
        epsilon = max(epsilon_end, epsilon * epsilon_decay_rate)
        
        # Beta for PER (linear annealing)
        beta_start = 0.4
        beta_frames = num_episodes * (len(X_train_labeled) // batch_size_rl if batch_size_rl > 0 else 1) # Approx total steps
        beta = min(1.0, beta_start + episode * (1.0 - beta_start) / (num_episodes * 0.8)) # Anneal beta over 80% of episodes


        agent.train()
        
        # Shuffle labeled data indices for this episode's interactions
        shuffled_indices = np.random.permutation(len(X_train_labeled))
        
        # Interact with the environment (each labeled sample is a step)
        for step_idx in range(len(X_train_labeled)):
            current_sample_idx_in_labeled = shuffled_indices[step_idx]
            
            state_np = X_train_labeled[current_sample_idx_in_labeled]
            true_label = y_train_labeled[current_sample_idx_in_labeled]
            
            state = torch.FloatTensor(state_np).to(device) # (seq_len, input_dim)
            
            # Agent takes an action
            action = agent.get_action(state, epsilon) # state already (1, seq_len, input_dim) if get_action handles it
            
            # Determine reward
            diversity_bonus = 0.1 if true_label == 1 else 0.0 # Small bonus for interacting with anomaly
            reward = enhanced_compute_reward(action, true_label, confidence=1.0, diversity_bonus=diversity_bonus)
            
            # Determine next_state (can be the next sample in shuffled list, or a fixed strategy)
            # For simplicity, let's use the next sample in the shuffled sequence, looping if at the end
            next_sample_idx_in_labeled = shuffled_indices[(step_idx + 1) % len(X_train_labeled)]
            next_state_np = X_train_labeled[next_sample_idx_in_labeled]
            next_state = torch.FloatTensor(next_state_np).to(device)
            
            done = False # In this setup, an episode doesn't naturally end after one sample.
                         # Consider 'done' if it's the last sample of an epoch/episode for some RL algos.
                         # Here, we treat each sample interaction as a transition.

            replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), done)
            
            # Train DQN step
            if len(replay_buffer) >= batch_size_rl * 2: # Start training once enough samples
                loss, td_err = enhanced_train_dqn_step(
                    agent, target_agent, replay_buffer, 
                    optimizer, device, batch_size=batch_size_rl, beta=beta
                )
                if loss is not None: episode_losses.append(loss)
                if td_err is not None: episode_td_errors.append(td_err)
        
        # Update target network
        if episode % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # Learning rate scheduler step (if based on episodes)
        scheduler.step() # Assuming CosineAnnealingWarmRestarts or similar
        
        # Validation
        val_dataset = HydraulicDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False) # Larger batch for eval
        val_metrics = enhanced_evaluate_model(agent, val_loader, device)
        
        # Record history
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0.0
        training_history['episodes'].append(episode)
        training_history['train_loss'].append(avg_loss)
        training_history['avg_td_error'].append(avg_td_error)
        training_history['val_f1'].append(val_metrics['f1'])
        training_history['val_precision'].append(val_metrics['precision'])
        training_history['val_recall'].append(val_metrics['recall'])
        training_history['epsilon'].append(epsilon)
        training_history['replay_buffer_size'].append(len(replay_buffer))
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        current_anomaly_f1 = val_metrics['f1_per_class'][1] if len(val_metrics['f1_per_class']) > 1 else 0.0
        current_normal_f1 = val_metrics['f1_per_class'][0] if len(val_metrics['f1_per_class']) > 0 else 0.0
        training_history['anomaly_f1'].append(current_anomaly_f1)
        training_history['normal_f1'].append(current_normal_f1)
        
        # Save best model - prioritize anomaly F1, then overall F1
        improved = False
        if current_anomaly_f1 > best_anomaly_val_f1:
            best_anomaly_val_f1 = current_anomaly_f1
            best_val_f1 = val_metrics['f1'] # Update overall F1 as well
            improved = True
        elif current_anomaly_f1 == best_anomaly_val_f1 and val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            improved = True

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
                'training_history_partial': training_history # Save current history
            }, os.path.join(output_dir, 'best_enhanced_rlad_model.pth'))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"早停在episode {episode}，最佳验证集F1: {best_val_f1:.4f}, 最佳验证集异常F1: {best_anomaly_val_f1:.4f}")
            break
        
        if episode % 5 == 0 or episode == num_episodes -1 : # Print more frequently
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"Loss: {avg_loss:.4f}, Avg TD Err: {avg_td_error:.4f}, Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val Anomaly F1: {current_anomaly_f1:.4f}, Val Normal F1: {current_normal_f1:.4f}")
            print(f"Epsilon: {epsilon:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, Buffer: {len(replay_buffer)}")
    
    print(f"\n训练完成！最佳验证集F1分数: {best_val_f1:.4f}, 最佳验证集异常F1: {best_anomaly_val_f1:.4f}")
    return training_history

# =================================
# 6. 增强的可视化类
# =================================

class EnhancedRLADVisualizer:
    def __init__(self, output_dir="./output_visuals"): # Ensure this dir is passed correctly
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_training_history(self, training_history):
        fig, axes = plt.subplots(3, 3, figsize=(22, 18)) # Adjusted size
        fig.suptitle('模型训练历史与性能指标', fontsize=16, fontweight='bold')
        
        episodes = training_history['episodes']
        
        axes[0, 0].plot(episodes, training_history['train_loss'], 'b-', linewidth=2, label='训练损失')
        axes[0, 0].set_title('训练损失', fontsize=12)
        axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].grid(True, alpha=0.4); axes[0,0].legend()

        axes[0, 1].plot(episodes, training_history['avg_td_error'], 'm-', linewidth=2, label='平均TD误差')
        axes[0, 1].set_title('平均TD误差', fontsize=12)
        axes[0, 1].set_xlabel('Episode'); axes[0, 1].set_ylabel('TD Error'); axes[0, 1].grid(True, alpha=0.4); axes[0,1].legend()
        
        axes[0, 2].plot(episodes, training_history['val_f1'], 'g-', label='整体F1 (验证集)', linewidth=2)
        axes[0, 2].set_title('验证集F1分数', fontsize=12)
        axes[0, 2].set_xlabel('Episode'); axes[0, 2].set_ylabel('F1 Score'); axes[0, 2].grid(True, alpha=0.4); axes[0,2].legend()
        
        axes[1, 0].plot(episodes, training_history['anomaly_f1'], 'r-', label='异常F1 (验证集)', linewidth=2)
        axes[1, 0].plot(episodes, training_history['normal_f1'], 'c-', label='正常F1 (验证集)', linewidth=2)
        axes[1, 0].set_title('分类别F1分数 (验证集)', fontsize=12)
        axes[1, 0].set_xlabel('Episode'); axes[1, 0].set_ylabel('F1 Score'); axes[1, 0].grid(True, alpha=0.4); axes[1,0].legend()
        
        axes[1, 1].plot(episodes, training_history['val_precision'], 'b-', label='精确率 (验证集)', linewidth=1.5)
        axes[1, 1].plot(episodes, training_history['val_recall'], 'r-', label='召回率 (验证集)', linewidth=1.5)
        axes[1, 1].set_title('精确率和召回率 (验证集)', fontsize=12)
        axes[1, 1].set_xlabel('Episode'); axes[1, 1].set_ylabel('Score'); axes[1, 1].grid(True, alpha=0.4); axes[1,1].legend()
        
        axes[1, 2].plot(episodes, training_history['epsilon'], 'purple', linewidth=2, label='Epsilon')
        axes[1, 2].set_title('探索率(Epsilon)衰减', fontsize=12)
        axes[1, 2].set_xlabel('Episode'); axes[1, 2].set_ylabel('Epsilon'); axes[1, 2].grid(True, alpha=0.4); axes[1,2].legend()
        
        axes[2, 0].plot(episodes, training_history['replay_buffer_size'], 'orange', linewidth=2, label='Buffer大小')
        axes[2, 0].set_title('经验回放池大小', fontsize=12)
        axes[2, 0].set_xlabel('Episode'); axes[2, 0].set_ylabel('Buffer Size'); axes[2, 0].grid(True, alpha=0.4); axes[2,0].legend()
        
        axes[2, 1].plot(episodes, training_history['learning_rate'], 'brown', linewidth=2, label='学习率')
        axes[2, 1].set_title('学习率变化', fontsize=12)
        axes[2, 1].set_xlabel('Episode'); axes[2, 1].set_ylabel('Learning Rate'); axes[2, 1].grid(True, alpha=0.4); axes[2,1].legend()
        
        if len(training_history['train_loss']) > 10:
            ws = min(10, len(training_history['train_loss']) // 2)
            smoothed_loss = np.convolve(training_history['train_loss'], np.ones(ws)/ws, mode='valid')
            axes[2, 2].plot(episodes[ws-1:], smoothed_loss, 'navy', linewidth=2, label=f'平滑损失 (窗口={ws})')
        else:
            axes[2, 2].plot(episodes, training_history['train_loss'], 'navy', linewidth=2, label='训练损失 (原始)')
        axes[2, 2].set_title('训练损失 (平滑)', fontsize=12)
        axes[2, 2].set_xlabel('Episode'); axes[2, 2].set_ylabel('Smoothed Loss'); axes[2, 2].grid(True, alpha=0.4); axes[2,2].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.savefig(os.path.join(self.output_dir, 'enhanced_training_history.png'), dpi=300)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, class_names=['正常', '异常']):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
        plt.title('混淆矩阵', fontsize=15, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.show()

    def plot_roc_pr_curves(self, y_true, y_scores_positive_class):
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores_positive_class)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores_positive_class)
        pr_auc = average_precision_score(y_true, y_scores_positive_class) # Area under PR curve

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('ROC曲线 和 Precision-Recall曲线', fontsize=16, fontweight='bold')

        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (面积 = {roc_auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('假阳性率 (FPR)', fontsize=12)
        axes[0].set_ylabel('真阳性率 (TPR)', fontsize=12)
        axes[0].set_title('ROC曲线', fontsize=14)
        axes[0].legend(loc="lower right", fontsize=10)
        axes[0].grid(True, alpha=0.4)

        axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {pr_auc:.3f})')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('召回率 (Recall)', fontsize=12)
        axes[1].set_ylabel('精确率 (Precision)', fontsize=12)
        axes[1].set_title('Precision-Recall曲线', fontsize=14)
        axes[1].legend(loc="lower left", fontsize=10)
        axes[1].grid(True, alpha=0.4)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'roc_pr_curves.png'), dpi=300)
        plt.show()

    def plot_tsne_features(self, features_np, labels_np, class_names=['正常', '异常']):
        if len(features_np) == 0:
            print("没有特征可供t-SNE可视化。")
            return
        
        print(f"开始t-SNE降维，特征数量: {len(features_np)}")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_np)-1), n_iter=300)
        features_2d = tsne.fit_transform(features_np)
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']
        for i, class_name in enumerate(class_names):
            plt.scatter(features_2d[labels_np == i, 0], features_2d[labels_np == i, 1], 
                        c=colors[i], label=class_name, alpha=0.6, s=50)
        
        plt.title('t-SNE 特征可视化 (测试集)', fontsize=15, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'tsne_features_visualization.png'), dpi=300)
        plt.show()

# =================================
# 7. 主函数
# =================================

def parse_args():
    parser = argparse.ArgumentParser(description='增强版RLAD液压支架异常检测')
    
    parser.add_argument('--data_path', type=str, 
                       default="C:/Users/Liu HaoTian/Desktop/rnn+tcn+transformer+kan/time_series/timeseries/examples/RLAD/clean_data.csv",
                       help='数据文件路径')
    parser.add_argument('--window_size', type=int, default=288, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=12, help='滑动窗口步长')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数 (原为3, 可调)') # Adjusted default
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率 (原为1e-3, 可调)') # Adjusted default
    parser.add_argument('--num_episodes', type=int, default=100, help='训练轮数 (原为150, 可调)') # Adjusted default
    parser.add_argument('--output_dir', type=str, 
                        default="C:/Users/Liu HaoTian/Desktop/rnn+tcn+transformer+kan/time_series/timeseries/examples/RLAD/output", 
                        help='输出目录的基础路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default="auto", help='设备选择 (auto, cpu, cuda)')
    parser.add_argument('--batch_size_rl', type=int, default=64, help='强化学习训练的批次大小')
    parser.add_argument('--target_update_freq', type=int, default=10, help='目标网络更新频率 (episodes)') # Adjusted default
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99, help='Epsilon指数衰减率')


    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Output directory will be args.output_dir / f"enhanced_rlad_results_{timestamp}"
    # Ensure args.output_dir itself exists, then create the timestamped subfolder
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir_timestamped = os.path.join(args.output_dir, f"enhanced_rlad_results_{timestamp}")
    os.makedirs(output_dir_timestamped, exist_ok=True)
    print(f"所有输出将保存到: {output_dir_timestamped}")
    
    config_to_save = convert_to_serializable(vars(args))
    # Update output_dir in saved config to the timestamped one for clarity
    config_to_save['output_dir_actual'] = output_dir_timestamped 
    with open(os.path.join(output_dir_timestamped, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=4) # Increased indent
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_hydraulic_data_improved(
            data_path=args.data_path, window_size=args.window_size, stride=args.stride
        )
        
        agent = EnhancedRLADAgent(
            input_dim=X_train.shape[-1], seq_len=args.window_size, # Use actual input_dim
            hidden_size=args.hidden_size, num_layers=args.num_layers
        ).to(device)
        target_agent = EnhancedRLADAgent(
            input_dim=X_train.shape[-1], seq_len=args.window_size,
            hidden_size=args.hidden_size, num_layers=args.num_layers
        ).to(device)
        target_agent.load_state_dict(agent.state_dict())
        target_agent.eval() # Target network is only for inference

        optimizer = optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(10, args.num_episodes // 5), T_mult=1, eta_min=1e-7)
        
        replay_buffer = PrioritizedReplayBuffer(capacity=50000, alpha=0.6) # Increased capacity
        
        print(f"模型参数数量: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")
        
        enhanced_warmup_with_multiple_methods(X_train, y_train, replay_buffer, agent, device)
        
        training_history = enhanced_train_rlad(
            agent, target_agent, optimizer, scheduler, replay_buffer,
            X_train, y_train, X_val, y_val, device,
            num_episodes=args.num_episodes, 
            target_update_freq=args.target_update_freq,
            epsilon_decay_rate=args.epsilon_decay_rate,
            batch_size_rl=args.batch_size_rl,
            output_dir=output_dir_timestamped
        )
        
        # Save final training history
        with open(os.path.join(output_dir_timestamped, 'training_history_final.json'), 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(training_history), f, ensure_ascii=False, indent=4)

        visualizer = EnhancedRLADVisualizer(output_dir_timestamped)
        if training_history['episodes']: # Check if training actually ran
             visualizer.plot_training_history(training_history)
        
        best_model_path = os.path.join(output_dir_timestamped, 'best_enhanced_rlad_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            agent.load_state_dict(checkpoint['agent_state_dict'])
            print(f"\n已加载最佳模型进行最终测试 (Episode {checkpoint.get('episode', 'N/A')})")
            print(f"最佳验证集 F1: {checkpoint.get('best_val_f1', 0.0):.4f}, 异常F1: {checkpoint.get('best_anomaly_val_f1',0.0):.4f}")
        
        test_dataset = HydraulicDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = enhanced_evaluate_model(agent, test_loader, device)
        
        print(f"\n最终测试集评估结果:")
        print(f"  整体精确率: {test_metrics['precision']:.4f}")
        print(f"  整体召回率: {test_metrics['recall']:.4f}")
        print(f"  整体F1分数: {test_metrics['f1']:.4f}")
        if len(test_metrics['f1_per_class']) >= 2:
            print(f"  正常类F1: {test_metrics['f1_per_class'][0]:.4f}")
            print(f"  异常类F1: {test_metrics['f1_per_class'][1]:.4f}")

        with open(os.path.join(output_dir_timestamped, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(test_metrics), f, ensure_ascii=False, indent=4)

        # Generate and save additional visualizations for the test set
        if test_metrics['labels']: # If there were labeled samples in test set
            visualizer.plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
            if test_metrics['probabilities_positive_class']:
                 visualizer.plot_roc_pr_curves(test_metrics['labels'], test_metrics['probabilities_positive_class'])
            if test_metrics['features']:
                 visualizer.plot_tsne_features(np.array(test_metrics['features']), np.array(test_metrics['labels']))

        print(f"\n增强版RLAD训练和评估完成！所有结果保存在: {output_dir_timestamped}")
        
    except Exception as e:
        print(f"发生严重错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
# ... (第三部分结束) ...