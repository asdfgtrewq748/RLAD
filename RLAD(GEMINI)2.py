"""
改进版RLAD: 基于强化学习与主动学习的时间序列异常检测
专用于液压支架工作阻力异常检测 - 修复与增强版本 (使用单一特征并实现逐点异常标记)
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
from mpl_toolkits.mplot3d import Axes3D

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'

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

def load_hydraulic_data_improved(data_path, window_size=288, stride=12): # MODIFIED: window_size default
    """
    改进的数据加载函数
    优化异常检测策略，增加数据多样性
    使用单一选择的支架列
    """
    print(f"正在加载数据: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")
    
    support_columns = [col for col in df.columns if '#' in col]
    selected_column_name = None

    if len(support_columns) > 0:
        selected_column_name = support_columns[0] # MODIFIED: Select only the first support column
        print(f"选择的支架列: {selected_column_name}")
        selected_cols = [selected_column_name]
    else:
        print("警告: 未在列名中找到 '#'。将尝试使用第一个数值列作为支架列。")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_column_name = numeric_cols[0] # MODIFIED: Select only the first numeric column
            print(f"选择的数值列: {selected_column_name}")
            selected_cols = [selected_column_name]
        else:
            raise ValueError("未找到包含 '#' 的列，且数据中没有数值列。")

    print(f"选择的列数: {len(selected_cols)}, 列名: {selected_cols}")

    # Store original data with Date for point-level mapping later
    # Ensure 'Date' column exists, otherwise handle
    if 'Date' not in df.columns:
        print("警告: 'Date' 列未在CSV中找到。将使用索引作为日期替代。")
        df['Date'] = df.index.astype(str) # Create a pseudo-date if not present
        
    df_for_point_mapping = df[['Date'] + selected_cols].copy()
    
    # Extract numerical data for model input and handle missing values
    data_values = df[selected_cols].fillna(method='ffill').fillna(method='bfill').fillna(0).values
    print(f"提取的用于模型训练的数据形状: {data_values.shape}") # Shape will be (num_timesteps, 1)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    print("创建滑动窗口...")
    windows = []
    window_start_indices_all = [] # Store start indices of all windows
    for i in range(0, len(data_scaled) - window_size + 1, stride):
        window = data_scaled[i:i + window_size]
        windows.append(window)
        window_start_indices_all.append(i)
    
    if not windows:
        raise ValueError(f"未能创建滑动窗口。数据长度 {len(data_scaled)}, 窗口大小 {window_size}。请检查参数。")

    X = np.array(windows) 
    window_start_indices_all = np.array(window_start_indices_all)
    print(f"窗口数据形状: {X.shape}")
    
    N = len(X)
    if N == 0:
        raise ValueError("未能创建滑动窗口。请检查 window_size, stride, 和数据长度。")
    
    X_flat = X.reshape(N, -1) 
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=300)
    iso_scores = iso_forest.fit_predict(X_flat)
    
    window_feature_means = np.mean(X_flat, axis=1)
    z_scores = np.abs((window_feature_means - np.mean(window_feature_means)) / (np.std(window_feature_means) + 1e-8))
    stat_anomalies = z_scores > 1.8
    
    window_feature_vars = np.var(X_flat, axis=1)
    var_threshold = np.percentile(window_feature_vars, 85)
    var_anomalies = window_feature_vars > var_threshold
    
    if X_flat.shape[1] > 1:
        gradients = np.abs(np.diff(X_flat, axis=1))
        gradient_means = np.mean(gradients, axis=1)
        grad_threshold = np.percentile(gradient_means, 85)
        grad_anomalies = gradient_means > grad_threshold
    else: # If only one value in flattened window (e.g. window_size=1, num_features=1)
        grad_anomalies = np.zeros(N, dtype=bool)
    
    iso_anomalies = (iso_scores == -1)
    ensemble_anomalies = iso_anomalies | stat_anomalies | var_anomalies | grad_anomalies
    
    y = np.zeros(N, dtype=int)
    y[ensemble_anomalies] = 1
    
    unlabeled_mask = np.random.random(N) < 0.03
    y[unlabeled_mask] = -1
    
    print(f"改进后标签分布: 正常={np.sum(y==0)}, 异常={np.sum(y==1)}, 未标注={np.sum(y==-1)}")
    
    train_size = int(0.7 * N)
    val_size = int(0.15 * N)
    
    indices = np.arange(N)
    np.random.shuffle(indices)

    X_train = X[indices[:train_size]]
    y_train = y[indices[:train_size]]
    X_val = X[indices[train_size:train_size + val_size]]
    y_val = y[indices[train_size:train_size + val_size]]
    X_test = X[indices[train_size + val_size:]]
    y_test = y[indices[train_size + val_size:]]

    # Get the original start indices for the test windows
    test_window_original_indices = window_start_indices_all[indices[train_size + val_size:]]
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, \
           test_window_original_indices, df_for_point_mapping, selected_column_name

# =================================
# 2. 优化的网络结构
# =================================

class EnhancedRLADAgent(nn.Module):
    """
    增强版RLAD智能体网络架构
    添加残差连接和注意力机制
    """
    
    def __init__(self, input_dim, seq_len=288, hidden_size=128, num_layers=3): # MODIFIED: seq_len default
        super(EnhancedRLADAgent, self).__init__()
        
        self.input_dim = input_dim # Should be 1 now
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
        combined = avg_pool + max_pool
        
        x_fc = self.fc1(combined)
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
# 3. 优化的奖励函数和训练机制
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

        # Handle case where all priorities are zero or very small leading to probs_sum = 0
        if probs_sum == 0 or len(self.buffer) == 0:
            # Fallback to uniform sampling if probs_sum is zero or buffer is empty
            if len(self.buffer) == 0: return None # Cannot sample from empty buffer
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

def enhanced_compute_reward(action, true_label, confidence=1.0, diversity_bonus=0.0):
    if true_label == -1: return 0.0
    base_reward = confidence * 1.0
    TP_REWARD, TN_REWARD, FN_PENALTY, FP_PENALTY = 5.0, 1.0, -3.0, -0.5
    if action == true_label:
        reward = base_reward * TP_REWARD if true_label == 1 else base_reward * TN_REWARD
    else:
        if true_label == 1 and action == 0: reward = base_reward * FN_PENALTY
        elif true_label == 0 and action == 1: reward = base_reward * FP_PENALTY
        else: reward = -base_reward
    reward += diversity_bonus
    return reward

def enhanced_warmup_with_multiple_methods(X_train, y_train, replay_buffer, agent, device):
    print("开始增强版预热...")
    labeled_mask = (y_train != -1)
    X_labeled, y_labeled = X_train[labeled_mask], y_train[labeled_mask]
    if len(X_labeled) == 0:
        print("警告: 没有已标注样本用于预热"); return
    print(f"预热样本数量: {len(X_labeled)}")
    iso_forest = IsolationForest(contamination=0.15, random_state=42, n_estimators=300)
    X_flat = X_labeled.reshape(len(X_labeled), -1)
    iso_forest.fit(X_flat)
    anomaly_scores = iso_forest.decision_function(X_flat)
    combined_scores = -anomaly_scores
    n_samples_to_add = min(len(X_labeled), 5000)
    sorted_indices = np.argsort(combined_scores)[::-1]
    n_abnormal_warmup, n_normal_warmup = int(0.2 * n_samples_to_add), int(0.5 * n_samples_to_add)
    abnormal_indices_warmup = sorted_indices[:n_abnormal_warmup]
    normal_indices_warmup = sorted_indices[-n_normal_warmup:]
    middle_indices_candidates = sorted_indices[n_abnormal_warmup:-n_normal_warmup]
    n_middle_warmup = min(len(middle_indices_candidates), n_samples_to_add - n_abnormal_warmup - n_normal_warmup)
    middle_indices_warmup = np.random.choice(middle_indices_candidates, size=n_middle_warmup, replace=False) if n_middle_warmup > 0 else []
    for indices_group, confidence_level in [(abnormal_indices_warmup, 0.95), (normal_indices_warmup, 0.85), (middle_indices_warmup, 0.6)]:
        for original_idx in indices_group:
            state_np, true_label = X_labeled[original_idx], y_labeled[original_idx]
            state = torch.FloatTensor(state_np).to(device)
            if confidence_level > 0.8: action = true_label
            elif confidence_level > 0.6: action = true_label if random.random() < 0.7 else 1 - true_label
            else: action = random.randint(0, 1)
            diversity_bonus = 0.1 if true_label == 1 else 0.0
            reward = enhanced_compute_reward(action, true_label, confidence_level, diversity_bonus)
            next_state_np = X_labeled[(original_idx + 1) % len(X_labeled)]
            next_state = torch.FloatTensor(next_state_np).to(device)
            replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), False)
    print(f"增强预热完成，回放池大小: {len(replay_buffer)}")

# =================================
# 4. 增强的训练函数
# =================================

def enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, 
                           gamma=0.99, batch_size=64, beta=0.4):
    if len(replay_buffer) < batch_size: return 0.0, 0.0
    
    sample_result = replay_buffer.sample(batch_size, beta)
    if sample_result is None: # MODIFIED: Handle None case from buffer sample
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
    all_predictions, all_labels, all_q_values_logits, all_probabilities_positive_class, all_features = [], [], [], [], []
    all_attention_weights = []
    with torch.no_grad():
        for states, labels in data_loader:
            states = states.to(device)
            labeled_mask = (labels != -1)
            if labeled_mask.sum() == 0: continue
            states_labeled, labels_labeled = states[labeled_mask], labels[labeled_mask]
            if states_labeled.size(0) == 0: continue
            
            q_values_logits, features_batch, attention_weights_batch = agent(states_labeled, return_features=True, return_attention_weights=True)
            
            predictions = q_values_logits.argmax(dim=1)
            probabilities = F.softmax(q_values_logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_labeled.cpu().numpy())
            all_q_values_logits.extend(q_values_logits.cpu().numpy())
            all_features.extend(features_batch.cpu().numpy())
            all_probabilities_positive_class.extend(probabilities[:, 1].cpu().numpy())
            all_attention_weights.extend(attention_weights_batch.cpu().numpy())
            
    agent.train()
    if len(all_predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "precision_per_class": [0.0, 0.0], 
                "recall_per_class": [0.0, 0.0], "f1_per_class": [0.0, 0.0], "predictions": [], 
                "labels": [], "q_values_logits": [], "probabilities_positive_class": [], "features": [],
                "attention_weights": []}
    all_predictions_np, all_labels_np = np.array(all_predictions), np.array(all_labels)
    precision_w = precision_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    recall_w = recall_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    f1_w = f1_score(all_labels_np, all_predictions_np, zero_division=0, average='weighted')
    
    # Ensure there are at least two classes for per-class metrics, or handle if only one class present in labels
    unique_labels = np.unique(all_labels_np)
    if len(unique_labels) < 2 and (0 in unique_labels or 1 in unique_labels): # Only one class present
        # Handle cases where only normal or only abnormal samples are in the batch
        # This might happen with small validation/test sets or highly imbalanced data
        if 0 in unique_labels: # Only normal
            precision_pc = np.array([precision_score(all_labels_np, all_predictions_np, pos_label=0, zero_division=0), 0.0])
            recall_pc = np.array([recall_score(all_labels_np, all_predictions_np, pos_label=0, zero_division=0), 0.0])
            f1_pc = np.array([f1_score(all_labels_np, all_predictions_np, pos_label=0, zero_division=0), 0.0])
        elif 1 in unique_labels: # Only abnormal
            precision_pc = np.array([0.0, precision_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)])
            recall_pc = np.array([0.0, recall_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)])
            f1_pc = np.array([0.0, f1_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)])
        else: # Should not happen if all_labels is not empty
            precision_pc, recall_pc, f1_pc = np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
    elif len(unique_labels) >= 2 :
         precision_pc = precision_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
         recall_pc = recall_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
         f1_pc = f1_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
    else: # No labels or unexpected labels
        precision_pc, recall_pc, f1_pc = np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])


    return {"precision": float(precision_w), "recall": float(recall_w), "f1": float(f1_w),
            "precision_per_class": [float(x) for x in precision_pc], 
            "recall_per_class": [float(x) for x in recall_pc], "f1_per_class": [float(x) for x in f1_pc],
            "predictions": all_predictions_np.tolist(), "labels": all_labels_np.tolist(), 
            "q_values_logits": np.array(all_q_values_logits).tolist(), 
            "probabilities_positive_class": np.array(all_probabilities_positive_class).tolist(), 
            "features": np.array(all_features).tolist(),
            "attention_weights": np.array(all_attention_weights).tolist()}
# =================================
# 5. 增强的主训练函数
# =================================

def enhanced_train_rlad(agent, target_agent, optimizer, scheduler, replay_buffer, 
                       X_train, y_train, X_val, y_val, device, 
                       num_episodes=150, target_update_freq=15,
                       epsilon_start=0.95, epsilon_end=0.02, epsilon_decay_rate=0.995,
                       batch_size_rl=64, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    training_history = {'episodes': [], 'train_loss': [], 'avg_td_error': [], 'val_f1': [], 
                        'val_precision': [], 'val_recall': [], 'epsilon': [], 
                        'replay_buffer_size': [], 'learning_rate': [], 'anomaly_f1': [], 'normal_f1': []}
    best_val_f1, best_anomaly_val_f1, patience, patience_counter = 0.0, 0.0, 40, 0
    print("开始增强的RLAD训练...")
    labeled_train_mask = (y_train != -1)
    X_train_labeled, y_train_labeled = X_train[labeled_train_mask], y_train[labeled_train_mask]
    if len(X_train_labeled) == 0: print("错误：训练集中没有已标记的样本。"); return training_history
    print(f"用于RL交互的已标记训练样本数量: {len(X_train_labeled)}")
    print(f"异常样本数量 (已标记): {np.sum(y_train_labeled == 1)}, 正常样本数量 (已标记): {np.sum(y_train_labeled == 0)}")
    epsilon = epsilon_start
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        episode_losses, episode_td_errors = [], []
        epsilon = max(epsilon_end, epsilon * epsilon_decay_rate)
        beta = min(1.0, 0.4 + episode * (1.0 - 0.4) / (num_episodes * 0.8))
        agent.train()
        shuffled_indices = np.random.permutation(len(X_train_labeled))
        for step_idx in range(len(X_train_labeled)):
            current_sample_idx_in_labeled = shuffled_indices[step_idx]
            state_np, true_label = X_train_labeled[current_sample_idx_in_labeled], y_train_labeled[current_sample_idx_in_labeled]
            state = torch.FloatTensor(state_np).to(device)
            action = agent.get_action(state, epsilon)
            reward = enhanced_compute_reward(action, true_label, 1.0, 0.1 if true_label == 1 else 0.0)
            next_state_np = X_train_labeled[shuffled_indices[(step_idx + 1) % len(X_train_labeled)]]
            next_state = torch.FloatTensor(next_state_np).to(device)
            replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), False)
            if len(replay_buffer) >= batch_size_rl * 2: # Ensure enough samples for a couple of batches
                loss, td_err = enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, batch_size=batch_size_rl, beta=beta)
                if loss is not None and loss > 0 : episode_losses.append(loss) # MODIFIED: check loss > 0
                if td_err is not None and td_err > 0: episode_td_errors.append(td_err) # MODIFIED: check td_err > 0
        if episode % target_update_freq == 0: target_agent.load_state_dict(agent.state_dict())
        scheduler.step()
        val_dataset = HydraulicDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        val_metrics = enhanced_evaluate_model(agent, val_loader, device)
        avg_loss, avg_td_error = np.mean(episode_losses) if episode_losses else 0.0, np.mean(episode_td_errors) if episode_td_errors else 0.0
        
        current_anomaly_f1 = val_metrics['f1_per_class'][1] if len(val_metrics['f1_per_class']) > 1 else 0.0
        current_normal_f1 = val_metrics['f1_per_class'][0] if len(val_metrics['f1_per_class']) > 0 else 0.0

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

        improved = False
        if current_anomaly_f1 > best_anomaly_val_f1: 
            best_anomaly_val_f1, best_val_f1, improved = current_anomaly_f1, val_metrics['f1'], True
        elif current_anomaly_f1 == best_anomaly_val_f1 and val_metrics['f1'] > best_val_f1: 
            best_val_f1, improved = val_metrics['f1'], True
        
        if improved:
            patience_counter = 0
            torch.save({'agent_state_dict': agent.state_dict(), 'target_agent_state_dict': target_agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                        'episode': episode, 'best_val_f1': best_val_f1, 'best_anomaly_val_f1': best_anomaly_val_f1,
                        'training_history_partial': convert_to_serializable(training_history)}, # MODIFIED: ensure serializable
                       os.path.join(output_dir, 'best_enhanced_rlad_model.pth'))
        else: 
            patience_counter += 1
        
        if patience_counter >= patience: print(f"早停在episode {episode}"); break
        if episode % 5 == 0 or episode == num_episodes -1:
            print(f"\nEpisode {episode}/{num_episodes}, Loss: {avg_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, Anomaly F1: {current_anomaly_f1:.4f}")
    
    print(f"\n训练完成！最佳验证集F1: {best_val_f1:.4f}, 最佳异常F1: {best_anomaly_val_f1:.4f}")
    return training_history

# =================================
# 6. 增强的可视化类
# =================================

class EnhancedRLADVisualizer:
    def __init__(self, output_dir="./output_visuals"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['axes.titleweight'] = 'bold'

    def _set_common_style(self, ax, title, xlabel, ylabel, fontsize=10, title_fontsize=12):
        ax.set_title(title, fontsize=title_fontsize, fontname='Times New Roman', fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontname='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontname='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=fontsize-1)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontname('Times New Roman')
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontname('Times New Roman')
                text.set_fontsize(fontsize -1)
        
    def plot_training_history(self, training_history):
        fig, axes = plt.subplots(3, 3, figsize=(22, 18))
        try:
            fig.suptitle('模型训练历史与性能指标', fontsize=16, fontweight='bold', fontname='SimHei')
            ax_fontname = 'SimHei'
        except:
            fig.suptitle('Model Training History and Performance Metrics', fontsize=16, fontweight='bold', fontname='Times New Roman')
            ax_fontname = 'Times New Roman'

        episodes = training_history['episodes']
        
        plot_configs = [
            (axes[0,0], training_history['train_loss'], '训练损失', 'Training Loss', 'Loss'),
            (axes[0,1], training_history['avg_td_error'], '平均TD误差', 'Average TD Error', 'TD Error'),
            (axes[0,2], training_history['val_f1'], '整体F1 (验证集)', 'Overall F1 (Validation)', 'F1 Score'),
            (axes[1,1], (training_history['val_precision'], training_history['val_recall']), 
             ('精确率 (验证集)', '召回率 (验证集)'), ('Precision (Validation)', 'Recall (Validation)'), 'Score', ['b-', 'r-']),
            (axes[1,2], training_history['epsilon'], '探索率(Epsilon)衰减', 'Epsilon Decay', 'Epsilon'),
            (axes[2,0], training_history['replay_buffer_size'], '经验回放池大小', 'Replay Buffer Size', 'Buffer Size'),
            (axes[2,1], training_history['learning_rate'], '学习率变化', 'Learning Rate', 'Learning Rate'),
        ]
        
        for ax, data, title_cn, title_en, ylabel, *line_styles in plot_configs:
            styles = line_styles[0] if line_styles else 'b-'
            current_title_tuple = (title_cn, title_en) if isinstance(title_cn, tuple) else ((title_cn,), (title_en,))
            
            current_labels = current_title_tuple[0] if ax_fontname == 'SimHei' else current_title_tuple[1]
            
            if isinstance(data, tuple): 
                ax.plot(episodes, data[0], styles[0], linewidth=1.5, label=current_labels[0])
                ax.plot(episodes, data[1], styles[1], linewidth=1.5, label=current_labels[1])
                ax_title_text = f'{current_labels[0]} 和 {current_labels[1]}' if ax_fontname == 'SimHei' else f'{current_labels[0]} and {current_labels[1]}'
                ax.set_title(ax_title_text, fontsize=12, fontname=ax_fontname)
            else:
                ax.plot(episodes, data, styles, linewidth=2, label=current_labels[0])
                ax.set_title(current_labels[0], fontsize=12, fontname=ax_fontname)
            
            ax.set_xlabel('Episode', fontname='Times New Roman', fontsize=10)
            ax.set_ylabel(ylabel, fontname='Times New Roman', fontsize=10)
            ax.grid(True, alpha=0.4); 
            leg = ax.legend(prop={'family': ax_fontname, 'size': 9})
            ax.tick_params(axis='both', labelsize=9)
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                 tick_label.set_fontname('Times New Roman')

        ax = axes[1,0]
        ax.plot(episodes, training_history['anomaly_f1'], 'r-', label='异常F1 (验证集)' if ax_fontname == 'SimHei' else 'Anomaly F1 (Validation)', linewidth=2)
        ax.plot(episodes, training_history['normal_f1'], 'c-', label='正常F1 (验证集)' if ax_fontname == 'SimHei' else 'Normal F1 (Validation)', linewidth=2)
        ax.set_title('分类别F1分数 (验证集)' if ax_fontname == 'SimHei' else 'Per-Class F1 Scores (Validation)', fontsize=12, fontname=ax_fontname)
        ax.set_xlabel('Episode', fontname='Times New Roman', fontsize=10); ax.set_ylabel('F1 Score', fontname='Times New Roman', fontsize=10)
        ax.grid(True, alpha=0.4); leg = ax.legend(prop={'family': ax_fontname, 'size': 9})
        ax.tick_params(axis='both', labelsize=9)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels(): tick_label.set_fontname('Times New Roman')

        ax = axes[2,2]
        ws = min(10, len(training_history['train_loss']) // 2) if len(training_history['train_loss']) > 10 else 1
        smoothed_loss = np.convolve(training_history['train_loss'], np.ones(ws)/ws, mode='valid') if ws > 1 else training_history['train_loss']
        plot_eps = episodes[ws-1:] if ws > 1 and len(episodes[ws-1:]) == len(smoothed_loss) else episodes
        
        label_text = f'平滑损失 (窗口={ws})' if ax_fontname == 'SimHei' else f'Smoothed Loss (Window={ws})'
        if len(plot_eps) == len(smoothed_loss): 
             ax.plot(plot_eps, smoothed_loss, 'navy', linewidth=2, label=label_text)
        else: 
             ax.plot(episodes, training_history['train_loss'], 'navy', linewidth=2, label='训练损失 (原始)' if ax_fontname == 'SimHei' else 'Training Loss (Raw)')
        
        ax.set_title('训练损失 (平滑)' if ax_fontname == 'SimHei' else 'Training Loss (Smoothed)', fontsize=12, fontname=ax_fontname)
        ax.set_xlabel('Episode', fontname='Times New Roman', fontsize=10); ax.set_ylabel('Smoothed Loss', fontname='Times New Roman', fontsize=10)
        ax.grid(True, alpha=0.4); leg = ax.legend(prop={'family': ax_fontname, 'size': 9})
        ax.tick_params(axis='both', labelsize=9)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels(): tick_label.set_fontname('Times New Roman')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'enhanced_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_confusion_matrix(self, y_true, y_pred, class_names=['正常', '异常']):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            is_chinese = any('\u4e00' <= char <= '\u9fff' for name in class_names for char in name)
            tick_fontname = 'SimHei' if is_chinese else 'Times New Roman'
        except: tick_fontname = 'Times New Roman'

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, 
                    annot_kws={"size": 14, "fontname": 'Times New Roman'}, ax=ax)
        
        title_text = '混淆矩阵' if tick_fontname == 'SimHei' else 'Confusion Matrix'
        xlabel_text = '预测标签' if tick_fontname == 'SimHei' else 'Predicted Label'
        ylabel_text = '真实标签' if tick_fontname == 'SimHei' else 'True Label'

        ax.set_title(title_text, fontsize=15, fontweight='bold', fontname=tick_fontname)
        ax.set_xlabel(xlabel_text, fontname=tick_fontname, fontsize=12)
        ax.set_ylabel(ylabel_text, fontname=tick_fontname, fontsize=12)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontname(tick_fontname); tick_label.set_fontsize(11)
        ax.grid(False)
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_roc_pr_curves(self, y_true, y_scores_positive_class):
        fpr, tpr, _ = roc_curve(y_true, y_scores_positive_class); roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_scores_positive_class); pr_auc = average_precision_score(y_true, y_scores_positive_class)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        try:
            fig.suptitle('ROC曲线 和 Precision-Recall曲线', fontsize=16, fontweight='bold', fontname='SimHei')
        except:
            fig.suptitle('ROC and Precision-Recall Curves', fontsize=16, fontweight='bold', fontname='Times New Roman')

        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        axes[0].plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
        self._set_common_style(axes[0], 'ROC Curve', 'False Positive Rate (FPR)', 'True Positive Rate (TPR)')
        axes[0].legend(loc="lower right", prop={'family':'Times New Roman', 'size':10})

        axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        self._set_common_style(axes[1], 'Precision-Recall Curve', 'Recall', 'Precision')
        axes[1].legend(loc="lower left", prop={'family':'Times New Roman', 'size':10})
        
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(os.path.join(self.output_dir, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_tsne_features(self, features_np, labels_np, class_names=['正常', '异常']):
        if len(features_np) == 0: print("没有特征可供t-SNE可视化。"); return
        print(f"开始t-SNE降维，特征数量: {len(features_np)}")
        tsne_perplexity = min(30, max(1, len(features_np)-2)) # Ensure perplexity < n_samples
        if tsne_perplexity <=0 : tsne_perplexity = 1
        
        features_2d = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, 
                           n_iter=300, init='pca', learning_rate='auto').fit_transform(features_np)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['blue', 'red']
        try:
            is_chinese = any('\u4e00' <= char <= '\u9fff' for name in class_names for char in name)
            legend_fontname = 'SimHei' if is_chinese else 'Times New Roman'
            title_text = 't-SNE 特征可视化 (测试集)' if is_chinese else 't-SNE Feature Visualization (Test Set)'
        except:
            legend_fontname = 'Times New Roman'
            title_text = 't-SNE Feature Visualization (Test Set)'

        for i, class_name_original in enumerate(class_names):
            mask = (labels_np == i)
            if np.sum(mask) > 0:
                 ax.scatter(features_2d[mask, 0], features_2d[mask, 1], c=colors[i], 
                            label=class_name_original, alpha=0.6, s=50)
        
        self._set_common_style(ax, title_text, 't-SNE Component 1', 't-SNE Component 2')
        leg = ax.legend(prop={'family': legend_fontname, 'size':10})
        plt.savefig(os.path.join(self.output_dir, 'tsne_features_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_3d_scatter_anomalies(self, all_X_data_np, all_labels_np, class_names=['正常', '异常']):
        # This function plots window-level data. If point-level 3D plot is needed, it requires different input.
        # For now, it uses X_test (windowed data) and y_test (window labels)
        if len(all_X_data_np) == 0 or len(all_labels_np) == 0: print("没有数据或标签可供3D散点图可视化。"); return
        if len(all_X_data_np) != len(all_labels_np): print("数据和标签数量不匹配。"); return
        print(f"开始生成3D异常散点图 (窗口级别)，样本数量: {len(all_X_data_np)}")
        
        # Since input_dim is now 1, all_X_data_np is (num_windows, window_size, 1)
        # We can use mean and variance of the single feature within the window.
        feature_mean = np.mean(all_X_data_np[:, :, 0], axis=1)
        feature_var = np.var(all_X_data_np[:, :, 0], axis=1)
        # For the third dimension, we can use a simple time index or another statistic if available.
        # Let's use the median as a third dimension for variety.
        feature_median = np.median(all_X_data_np[:, :, 0], axis=1)

        dim1, dim2, dim3 = feature_mean, feature_var, feature_median
        ax_labels = ['特征均值 (窗口内)', '特征方差 (窗口内)', '特征中位数 (窗口内)']
        plot_title = '3D 窗口特征散点图 (异常标记)'
        
        fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'red']; markers = ['o', '^']
        unique_labels_present = np.unique(all_labels_np)

        for class_label_val in np.sort(unique_labels_present):
            if class_label_val not in [0, 1]: continue # Only plot normal (0) and anomaly (1)
            class_name_idx = int(class_label_val)
            mask = (all_labels_np == class_label_val)
            if np.sum(mask) == 0: continue
            
            ax.scatter(dim1[mask], dim2[mask], dim3[mask], c=colors[class_name_idx], 
                       label=class_names[class_name_idx],
                       marker=markers[class_name_idx], s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
        
        ax.set_title(plot_title, fontsize=15, fontweight='bold', fontname='Times New Roman')
        ax.set_xlabel(ax_labels[0], fontname='Times New Roman', fontsize=10)
        ax.set_ylabel(ax_labels[1], fontname='Times New Roman', fontsize=10)
        ax.set_zlabel(ax_labels[2], fontname='Times New Roman', fontsize=10)
        
        try:
            is_chinese_class_names = any('\u4e00' <= char <= '\u9fff' for name in class_names for char in name)
            legend_font = 'SimHei' if is_chinese_class_names else 'Times New Roman'
        except: legend_font = 'Times New Roman'

        leg = ax.legend(prop={'family': legend_font, 'size':10})
        ax.grid(True, alpha=0.3); ax.view_init(elev=20., azim=-65)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            tick_label.set_fontname('Times New Roman'); tick_label.set_fontsize(9)

        plt.savefig(os.path.join(self.output_dir, '3d_scatter_window_anomalies.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_metric_and_anomaly_bar(self, episodes, metric_data, anomaly_data, metric_label, anomaly_label, threshold_value, output_filename="metric_anomaly_bar.png"):
        fig, ax1 = plt.subplots(figsize=(12, 7))
        color_metric = 'navy'
        ax1.plot(episodes, metric_data, color=color_metric, linewidth=1.2, label=metric_label)
        ax1.set_xlabel('Episode', fontsize=10, fontname='Times New Roman')
        ax1.set_ylabel(metric_label, color=color_metric, fontsize=10, fontname='Times New Roman')
        ax1.tick_params(axis='y', labelcolor=color_metric, labelsize=9)
        for tick_label in ax1.get_xticklabels() + ax1.get_yticklabels(): tick_label.set_fontname('Times New Roman')
        if threshold_value is not None:
            ax1.axhline(y=threshold_value, color='gray', linestyle='--', linewidth=1.2, label=f'{metric_label} Threshold')

        ax2 = ax1.twinx()
        color_anomaly = 'crimson'
        bar_positions = np.arange(len(episodes)) 
        ax2.bar(bar_positions, anomaly_data, color=color_anomaly, alpha=0.85, width=0.9, label=anomaly_label)
        ax2.set_ylabel(anomaly_label, color=color_anomaly, fontsize=10, fontname='Times New Roman')
        ax2.tick_params(axis='y', labelcolor=color_anomaly, labelsize=9)
        for tick_label in ax2.get_yticklabels(): tick_label.set_fontname('Times New Roman')

        if len(anomaly_data) > 0 :
             min_val, max_val = np.min(anomaly_data), np.max(anomaly_data)
             padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
             ax2.set_ylim(min(0, min_val - padding) , max_val + padding)
        if len(metric_data) > 0:
            min_val, max_val = np.min(metric_data), np.max(metric_data)
            padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
            ax1.set_ylim(min_val - padding, max_val + padding * 2)

        fig.suptitle('Metric Over Training with Anomaly Indication', fontsize=12, fontname='Times New Roman', fontweight='bold')
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        leg = ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=9, prop={'family':'Times New Roman'})
        ax1.grid(True, linestyle=':', alpha=0.6, color='gray')
        
        if len(episodes) > 10:
            step = max(1, len(episodes) // 10) # Ensure step is at least 1
            ax1.set_xticks(bar_positions[::step]); ax1.set_xticklabels(episodes[::step])
        else:
            ax1.set_xticks(bar_positions); ax1.set_xticklabels(episodes)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, output_filename), dpi=300, bbox_inches='tight')
        plt.close(fig); print(f"图 (a) 风格的 '{output_filename}' 已保存。")

    def plot_attention_heatmap(self, attention_scores, title="Attention Heatmap", output_filename="attention_heatmap.png", seq_len_for_ticks=None):
        if attention_scores is None or attention_scores.ndim != 2:
            print(f"无法绘制注意力热力图: attention_scores 无效 (shape: {attention_scores.shape if attention_scores is not None else 'None'})"); return
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_scores, cmap='viridis', vmin=0, vmax=np.max(attention_scores), 
                    cbar_kws={'label': 'Attention Score', 'shrink': 0.8}, ax=ax)
        
        if attention_scores.shape[0] > 25 and attention_scores.shape[1] > 25:
            rect_x, rect_y, rect_w, rect_h = 20, 20, 25, 25
            ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                       edgecolor='white', linestyle='--', linewidth=1.5, fill=False))
            ax.text(rect_x + 2 , rect_y - 2, 'Anomalous\nsubsequence', color='white', fontsize=9, 
                    fontname='Times New Roman', ha='left', va='bottom')

        ax.set_title(title, fontsize=12, fontname='Times New Roman', fontweight='bold')
        ax.set_xlabel('Key/Memory Sequence Position', fontsize=10, fontname='Times New Roman')
        ax.set_ylabel('Query Sequence Position', fontsize=10, fontname='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=9)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels(): tick_label.set_fontname('Times New Roman')
        
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_fontname('Times New Roman'); cbar.ax.yaxis.label.set_size(10)
        for tick_label in cbar.ax.get_yticklabels(): tick_label.set_fontname('Times New Roman'); tick_label.set_fontsize(9)
        ax.grid(False)
        
        if seq_len_for_ticks is not None and isinstance(seq_len_for_ticks, int) and seq_len_for_ticks > 0:
            num_ticks = min(10, seq_len_for_ticks)
            tick_positions = np.linspace(0, attention_scores.shape[0] -1 , num_ticks, dtype=int)
            tick_labels = np.linspace(0, seq_len_for_ticks -1, num_ticks, dtype=int)
            ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename), dpi=300, bbox_inches='tight')
        plt.close(fig); print(f"图 (b) 风格的 '{output_filename}' 已保存。")

# =================================
# 7. 主函数
# =================================

def parse_args():
    parser = argparse.ArgumentParser(description='增强版RLAD液压支架异常检测')
    parser.add_argument('--data_path', type=str, default="C:/Users/Liu HaoTian/Desktop/Python files/deeplearning/example/timeseries/examples/RLAD/clean_data.csv", help='数据文件路径')
    parser.add_argument('--window_size', type=int, default=288, help='滑动窗口大小') # MODIFIED
    parser.add_argument('--stride', type=int, default=12, help='滑动窗口步长') # Consider adjusting if window_size changed drastically
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--num_episodes', type=int, default=100, help='训练轮数')
    parser.add_argument('--output_dir', type=str, default="C:/Users/Liu HaoTian/Desktop/Python files/deeplearning/example/timeseries/examples/RLAD/output_modified", help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default="auto", help='设备选择 (auto, cpu, cuda)')
    parser.add_argument('--batch_size_rl', type=int, default=64, help='强化学习训练批次大小')
    parser.add_argument('--target_update_freq', type=int, default=10, help='目标网络更新频率')
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99, help='Epsilon指数衰减率')
    return parser.parse_args()

def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device != "cuda" else "cuda")
    print(f"使用设备: {device}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_timestamped = os.path.join(args.output_dir, f"enhanced_rlad_results_single_feature_{timestamp}")
    os.makedirs(output_dir_timestamped, exist_ok=True); print(f"所有输出将保存到: {output_dir_timestamped}")
    config_to_save = convert_to_serializable(vars(args)); config_to_save['output_dir_actual'] = output_dir_timestamped
    with open(os.path.join(output_dir_timestamped, 'config.json'), 'w', encoding='utf-8') as f: json.dump(config_to_save, f, ensure_ascii=False, indent=4)
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, \
        test_window_global_start_indices, df_full_data_with_selected_col, selected_col_name = \
            load_hydraulic_data_improved(
                data_path=args.data_path, window_size=args.window_size, stride=args.stride
            )
        
        input_dim = X_train.shape[-1] 
        print(f"模型将使用的输入维度 (特征数): {input_dim}") # Should be 1

        agent = EnhancedRLADAgent(input_dim=input_dim, seq_len=args.window_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
        target_agent = EnhancedRLADAgent(input_dim=input_dim, seq_len=args.window_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
        target_agent.load_state_dict(agent.state_dict()); target_agent.eval()
        optimizer = optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(10, args.num_episodes // 5), T_mult=1, eta_min=1e-7)
        replay_buffer = PrioritizedReplayBuffer(capacity=50000, alpha=0.6)
        print(f"模型参数数量: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")
        
        enhanced_warmup_with_multiple_methods(X_train, y_train, replay_buffer, agent, device)
        
        training_history = enhanced_train_rlad(agent, target_agent, optimizer, scheduler, replay_buffer, X_train, y_train, X_val, y_val, device,
                                               num_episodes=args.num_episodes, target_update_freq=args.target_update_freq,
                                               epsilon_decay_rate=args.epsilon_decay_rate, batch_size_rl=args.batch_size_rl, output_dir=output_dir_timestamped)
        with open(os.path.join(output_dir_timestamped, 'training_history_final.json'), 'w', encoding='utf-8') as f: json.dump(convert_to_serializable(training_history), f, ensure_ascii=False, indent=4)
        
        visualizer = EnhancedRLADVisualizer(output_dir_timestamped)
        if training_history['episodes']: 
            visualizer.plot_training_history(training_history)
            if 'avg_td_error' in training_history and 'anomaly_f1' in training_history:
                visualizer.plot_metric_and_anomaly_bar(
                    episodes=training_history['episodes'], metric_data=training_history['avg_td_error'],
                    anomaly_data=training_history['anomaly_f1'], metric_label='Average TD Error',
                    anomaly_label='Anomaly F1 (Validation)', threshold_value=0.1,
                    output_filename='td_error_and_anomaly_f1_plot.png')
        
        best_model_path = os.path.join(output_dir_timestamped, 'best_enhanced_rlad_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            agent.load_state_dict(checkpoint['agent_state_dict'])
            print(f"\n已加载最佳模型 (Episode {checkpoint.get('episode', 'N/A')}), 最佳验证集F1: {checkpoint.get('best_val_f1',0.0):.4f}, 异常F1: {checkpoint.get('best_anomaly_val_f1',0.0):.4f}")
        
        test_dataset = HydraulicDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_metrics = enhanced_evaluate_model(agent, test_loader, device) # Window-level metrics
        print(f"\n最终测试集评估结果 (窗口级别): 精确率={test_metrics['precision']:.4f}, 召回率={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
        if len(test_metrics['f1_per_class']) >= 2: print(f"  正常类F1: {test_metrics['f1_per_class'][0]:.4f}, 异常类F1: {test_metrics['f1_per_class'][1]:.4f}")
        with open(os.path.join(output_dir_timestamped, 'test_results_window_level.json'), 'w', encoding='utf-8') as f: json.dump(convert_to_serializable(test_metrics), f, ensure_ascii=False, indent=4)

        # MODIFIED: Point-level anomaly mapping and extraction
        point_level_anomalies = np.zeros(len(df_full_data_with_selected_col), dtype=int)
        
        # Get predictions only for windows that were actually evaluated (had labels != -1)
        predictions_for_labeled_windows = np.array(test_metrics.get('predictions', []))

        print(f"\n进行逐点异常标记...")
        print(f"原始X_test中窗口总数: {len(X_test)}")
        print(f"已标记的测试窗口的预测数量: {len(predictions_for_labeled_windows)}")
        print(f"y_test中已标记窗口数量: {np.sum(y_test != -1)}")
        print(f"test_window_global_start_indices 长度: {len(test_window_global_start_indices)}")

        labeled_pred_idx = 0 # Index for predictions_for_labeled_windows
        for i in range(len(X_test)): # Iterate through ALL original test windows
            if y_test[i] != -1: # Check if this original test window was labeled (and thus has a prediction)
                if labeled_pred_idx < len(predictions_for_labeled_windows):
                    prediction_for_this_window = predictions_for_labeled_windows[labeled_pred_idx]
                    if prediction_for_this_window == 1: # If this labeled window is predicted anomalous
                        start_original_data_idx = test_window_global_start_indices[i]
                        for point_offset in range(args.window_size):
                            actual_idx_in_full_data = start_original_data_idx + point_offset
                            if actual_idx_in_full_data < len(point_level_anomalies):
                                point_level_anomalies[actual_idx_in_full_data] = 1
                    labeled_pred_idx += 1
                else:
                    # This case implies a mismatch if the number of predictions doesn't align with labeled y_test items.
                    # It should ideally not be reached if data processing is consistent.
                    print(f"警告: 在处理X_test窗口索引 {i} 时，预期有预测结果，但已用尽所有预测。")
            # Else (y_test[i] == -1), this window was unlabeled, so it has no prediction from test_metrics. Skip.
        
        df_full_data_with_selected_col['is_anomaly_predicted'] = point_level_anomalies
        
        extracted_abnormal_df = df_full_data_with_selected_col[df_full_data_with_selected_col['is_anomaly_predicted'] == 1]
        print(f"\n提取出的异常数据点数量: {len(extracted_abnormal_df)}")
        if not extracted_abnormal_df.empty:
            print("提取出的异常数据点 (前5条):")
            print(extracted_abnormal_df.head())
        extracted_abnormal_df.to_csv(os.path.join(output_dir_timestamped, 'extracted_abnormal_data_points.csv'), index=False, encoding='utf-8-sig')

        print("\n所有数据点及其预测的异常标签 (前5条):")
        print(df_full_data_with_selected_col.head())
        df_full_data_with_selected_col.to_csv(os.path.join(output_dir_timestamped, 'all_data_with_point_predictions.csv'), index=False, encoding='utf-8-sig')


        if test_metrics['labels']: # These are window-level labels and predictions
            visualizer.plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
            if test_metrics['probabilities_positive_class']: visualizer.plot_roc_pr_curves(test_metrics['labels'], test_metrics['probabilities_positive_class'])
            if test_metrics['features']: visualizer.plot_tsne_features(np.array(test_metrics['features']), np.array(test_metrics['labels']))

            if test_metrics.get('attention_weights') and len(test_metrics['attention_weights']) > 0:
                sample_attention_weights = np.array(test_metrics['attention_weights'][0])
                if sample_attention_weights.ndim == 3: sample_attention_weights = np.mean(sample_attention_weights, axis=0)
                if sample_attention_weights.ndim == 2:
                     actual_seq_len = X_test.shape[1]
                     visualizer.plot_attention_heatmap(sample_attention_weights, title="Attention Heatmap (Test Sample 0)",
                        output_filename="attention_heatmap_sample0.png", seq_len_for_ticks=actual_seq_len)
                     if 1 in test_metrics['labels']:
                        try:
                            first_anomaly_idx = test_metrics['labels'].index(1)
                            if first_anomaly_idx < len(test_metrics['attention_weights']):
                                anomaly_attn_weights = np.array(test_metrics['attention_weights'][first_anomaly_idx])
                                if anomaly_attn_weights.ndim == 3: anomaly_attn_weights = np.mean(anomaly_attn_weights, axis=0)
                                if anomaly_attn_weights.ndim == 2:
                                    visualizer.plot_attention_heatmap(anomaly_attn_weights, title=f"Attention Heatmap (First Anomaly Sample {first_anomaly_idx})",
                                        output_filename=f"attention_heatmap_anomaly_sample{first_anomaly_idx}.png", seq_len_for_ticks=actual_seq_len)
                        except ValueError: print("测试集中未找到异常样本标签为1的样本。")
                        except Exception as e_attn: print(f"绘制异常样本注意力图时出错: {e_attn}")

        # The 3D scatter plot uses X_test (windowed data) and y_test (window-level labels)
        if X_test is not None and y_test is not None and len(X_test) > 0:
            mask_labeled_test_windows = (y_test != -1)
            X_test_for_3d = X_test[mask_labeled_test_windows]
            y_test_labels_for_3d = y_test[mask_labeled_test_windows]
            if len(X_test_for_3d) > 0:
                visualizer.plot_3d_scatter_anomalies(X_test_for_3d, y_test_labels_for_3d)
            else:
                print("测试集中没有已标记的窗口可用于3D散点图可视化。")

        print(f"\n增强版RLAD训练和评估完成！所有结果保存在: {output_dir_timestamped}")
    except Exception as e:
        print(f"发生严重错误: {str(e)}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()