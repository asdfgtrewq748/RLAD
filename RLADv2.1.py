"""
改进版RLAD: 基于强化学习与主动学习的时间序列异常检测
专用于液压支架工作阻力异常检测 - 修复与增强版本 (使用单一特征并实现逐点异常标记)
修改：初步异常检测逻辑基于 yichang.py 中的“来压判据”
新增：交互式选择支架号功能
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
import traceback # 用于打印详细错误信息

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import IsolationForest # 保留以备将来使用或比较
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文字体显示
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Times New Roman'] # Visualizer class handles this
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'Times New Roman'

def set_seed(seed=42):
    """设置随机种子保证可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    else:
        try: # 尝试基本类型转换
            return str(obj) if not isinstance(obj, (int, float, bool, str, type(None))) else obj
        except Exception:
            return f"Unserializable object: {type(obj)}"


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

def load_hydraulic_data_improved(data_path, window_size=288, stride=12, specific_feature_column=None):
    """
    改进的数据加载函数
    初步异常标签基于选定列数据的“来压判据” (Q3 + 0.5 * IQR)
    新增: specific_feature_column 参数用于指定加载的特征列
    返回: ... , actual_selected_column_name (实际使用的列名)
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
    if data_values.size == 0:
        raise ValueError(f"未能从选定列 '{actual_selected_column_name}' 提取任何数据点。")
    print(f"用于阈值计算的 data_values (来自列 '{actual_selected_column_name}', 前5个): \n{data_values[:5]}")
    print(f"用于阈值计算的 data_values (来自列 '{actual_selected_column_name}') 的最大值: {np.max(data_values) if data_values.size > 0 else 'N/A'}, 最小值: {np.min(data_values) if data_values.size > 0 else 'N/A'}")

    # --- 基于选定列计算“来压判据”阈值 ---
    all_points_for_thresholds = data_values.flatten()
    laiya_criterion_point = np.inf 

    if len(all_points_for_thresholds) > 0:
        q1_point = np.percentile(all_points_for_thresholds, 25)
        q3_point = np.percentile(all_points_for_thresholds, 75)
        iqr_point = q3_point - q1_point
        
        # 计算来压判据阈值 (Q3 + 0.5 * IQR)
        laiya_criterion_point = q3_point + 0.5 * iqr_point 
        
        print(f"逐点异常阈值计算 (基于列 '{actual_selected_column_name}'):")
        print(f"  数据点总数: {len(all_points_for_thresholds)}")
        print(f"  数据点最小值: {np.min(all_points_for_thresholds):.2f}, 最大值: {np.max(all_points_for_thresholds):.2f}")
        print(f"  Q1={q1_point:.2f}, Q3={q3_point:.2f}, IQR={iqr_point:.2f}")
        print(f"  计算得到的 来压判据阈值 (point-level): {laiya_criterion_point:.2f}")
    else:
        print("警告: 没有有效数据点用于计算“来压判据”阈值。异常标签可能不准确。")
    # --- 阈值计算结束 ---

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    print("创建滑动窗口...")
    windows_scaled_list = []
    windows_raw_max_list = [] 
    window_start_indices_all = []

    for i in range(0, len(data_scaled) - window_size + 1, stride):
        window_s = data_scaled[i:i + window_size]
        windows_scaled_list.append(window_s)
        
        window_r_points = data_values[i:i + window_size].flatten()
        if len(window_r_points) > 0:
            windows_raw_max_list.append(np.max(window_r_points))
        else:
            windows_raw_max_list.append(-np.inf) 
            
        window_start_indices_all.append(i)
    
    if not windows_scaled_list:
        raise ValueError(f"未能创建滑动窗口。数据长度 {len(data_scaled)}, 窗口大小 {window_size}。请检查参数。")

    X = np.array(windows_scaled_list)
    windows_raw_max_np = np.array(windows_raw_max_list)
    window_start_indices_all_np = np.array(window_start_indices_all)
    print(f"窗口数据形状 (scaled for model): {X.shape}")
    if windows_raw_max_np.size > 0:
        print(f"所有窗口原始数据最大值中的最大值: {np.max(windows_raw_max_np):.2f}")
        print(f"所有窗口原始数据最大值中的最小值: {np.min(windows_raw_max_np):.2f}")
        print(f"所有窗口原始数据最大值中的均值: {np.mean(windows_raw_max_np):.2f}")
    
    N = len(X)
    if N == 0:
        raise ValueError("未能创建滑动窗口。请检查 window_size, stride, 和数据长度。")
    
    # --- 基于“来压判据”的打标签逻辑 ---
    y = np.zeros(N, dtype=int) 
    num_anomalies_found_by_laiya = 0 
    for i in range(N):
        max_val_in_raw_window = windows_raw_max_np[i]
        if max_val_in_raw_window > laiya_criterion_point: 
            y[i] = 1 
            num_anomalies_found_by_laiya +=1
    
    print(f"初步窗口标签统计 (在设为-1之前):")
    print(f"  通过 来压判据阈值 ({laiya_criterion_point:.2f}) 标记的异常窗口数: {num_anomalies_found_by_laiya}")
    print(f"  总计初步异常窗口数 (y==1): {np.sum(y==1)}")
    print(f"  总计初步正常窗口数 (y==0): {np.sum(y==0)}")

    unlabeled_mask = np.random.random(N) < 0.03 
    y[unlabeled_mask] = -1 
    
    print(f"基于“来压判据”的标签分布 (在设为-1之后): 正常={np.sum(y==0)}, 异常={np.sum(y==1)}, 未标注={np.sum(y==-1)}")
    
    train_size = int(0.7 * N)
    val_size = int(0.15 * N)
    
    indices = np.arange(N)
    np.random.shuffle(indices)

    X_train = X[indices[:train_size]]
    y_train = y[indices[:train_size]]
    print(f"划分到训练集 y_train 中的标签分布: 正常={np.sum(y_train==0)}, 异常={np.sum(y_train==1)}, 未标注={np.sum(y_train==-1)}")

    X_val = X[indices[train_size:train_size + val_size]]
    y_val = y[indices[train_size:train_size + val_size]]
    X_test = X[indices[train_size + val_size:]]
    y_test = y[indices[train_size + val_size:]]

    test_window_original_indices = window_start_indices_all_np[indices[train_size + val_size:]]
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, \
           test_window_original_indices, df_for_point_mapping, actual_selected_column_name

# =================================
# 2. 优化的网络结构
# =================================

class EnhancedRLADAgent(nn.Module):
    """
    增强版RLAD智能体网络架构
    添加残差连接和注意力机制
    """
    
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
        self.fc3 = nn.Linear(hidden_size // 2, 2) # 输出Q值 (正常, 异常)
        
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
# ... (第一部分代码结束) ...

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

        if probs_sum == 0 or len(self.buffer) == 0:
            if len(self.buffer) == 0: return None
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
    
    n_samples_to_add = min(len(X_labeled), 5000) 
    
    anomaly_indices = np.where(y_labeled == 1)[0]
    normal_indices = np.where(y_labeled == 0)[0]
    
    np.random.shuffle(anomaly_indices)
    np.random.shuffle(normal_indices)
    
    num_anomalies_to_add = min(len(anomaly_indices), int(n_samples_to_add * 0.4)) 
    num_normals_to_add = min(len(normal_indices), n_samples_to_add - num_anomalies_to_add)
    
    selected_indices = []
    confidences = []

    for idx in anomaly_indices[:num_anomalies_to_add]:
        selected_indices.append(idx)
        confidences.append(0.95) 
        
    for idx in normal_indices[:num_normals_to_add]:
        selected_indices.append(idx)
        confidences.append(0.85) 

    print(f"预热将添加 {len(selected_indices)} 个样本 (异常: {num_anomalies_to_add}, 正常: {num_normals_to_add})")

    for i, original_idx in enumerate(selected_indices):
        state_np, true_label = X_labeled[original_idx], y_labeled[original_idx]
        confidence_level = confidences[i]
        state = torch.FloatTensor(state_np).to(device)
        action = true_label 
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
    
    unique_labels = np.unique(all_labels_np)
    if len(unique_labels) < 2 and (0 in unique_labels or 1 in unique_labels): 
        if 0 in unique_labels: 
            precision_pc = np.array([precision_score(all_labels_np, all_predictions_np, pos_label=0, zero_division=0), 0.0])
            recall_pc = np.array([recall_score(all_labels_np, all_predictions_np, pos_label=0, zero_division=0), 0.0])
            f1_pc = np.array([f1_score(all_labels_np, all_predictions_np, pos_label=0, zero_division=0), 0.0])
        elif 1 in unique_labels: 
            precision_pc = np.array([0.0, precision_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)])
            recall_pc = np.array([0.0, recall_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)])
            f1_pc = np.array([0.0, f1_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)])
        else: 
            precision_pc, recall_pc, f1_pc = np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
    elif len(unique_labels) >= 2 :
         precision_pc = precision_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
         recall_pc = recall_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
         f1_pc = f1_score(all_labels_np, all_predictions_np, zero_division=0, average=None, labels=[0,1])
    else: 
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
            if len(replay_buffer) >= batch_size_rl * 2: 
                loss, td_err = enhanced_train_dqn_step(agent, target_agent, replay_buffer, optimizer, device, batch_size=batch_size_rl, beta=beta)
                if loss is not None and loss > 0 : episode_losses.append(loss)
                if td_err is not None and td_err > 0: episode_td_errors.append(td_err)
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
                        'training_history_partial': convert_to_serializable(training_history)},
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
        # Set fonts here to ensure they are applied for all plots by this visualizer
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'Times New Roman' # Default to Times New Roman if SimHei not found
            # Test if SimHei is available, otherwise fallback
            fig_test_font = plt.figure()
            plt.title("测试", fontname='SimHei') # Test with SimHei
            plt.close(fig_test_font)
            print("SimHei font found and set for visualizer.")
        except Exception:
            plt.rcParams['font.family'] = 'Times New Roman'
            print("SimHei font not found, using Times New Roman for visualizer.")

        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['axes.titleweight'] = 'bold'


    def _set_common_style(self, ax, title, xlabel, ylabel, fontsize=10, title_fontsize=12):
        ax.set_title(title, fontsize=title_fontsize, fontname=plt.rcParams['font.family'], fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontname=plt.rcParams['font.family'])
        ax.set_ylabel(ylabel, fontsize=fontsize, fontname=plt.rcParams['font.family'])
        ax.tick_params(axis='both', which='major', labelsize=fontsize-1)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontname(plt.rcParams['font.family'])
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontname(plt.rcParams['font.family'])
                text.set_fontsize(fontsize -1)
        
    def plot_training_history(self, training_history):
        fig, axes = plt.subplots(3, 3, figsize=(22, 18))
        
        # Determine if SimHei is effectively being used for titles
        ax_fontname = 'SimHei' if 'SimHei' in plt.rcParams['font.sans-serif'] else plt.rcParams['font.family']

        fig.suptitle('模型训练历史与性能指标' if ax_fontname == 'SimHei' else 'Model Training History and Performance Metrics', 
                     fontsize=16, fontweight='bold', fontname=ax_fontname)

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
            
            ax.set_xlabel('Episode', fontname=plt.rcParams['font.family'], fontsize=10)
            ax.set_ylabel(ylabel, fontname=plt.rcParams['font.family'], fontsize=10)
            ax.grid(True, alpha=0.4); 
            leg = ax.legend(prop={'family': ax_fontname if ax_fontname == 'SimHei' else plt.rcParams['font.family'], 'size': 9})
            ax.tick_params(axis='both', labelsize=9)
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                 tick_label.set_fontname(plt.rcParams['font.family'])

        ax = axes[1,0]
        ax.plot(episodes, training_history['anomaly_f1'], 'r-', label='异常F1 (验证集)' if ax_fontname == 'SimHei' else 'Anomaly F1 (Validation)', linewidth=2)
        ax.plot(episodes, training_history['normal_f1'], 'c-', label='正常F1 (验证集)' if ax_fontname == 'SimHei' else 'Normal F1 (Validation)', linewidth=2)
        ax.set_title('分类别F1分数 (验证集)' if ax_fontname == 'SimHei' else 'Per-Class F1 Scores (Validation)', fontsize=12, fontname=ax_fontname)
        ax.set_xlabel('Episode', fontname=plt.rcParams['font.family'], fontsize=10); ax.set_ylabel('F1 Score', fontname=plt.rcParams['font.family'], fontsize=10)
        ax.grid(True, alpha=0.4); leg = ax.legend(prop={'family': ax_fontname if ax_fontname == 'SimHei' else plt.rcParams['font.family'], 'size': 9})
        ax.tick_params(axis='both', labelsize=9)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels(): tick_label.set_fontname(plt.rcParams['font.family'])

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
        ax.set_xlabel('Episode', fontname=plt.rcParams['font.family'], fontsize=10); ax.set_ylabel('Smoothed Loss', fontname=plt.rcParams['font.family'], fontsize=10)
        ax.grid(True, alpha=0.4); leg = ax.legend(prop={'family': ax_fontname if ax_fontname == 'SimHei' else plt.rcParams['font.family'], 'size': 9})
        ax.tick_params(axis='both', labelsize=9)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels(): tick_label.set_fontname(plt.rcParams['font.family'])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'enhanced_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_confusion_matrix(self, y_true, y_pred, class_names=['正常', '异常']):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(8, 6))
        
        is_chinese = any('\u4e00' <= char <= '\u9fff' for name in class_names for char in name) and 'SimHei' in plt.rcParams['font.sans-serif']
        tick_fontname = 'SimHei' if is_chinese else plt.rcParams['font.family']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, 
                    annot_kws={"size": 14, "fontname": plt.rcParams['font.family']}, ax=ax)
        
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
        
        title_fontname = 'SimHei' if 'SimHei' in plt.rcParams['font.sans-serif'] else plt.rcParams['font.family']
        fig.suptitle('ROC曲线 和 Precision-Recall曲线' if title_fontname == 'SimHei' else 'ROC and Precision-Recall Curves', 
                     fontsize=16, fontweight='bold', fontname=title_fontname)

        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        axes[0].plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
        self._set_common_style(axes[0], 'ROC Curve', 'False Positive Rate (FPR)', 'True Positive Rate (TPR)')
        axes[0].legend(loc="lower right", prop={'family':plt.rcParams['font.family'], 'size':10})

        axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        self._set_common_style(axes[1], 'Precision-Recall Curve', 'Recall', 'Precision')
        axes[1].legend(loc="lower left", prop={'family':plt.rcParams['font.family'], 'size':10})
        
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(os.path.join(self.output_dir, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_tsne_features(self, features_np, labels_np, class_names=['正常', '异常']):
        if len(features_np) == 0: print("没有特征可供t-SNE可视化。"); return
        print(f"开始t-SNE降维，特征数量: {len(features_np)}")
        tsne_perplexity = min(30, max(1, len(features_np)-2)) 
        if tsne_perplexity <=0 : tsne_perplexity = 1
        
        features_2d = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, 
                           n_iter=300, init='pca', learning_rate='auto').fit_transform(features_np)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['blue', 'red']
        
        is_chinese = any('\u4e00' <= char <= '\u9fff' for name in class_names for char in name) and 'SimHei' in plt.rcParams['font.sans-serif']
        legend_fontname = 'SimHei' if is_chinese else plt.rcParams['font.family']
        title_text = 't-SNE 特征可视化 (测试集)' if is_chinese else 't-SNE Feature Visualization (Test Set)'
        
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
        if len(all_X_data_np) == 0 or len(all_labels_np) == 0: print("没有数据或标签可供3D散点图可视化。"); return
        if len(all_X_data_np) != len(all_labels_np): print("数据和标签数量不匹配。"); return
        print(f"开始生成3D异常散点图 (窗口级别)，样本数量: {len(all_X_data_np)}")
        
        feature_mean = np.mean(all_X_data_np[:, :, 0], axis=1)
        feature_var = np.var(all_X_data_np[:, :, 0], axis=1)
        feature_median = np.median(all_X_data_np[:, :, 0], axis=1)

        dim1, dim2, dim3 = feature_mean, feature_var, feature_median
        
        is_chinese_class_names = any('\u4e00' <= char <= '\u9fff' for name in class_names for char in name) and 'SimHei' in plt.rcParams['font.sans-serif']
        ax_labels_cn = ['特征均值 (窗口内)', '特征方差 (窗口内)', '特征中位数 (窗口内)']
        ax_labels_en = ['Feature Mean (Window)', 'Feature Variance (Window)', 'Feature Median (Window)']
        ax_labels = ax_labels_cn if is_chinese_class_names else ax_labels_en
        plot_title = '3D 窗口特征散点图 (异常标记)' if is_chinese_class_names else '3D Window Feature Scatter Plot (Anomalies)'
        
        fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'red']; markers = ['o', '^']
        unique_labels_present = np.unique(all_labels_np)

        for class_label_val in np.sort(unique_labels_present):
            if class_label_val not in [0, 1]: continue 
            class_name_idx = int(class_label_val)
            mask = (all_labels_np == class_label_val)
            if np.sum(mask) == 0: continue
            
            ax.scatter(dim1[mask], dim2[mask], dim3[mask], c=colors[class_name_idx], 
                       label=class_names[class_name_idx],
                       marker=markers[class_name_idx], s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
        
        ax.set_title(plot_title, fontsize=15, fontweight='bold', fontname=plt.rcParams['font.family'])
        ax.set_xlabel(ax_labels[0], fontname=plt.rcParams['font.family'], fontsize=10)
        ax.set_ylabel(ax_labels[1], fontname=plt.rcParams['font.family'], fontsize=10)
        ax.set_zlabel(ax_labels[2], fontname=plt.rcParams['font.family'], fontsize=10)
        
        legend_font = 'SimHei' if is_chinese_class_names else plt.rcParams['font.family']

        leg = ax.legend(prop={'family': legend_font, 'size':10})
        ax.grid(True, alpha=0.3); ax.view_init(elev=20., azim=-65)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            tick_label.set_fontname(plt.rcParams['font.family']); tick_label.set_fontsize(9)

        plt.savefig(os.path.join(self.output_dir, '3d_scatter_window_anomalies.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_metric_and_anomaly_bar(self, episodes, metric_data, anomaly_data, metric_label, anomaly_label, threshold_value, output_filename="metric_anomaly_bar.png"):
        fig, ax1 = plt.subplots(figsize=(12, 7))
        color_metric = 'navy'
        ax1.plot(episodes, metric_data, color=color_metric, linewidth=1.2, label=metric_label)
        ax1.set_xlabel('Episode', fontsize=10, fontname=plt.rcParams['font.family'])
        ax1.set_ylabel(metric_label, color=color_metric, fontsize=10, fontname=plt.rcParams['font.family'])
        ax1.tick_params(axis='y', labelcolor=color_metric, labelsize=9)
        for tick_label in ax1.get_xticklabels() + ax1.get_yticklabels(): tick_label.set_fontname(plt.rcParams['font.family'])
        if threshold_value is not None:
            ax1.axhline(y=threshold_value, color='gray', linestyle='--', linewidth=1.2, label=f'{metric_label} Threshold')

        ax2 = ax1.twinx()
        color_anomaly = 'crimson'
        bar_positions = np.arange(len(episodes)) 
        ax2.bar(bar_positions, anomaly_data, color=color_anomaly, alpha=0.85, width=0.9, label=anomaly_label)
        ax2.set_ylabel(anomaly_label, color=color_anomaly, fontsize=10, fontname=plt.rcParams['font.family'])
        ax2.tick_params(axis='y', labelcolor=color_anomaly, labelsize=9)
        for tick_label in ax2.get_yticklabels(): tick_label.set_fontname(plt.rcParams['font.family'])

        if len(anomaly_data) > 0 :
             min_val, max_val = np.min(anomaly_data), np.max(anomaly_data)
             padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
             ax2.set_ylim(min(0, min_val - padding) , max_val + padding)
        if len(metric_data) > 0:
            min_val, max_val = np.min(metric_data), np.max(metric_data)
            padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
            ax1.set_ylim(min_val - padding, max_val + padding * 2)

        fig.suptitle('Metric Over Training with Anomaly Indication', fontsize=12, fontname=plt.rcParams['font.family'], fontweight='bold')
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        leg = ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=9, prop={'family':plt.rcParams['font.family']})
        ax1.grid(True, linestyle=':', alpha=0.6, color='gray')
        
        if len(episodes) > 10:
            step = max(1, len(episodes) // 10) 
            ax1.set_xticks(bar_positions[::step]); ax1.set_xticklabels(episodes[::step])
        else:
            ax1.set_xticks(bar_positions); ax1.set_xticklabels(episodes)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, output_filename), dpi=300, bbox_inches='tight')
        plt.close(fig); print(f"图 '{output_filename}' 已保存。")

    def plot_attention_heatmap(self, attention_scores, title="Attention Heatmap", output_filename="attention_heatmap.png", seq_len_for_ticks=None):
        if attention_scores is None or attention_scores.ndim != 2:
            print(f"无法绘制注意力热力图: attention_scores 无效 (shape: {attention_scores.shape if attention_scores is not None else 'None'})"); return
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_scores, cmap='viridis', vmin=0, vmax=np.max(attention_scores), 
                    cbar_kws={'label': 'Attention Score', 'shrink': 0.8}, ax=ax)
        
        ax.set_title(title, fontsize=12, fontname=plt.rcParams['font.family'], fontweight='bold')
        ax.set_xlabel('Key/Memory Sequence Position', fontsize=10, fontname=plt.rcParams['font.family'])
        ax.set_ylabel('Query Sequence Position', fontsize=10, fontname=plt.rcParams['font.family'])
        ax.tick_params(axis='both', which='major', labelsize=9)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels(): tick_label.set_fontname(plt.rcParams['font.family'])
        
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_fontname(plt.rcParams['font.family']); cbar.ax.yaxis.label.set_size(10)
        for tick_label in cbar.ax.get_yticklabels(): tick_label.set_fontname(plt.rcParams['font.family']); tick_label.set_fontsize(9)
        ax.grid(False)
        
        if seq_len_for_ticks is not None and isinstance(seq_len_for_ticks, int) and seq_len_for_ticks > 0:
            num_ticks = min(10, seq_len_for_ticks)
            tick_positions = np.linspace(0, attention_scores.shape[0] -1 , num_ticks, dtype=int)
            tick_labels = np.linspace(0, seq_len_for_ticks -1, num_ticks, dtype=int)
            ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename), dpi=300, bbox_inches='tight')
        plt.close(fig); print(f"图 '{output_filename}' 已保存。")

# =================================
# 7. 主函数
# =================================

def parse_args():
    parser = argparse.ArgumentParser(description='增强版RLAD液压支架异常检测')
    parser.add_argument('--data_path', type=str, default="C:/Users/Liu HaoTian/Desktop/Python files/deeplearning/example/timeseries/examples/RLAD/clean_data.csv", help='数据文件路径')
    parser.add_argument('--feature_column_name', type=str, default=None, help='要处理的特定特征列名 (例如 "100#")。如果为None，则交互式选择或自动选择第一个带 "#" 的列。')
    parser.add_argument('--window_size', type=int, default=288, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=12, help='滑动窗口步长')
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
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device != "cuda" else "cuda")
    print(f"使用设备: {device}")

    feature_column_to_attempt = args.feature_column_name
    actual_selected_col_name = None 

    try:
        if not os.path.exists(args.data_path):
            print(f"错误: 数据文件 '{args.data_path}' 未找到。")
            if not feature_column_to_attempt:
                print("请提供有效的数据路径或通过 --feature_column_name 参数指定列名。程序即将退出。")
                return
            print(f"将尝试使用命令行指定的特征列: '{feature_column_to_attempt}' (如果数据加载时能找到)。")
        else:
            print(f"\n正在从 '{args.data_path}' 读取列名以供选择...")
            all_columns = pd.read_csv(args.data_path, nrows=0).columns.tolist()
            
            candidate_columns = [col for col in all_columns if '#' in col]
            if not candidate_columns:
                print("未在CSV列名中找到包含 '#' 的列，将尝试查找所有数值列作为候选...")
                try:
                    temp_df_for_types = pd.read_csv(args.data_path, nrows=50) 
                    candidate_columns = temp_df_for_types.select_dtypes(include=np.number).columns.tolist()
                except Exception as e_read_small:
                    print(f"读取少量数据以推断数值列类型时出错: {e_read_small}")
                    candidate_columns = []

            if not candidate_columns:
                print("错误: 在CSV文件中未能找到合适的候选列 (含'#'或数值列)。")
                if not feature_column_to_attempt:
                    print("并且未通过 --feature_column_name 参数指定列名。程序即将退出。")
                    return
                print(f"将继续尝试使用命令行指定的特征列: '{feature_column_to_attempt}'")
            else:
                print("\n检测到以下可用的支架/特征列:")
                for i, col_name in enumerate(candidate_columns):
                    print(f"  {i+1}: {col_name}")

                user_made_interactive_choice = False
                if feature_column_to_attempt and feature_column_to_attempt in candidate_columns:
                    while True:
                        choice = input(f"\n命令行指定了特征列: '{feature_column_to_attempt}'. 是否使用此列? (y/n, 直接回车默认为 y): ").strip().lower()
                        if choice == 'n':
                            feature_column_to_attempt = None 
                            break
                        elif choice == 'y' or choice == '':
                            print(f"将使用命令行指定的特征列: '{feature_column_to_attempt}'")
                            user_made_interactive_choice = True 
                            break
                        else:
                            print("无效输入，请输入 'y' 或 'n'。")
                
                if not feature_column_to_attempt and not user_made_interactive_choice : 
                    while True:
                        try:
                            choice_str = input("请输入要分析的支架列对应的数字 (例如, 1): ").strip()
                            if not choice_str:
                                print("未输入选择。如果您想退出，请关闭程序。")
                                continue
                            choice_idx = int(choice_str) - 1
                            if 0 <= choice_idx < len(candidate_columns):
                                feature_column_to_attempt = candidate_columns[choice_idx]
                                print(f"已选择特征列: '{feature_column_to_attempt}'")
                                break
                            else:
                                print(f"无效的选择。请输入1到{len(candidate_columns)}之间的数字。")
                        except ValueError:
                            print("无效输入。请输入一个数字。")
                        except Exception as e_select:
                            print(f"选择过程中发生错误: {e_select}")
                            feature_column_to_attempt = None 
                            break
    except FileNotFoundError:
        print(f"错误: 数据文件 '{args.data_path}' 未找到。")
        if not feature_column_to_attempt:
            print("请提供有效的数据路径或通过 --feature_column_name 指定列名。程序即将退出。")
            return
    except Exception as e_read_cols:
        print(f"读取或处理列名时发生意外错误: {e_read_cols}")
        if not feature_column_to_attempt:
            print("无法确定特征列。程序即将退出。")
            return
        print(f"将继续尝试使用命令行指定的特征列 '{feature_column_to_attempt}' (如果存在)。")
    
    try:
        print(f"\n准备加载数据，尝试使用的特征列: {feature_column_to_attempt if feature_column_to_attempt else '将由加载函数自动选择第一个带#的列'}")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, \
        test_window_global_start_indices, df_full_data_with_selected_col, actual_selected_col_name = \
            load_hydraulic_data_improved(
                data_path=args.data_path, 
                window_size=args.window_size, 
                stride=args.stride,
                specific_feature_column=feature_column_to_attempt
            )
        
        if not actual_selected_col_name:
            print("错误：数据加载后未能确定实际使用的特征列名。程序即将退出。")
            return
        print(f"实际使用并加载的特征列: '{actual_selected_col_name}'")

    except ValueError as ve: 
        print(f"数据加载过程中发生错误: {ve}")
        print("请检查数据文件和列名。程序即将退出。")
        return
    except Exception as e_load:
        print(f"数据加载过程中发生未知严重错误: {e_load}")
        traceback.print_exc()
        print("程序即将退出。")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_feature_name_for_output = "".join(c if c.isalnum() else "_" for c in actual_selected_col_name)
    output_dir_timestamped = os.path.join(args.output_dir, f"enhanced_rlad_results_{clean_feature_name_for_output}_{timestamp}")
    
    os.makedirs(output_dir_timestamped, exist_ok=True)
    print(f"所有输出将保存到: {output_dir_timestamped}")
    
    config_to_save = convert_to_serializable(vars(args))
    config_to_save['output_dir_actual'] = output_dir_timestamped
    config_to_save['actual_feature_column_used'] = actual_selected_col_name 
    with open(os.path.join(output_dir_timestamped, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=4)
    
    try:
        input_dim = X_train.shape[-1] 
        print(f"模型将使用的输入维度 (特征数): {input_dim}")

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
        test_metrics = enhanced_evaluate_model(agent, test_loader, device) 
        print(f"\n最终测试集评估结果 (窗口级别, 基于列 '{actual_selected_col_name}'): 精确率={test_metrics['precision']:.4f}, 召回率={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
        if len(test_metrics['f1_per_class']) >= 2: print(f"  正常类F1: {test_metrics['f1_per_class'][0]:.4f}, 异常类F1: {test_metrics['f1_per_class'][1]:.4f}")
        with open(os.path.join(output_dir_timestamped, 'test_results_window_level.json'), 'w', encoding='utf-8') as f: json.dump(convert_to_serializable(test_metrics), f, ensure_ascii=False, indent=4)

        point_level_anomalies = np.zeros(len(df_full_data_with_selected_col), dtype=int)
        predictions_for_labeled_windows = np.array(test_metrics.get('predictions', []))
        print(f"\n进行逐点异常标记 (基于列 '{actual_selected_col_name}')...")
        
        labeled_pred_idx = 0 
        for i in range(len(X_test)): 
            if y_test[i] != -1: 
                if labeled_pred_idx < len(predictions_for_labeled_windows):
                    prediction_for_this_window = predictions_for_labeled_windows[labeled_pred_idx]
                    if prediction_for_this_window == 1: 
                        start_original_data_idx = test_window_global_start_indices[i]
                        for point_offset in range(args.window_size):
                            actual_idx_in_full_data = start_original_data_idx + point_offset
                            if actual_idx_in_full_data < len(point_level_anomalies):
                                point_level_anomalies[actual_idx_in_full_data] = 1
                    labeled_pred_idx += 1
                else:
                    print(f"警告: 在处理X_test窗口索引 {i} 时，预期有预测结果，但已用尽所有预测。")
        
        df_full_data_with_selected_col['is_anomaly_predicted'] = point_level_anomalies
        extracted_abnormal_df = df_full_data_with_selected_col[df_full_data_with_selected_col['is_anomaly_predicted'] == 1]
        print(f"\n提取出的异常数据点数量 (基于列 '{actual_selected_col_name}'): {len(extracted_abnormal_df)}")
        if not extracted_abnormal_df.empty:
            print("提取出的异常数据点 (前5条):"); print(extracted_abnormal_df.head())
        extracted_abnormal_df.to_csv(os.path.join(output_dir_timestamped, 'extracted_abnormal_data_points.csv'), index=False, encoding='utf-8-sig')
        print(f"\n所有数据点及其预测的异常标签 (基于列 '{actual_selected_col_name}', 前5条):"); print(df_full_data_with_selected_col.head())
        df_full_data_with_selected_col.to_csv(os.path.join(output_dir_timestamped, 'all_data_with_point_predictions.csv'), index=False, encoding='utf-8-sig')

        if test_metrics['labels']: 
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
        if X_test is not None and y_test is not None and len(X_test) > 0:
            mask_labeled_test_windows = (y_test != -1)
            X_test_for_3d = X_test[mask_labeled_test_windows]
            y_test_labels_for_3d = y_test[mask_labeled_test_windows]
            if len(X_test_for_3d) > 0:
                visualizer.plot_3d_scatter_anomalies(X_test_for_3d, y_test_labels_for_3d)
            else:
                print("测试集中没有已标记的窗口可用于3D散点图可视化。")
        print(f"\n增强版RLAD训练和评估完成！所有结果保存在: {output_dir_timestamped}")

    except Exception as e_main_process:
        print(f"主处理流程中发生严重错误: {str(e_main_process)}")
        traceback.print_exc()
        print(f"错误发生，输出可能不完整。已保存部分结果（如果适用）于: {output_dir_timestamped if 'output_dir_timestamped' in locals() else args.output_dir}")
if __name__ == "__main__":
    set_seed(42) 
    main()                                        