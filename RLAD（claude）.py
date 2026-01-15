"""
改进版RLAD: 基于强化学习与主动学习的时间序列异常检测
专用于液压支架工作阻力异常检测 - 修复版本
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
        support_columns = df.select_dtypes(include=[np.number]).columns[:3].tolist()
    
    print(f"发现支架列: {support_columns[:10]}...")
    
    # 选择前3个支架作为多通道输入
    if len(support_columns) >= 3:
        selected_cols = support_columns[:3]
    else:
        selected_cols = support_columns + [support_columns[0]] * (3 - len(support_columns))
    
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
    
    # 4. 基于梯度变化的异常检测
    gradients = np.abs(np.diff(X_flat, axis=1))
    gradient_means = np.mean(gradients, axis=1)
    grad_threshold = np.percentile(gradient_means, 85)
    grad_anomalies = gradient_means > grad_threshold
    
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
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # 自注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 残差连接
        lstm_out = self.ln_attention(lstm_out + attn_out)
        
        # 全局平均池化和最大池化结合
        avg_pool = torch.mean(lstm_out, dim=1)  # (batch, hidden_size*2)
        max_pool, _ = torch.max(lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # 融合不同池化结果
        combined = avg_pool + max_pool  # 残差连接风格的融合
        
        # 全连接层
        x = self.fc1(combined)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        features = x  # 保存特征用于可视化
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        q_values = self.fc3(x)
        
        if return_features:
            return q_values, features
        else:
            return q_values
    
    def get_action(self, state, epsilon=0.0):
        """获取动作，支持单样本推理"""
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            # 确保在推理模式下
            was_training = self.training
            self.eval()
            
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
            
            # 恢复原来的模式
            if was_training:
                self.train()
            
            return action

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
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = Experience(state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
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
        return 0.0
    
    base_reward = confidence * 1.0
    
    if action == true_label:
        if true_label == 1:  # 正确识别异常
            reward = base_reward * 5.0  # 进一步提高异常检测奖励
        else:  # 正确识别正常
            reward = base_reward * 1.0
    else:
        if true_label == 1 and action == 0:  # 漏报
            reward = -base_reward * 3.0  # 重点惩罚漏报
        else:  # 误报
            reward = -base_reward * 0.3  # 轻微惩罚误报
    
    # 添加多样性奖励
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
    anomaly_scores = iso_forest.decision_function(X_flat)
    
    # 2. 基于聚类的异常检测
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
    
    n_clusters = min(10, len(X_labeled) // 5)
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_flat)
        
        # 计算到聚类中心的距离
        distances = []
        for i, label in enumerate(cluster_labels):
            dist = np.linalg.norm(X_flat[i] - kmeans.cluster_centers_[label])
            distances.append(dist)
        distances = np.array(distances)
        
        # 结合IsolationForest和聚类距离
        combined_scores = 0.7 * anomaly_scores + 0.3 * (distances / np.max(distances))
    else:
        combined_scores = anomaly_scores
    
    # 分层抽样
    n_samples = len(X_labeled)
    n_abnormal = max(1, int(0.2 * n_samples))
    n_normal = max(1, int(0.5 * n_samples))
    n_middle = max(0, n_samples - n_abnormal - n_normal)
    
    sorted_indices = np.argsort(combined_scores)
    abnormal_indices = sorted_indices[:n_abnormal]
    normal_indices = sorted_indices[-n_normal:]
    if n_middle > 0:
        middle_indices = sorted_indices[n_abnormal:n_abnormal + n_middle]
    else:
        middle_indices = []
    
    # 添加到回放池
    for indices, confidence in [(abnormal_indices, 0.95), (normal_indices, 0.85), (middle_indices, 0.6)]:
        for i in indices:
            state = torch.FloatTensor(X_labeled[i]).to(device)
            true_label = y_labeled[i]
            
            # 智能动作选择
            if confidence > 0.8:
                action = true_label
            elif confidence > 0.6:
                action = true_label if random.random() < 0.8 else 1 - true_label
            else:
                action = random.randint(0, 1)
            
            # 多样性奖励
            diversity_bonus = 0.1 if true_label == 1 else 0.0
            reward = enhanced_compute_reward(action, true_label, confidence, diversity_bonus)
            
            next_state = state
            done = False
            
            replay_buffer.push(
                state.cpu().squeeze(0),
                action,
                reward,
                next_state.cpu().squeeze(0),
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
        return 0.0
    
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)
    
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    weights = torch.FloatTensor(weights).to(device)
    
    # 当前Q值
    current_q_values = agent(states).gather(1, actions.unsqueeze(1))
    
    # 目标Q值 - 使用Double DQN
    with torch.no_grad():
        # 使用在线网络选择动作
        next_actions = agent(next_states).argmax(1)
        # 使用目标网络计算Q值
        next_q_values = target_agent(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (gamma * next_q_values * ~dones)
    
    # TD误差
    td_errors = current_q_values.squeeze() - target_q_values
    
    # 加权Huber Loss
    loss = (weights * F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
    
    # 优化
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 更新优先级
    priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
    replay_buffer.update_priorities(indices, priorities)
    
    return loss.item()

def enhanced_evaluate_model(agent, data_loader, device):
    """增强的模型评估，包含更多指标"""
    agent.eval()
    
    all_predictions = []
    all_labels = []
    all_q_values = []
    all_features = []
    
    with torch.no_grad():
        for states, labels in data_loader:
            states = states.to(device)
            
            labeled_mask = (labels != -1)
            if labeled_mask.sum() == 0:
                continue
                
            states_labeled = states[labeled_mask]
            labels_labeled = labels[labeled_mask]
            
            q_values, features = agent(states_labeled, return_features=True)
            predictions = q_values.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_labeled.cpu().numpy())
            all_q_values.extend(q_values.cpu().numpy())
            all_features.extend(features.cpu().numpy())
    
    if len(all_predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 计算详细指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 基础指标
    precision = precision_score(all_labels, all_predictions, zero_division=0, average='weighted')
    recall = recall_score(all_labels, all_predictions, zero_division=0, average='weighted')
    f1 = f1_score(all_labels, all_predictions, zero_division=0, average='weighted')
    
    # 分类别指标
    precision_per_class = precision_score(all_labels, all_predictions, zero_division=0, average=None)
    recall_per_class = recall_score(all_labels, all_predictions, zero_division=0, average=None)
    f1_per_class = f1_score(all_labels, all_predictions, zero_division=0, average=None)
    
    agent.train()
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_per_class": [float(x) for x in precision_per_class],
        "recall_per_class": [float(x) for x in recall_per_class],
        "f1_per_class": [float(x) for x in f1_per_class],
        "predictions": all_predictions.tolist(),
        "labels": all_labels.tolist(),
        "q_values": np.array(all_q_values).tolist(),
        "features": np.array(all_features).tolist()
    }

# =================================
# 5. 增强的主训练函数
# =================================

def enhanced_train_rlad(agent, target_agent, optimizer, scheduler, replay_buffer, 
                       X_train, y_train, X_val, y_val, device, 
                       num_episodes=150, target_update_freq=15,
                       epsilon_start=0.95, epsilon_end=0.02, epsilon_decay=1200,
                       output_dir="./output"):
    """增强的RLAD主训练函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    training_history = {
        'episodes': [],
        'train_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'epsilon': [],
        'replay_buffer_size': [],
        'learning_rate': [],
        'anomaly_f1': [],  # 专门追踪异常类的F1
        'normal_f1': []    # 专门追踪正常类的F1
    }
    
    best_f1 = 0.0
    best_anomaly_f1 = 0.0
    patience = 40
    patience_counter = 0
    
    print("开始增强的RLAD训练...")
    
    # 过滤掉未标注样本用于训练
    labeled_mask = (y_train != -1)
    X_train_labeled = X_train[labeled_mask]
    y_train_labeled = y_train[labeled_mask]
    
    print(f"训练样本数量: {len(X_train_labeled)} (过滤后)")
    print(f"异常样本数量: {np.sum(y_train_labeled == 1)}")
    print(f"正常样本数量: {np.sum(y_train_labeled == 0)}")
    
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        episode_losses = []
        
        # 动态epsilon衰减
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)
        
        # 动态beta（优先级经验回放）
        beta = min(1.0, 0.4 + 0.6 * episode / num_episodes)
        
        agent.train()
        
        # 改进的训练循环
        n_train = len(X_train_labeled)
        batch_size = min(64, max(16, n_train // 8))
        
        # 类别平衡抽样
        normal_indices = np.where(y_train_labeled == 0)[0]
        anomaly_indices = np.where(y_train_labeled == 1)[0]
        
        # 平衡采样
        n_normal_samples = len(normal_indices)
        n_anomaly_samples = len(anomaly_indices)
        
        if n_anomaly_samples > 0:
            # 过采样异常样本
            anomaly_samples = np.random.choice(anomaly_indices, 
                                             size=min(n_normal_samples, n_anomaly_samples * 3), 
                                             replace=True)
            normal_samples = np.random.choice(normal_indices, 
                                            size=len(anomaly_samples), 
                                            replace=False)
            balanced_indices = np.concatenate([normal_samples, anomaly_samples])
        else:
            balanced_indices = normal_indices
        
        np.random.shuffle(balanced_indices)
        
        for i in range(0, len(balanced_indices), batch_size):
            batch_indices = balanced_indices[i:min(i + batch_size, len(balanced_indices))]
            
            for idx in batch_indices:
                state = torch.FloatTensor(X_train_labeled[idx]).to(device)
                action = agent.get_action(state.unsqueeze(0), epsilon)
                
                # 动态奖励调整
                if y_train_labeled[idx] == 1:  # 异常样本
                    diversity_bonus = 0.2
                else:
                    diversity_bonus = 0.0
                
                reward = enhanced_compute_reward(action, y_train_labeled[idx], 
                                               confidence=1.0, diversity_bonus=diversity_bonus)
                
                # 构造下一状态
                next_idx = (idx + 1) % len(balanced_indices)
                next_state = torch.FloatTensor(X_train_labeled[balanced_indices[next_idx]]).to(device)
                done = False
                
                replay_buffer.push(
                    state.cpu(),
                    action,
                    reward,
                    next_state.cpu(),
                    done
                )
                
                # 训练步骤
                if len(replay_buffer) >= 128:
                    loss = enhanced_train_dqn_step(
                        agent, target_agent, replay_buffer, 
                        optimizer, device, batch_size=batch_size, beta=beta
                    )
                    episode_losses.append(loss)
        
        # 更新目标网络
        if episode % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # 学习率调度
        scheduler.step()
        
        # 验证
        val_loader = DataLoader(
            HydraulicDataset(X_val, y_val), 
            batch_size=32, shuffle=False
        )
        val_metrics = enhanced_evaluate_model(agent, val_loader, device)
        
        # 记录历史
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        training_history['episodes'].append(episode)
        training_history['train_loss'].append(avg_loss)
        training_history['val_f1'].append(val_metrics['f1'])
        training_history['val_precision'].append(val_metrics['precision'])
        training_history['val_recall'].append(val_metrics['recall'])
        training_history['epsilon'].append(epsilon)
        training_history['replay_buffer_size'].append(len(replay_buffer))
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 记录分类别F1
        if len(val_metrics['f1_per_class']) >= 2:
            training_history['normal_f1'].append(val_metrics['f1_per_class'][0])
            training_history['anomaly_f1'].append(val_metrics['f1_per_class'][1])
        else:
            training_history['normal_f1'].append(0.0)
            training_history['anomaly_f1'].append(0.0)
        
        # 保存最佳模型 - 优先考虑异常检测性能
        current_anomaly_f1 = training_history['anomaly_f1'][-1]
        if val_metrics['f1'] > best_f1 or current_anomaly_f1 > best_anomaly_f1:
            best_f1 = max(best_f1, val_metrics['f1'])
            best_anomaly_f1 = max(best_anomaly_f1, current_anomaly_f1)
            patience_counter = 0
            
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'target_agent_state_dict': target_agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'best_f1': best_f1,
                'best_anomaly_f1': best_anomaly_f1,
                'training_history': training_history
            }, os.path.join(output_dir, 'best_enhanced_rlad_model.pth'))
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"早停在episode {episode}，最佳F1: {best_f1:.4f}, 最佳异常F1: {best_anomaly_f1:.4f}")
            break
        
        # 打印进度
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"Loss: {avg_loss:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print(f"异常F1: {current_anomaly_f1:.4f}, 正常F1: {training_history['normal_f1'][-1]:.4f}")
            print(f"Epsilon: {epsilon:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Buffer Size: {len(replay_buffer)}")
    
    print(f"\n训练完成！最佳F1分数: {best_f1:.4f}, 最佳异常F1: {best_anomaly_f1:.4f}")
    return training_history

# =================================
# 6. 增强的可视化类
# =================================

class EnhancedRLADVisualizer:
    """增强的RLAD结果可视化类"""
    
    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_training_history(self, training_history):
        """绘制增强的训练历史"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        episodes = training_history['episodes']
        
        # 第一行
        # 训练损失
        axes[0, 0].plot(episodes, training_history['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('训练损失', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 整体F1分数
        axes[0, 1].plot(episodes, training_history['val_f1'], 'g-', label='整体F1', linewidth=2)
        axes[0, 1].set_title('验证集F1分数', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 分类别F1分数
        if 'anomaly_f1' in training_history and 'normal_f1' in training_history:
            axes[0, 2].plot(episodes, training_history['anomaly_f1'], 'r-', label='异常F1', linewidth=2)
            axes[0, 2].plot(episodes, training_history['normal_f1'], 'b-', label='正常F1', linewidth=2)
            axes[0, 2].set_title('分类别F1分数', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('F1 Score')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
        
        # 第二行
        # 精确率和召回率
        axes[1, 0].plot(episodes, training_history['val_precision'], 'b-', label='Precision', linewidth=2)
        axes[1, 0].plot(episodes, training_history['val_recall'], 'r-', label='Recall', linewidth=2)
        axes[1, 0].set_title('精确率和召回率', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Epsilon衰减
        axes[1, 1].plot(episodes, training_history['epsilon'], 'purple', linewidth=2)
        axes[1, 1].set_title('探索率(Epsilon)衰减', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 回放池大小
        axes[1, 2].plot(episodes, training_history['replay_buffer_size'], 'orange', linewidth=2)
        axes[1, 2].set_title('经验回放池大小', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Buffer Size')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 第三行
        # 学习率
        axes[2, 0].plot(episodes, training_history['learning_rate'], 'brown', linewidth=2)
        axes[2, 0].set_title('学习率变化', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Learning Rate')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 性能综合对比
        axes[2, 1].plot(episodes, training_history['val_f1'], 'g-', label='F1', linewidth=2)
        axes[2, 1].plot(episodes, training_history['val_precision'], 'b-', label='Precision', linewidth=2)
        axes[2, 1].plot(episodes, training_history['val_recall'], 'r-', label='Recall', linewidth=2)
        axes[2, 1].set_title('综合性能指标', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Score')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
        
        # 训练稳定性（损失的滑动平均）
        if len(training_history['train_loss']) > 10:
            window_size = min(10, len(training_history['train_loss']) // 5)
            smoothed_loss = np.convolve(training_history['train_loss'], 
                                      np.ones(window_size)/window_size, mode='valid')
            smoothed_episodes = episodes[window_size-1:]
            axes[2, 2].plot(smoothed_episodes, smoothed_loss, 'navy', linewidth=2)
            axes[2, 2].set_title(f'平滑损失(窗口={window_size})', fontsize=14, fontweight='bold')
            axes[2, 2].set_xlabel('Episode')
            axes[2, 2].set_ylabel('Smoothed Loss')
            axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'enhanced_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

# =================================
# 7. 主函数
# =================================

def parse_args():
    parser = argparse.ArgumentParser(description='增强版RLAD液压支架异常检测')
    
    parser.add_argument('--data_path', type=str, 
                       default="C://Users//Liu HaoTian//Desktop//rnn+tcn+transformer+kan//time_series//timeseries//examples//RLAD//clean_data.csv",
                       help='数据文件路径')
    parser.add_argument('--window_size', type=int, default=288, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=12, help='滑动窗口步长')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=3, help='LSTM层数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--num_episodes', type=int, default=150, help='训练轮数')
    parser.add_argument('--output_dir', type=str, default="./", help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default="auto", help='设备选择')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"enhanced_rlad_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    
    # 保存配置
    config_to_save = convert_to_serializable(vars(args))
    config_to_save['output_dir'] = output_dir
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)
    
    try:
        # 1. 加载数据
        print("=" * 50)
        print("1. 加载数据")
        print("=" * 50)
        
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_hydraulic_data_improved(
            data_path=args.data_path,
            window_size=args.window_size,
            stride=args.stride
        )
        
        # 2. 创建增强模型
        print("\n" + "=" * 50)
        print("2. 创建增强的RLAD模型")
        print("=" * 50)
        
        agent = EnhancedRLADAgent(
            input_dim=3,
            seq_len=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        ).to(device)
        
        target_agent = EnhancedRLADAgent(
            input_dim=3,
            seq_len=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        ).to(device)
        
        target_agent.load_state_dict(agent.state_dict())
        
        # 使用AdamW优化器和更复杂的学习率调度
        optimizer = optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
        
        replay_buffer = PrioritizedReplayBuffer(capacity=30000)  # 使用优先级经验回放
        
        print(f"模型参数数量: {sum(p.numel() for p in agent.parameters()):,}")
        
        # 3. 增强预热
        print("\n" + "=" * 50)
        print("3. 增强的预热")
        print("=" * 50)
        
        enhanced_warmup_with_multiple_methods(X_train, y_train, replay_buffer, agent, device)
        
        # 4. 增强训练
        print("\n" + "=" * 50)
        print("4. 开始增强的RLAD训练")
        print("=" * 50)
        
        training_history = enhanced_train_rlad(
            agent=agent,
            target_agent=target_agent,
            optimizer=optimizer,
            scheduler=scheduler,
            replay_buffer=replay_buffer,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            num_episodes=args.num_episodes,
            output_dir=output_dir
        )
        
        # 5. 可视化训练历史
        print("\n" + "=" * 50)
        print("5. 可视化训练历史")
        print("=" * 50)
        
        visualizer = EnhancedRLADVisualizer(output_dir)
        visualizer.plot_training_history(training_history)
        
        # 6. 测试最佳模型
        print("\n" + "=" * 50)
        print("6. 测试最佳模型")
        print("=" * 50)
        
        best_model_path = os.path.join(output_dir, 'best_enhanced_rlad_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            agent.load_state_dict(checkpoint['agent_state_dict'])
            print(f"加载最佳模型，F1分数: {checkpoint['best_f1']:.4f}")
            print(f"最佳异常F1分数: {checkpoint['best_anomaly_f1']:.4f}")
        
        # 测试集评估
        test_loader = DataLoader(HydraulicDataset(X_test, y_test), batch_size=32, shuffle=False)
        test_metrics = enhanced_evaluate_model(agent, test_loader, device)
        
        print(f"\n最终测试结果:")
        print(f"整体精确率: {test_metrics['precision']:.4f}")
        print(f"整体召回率: {test_metrics['recall']:.4f}")
        print(f"整体F1分数: {test_metrics['f1']:.4f}")
        
        if len(test_metrics['f1_per_class']) >= 2:
            print(f"正常类F1: {test_metrics['f1_per_class'][0]:.4f}")
            print(f"异常类F1: {test_metrics['f1_per_class'][1]:.4f}")
        
        # 保存训练历史 - 使用安全的序列化
        training_history_safe = convert_to_serializable(training_history)
        with open(os.path.join(output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
            json.dump(training_history_safe, f, ensure_ascii=False, indent=2)
        
        # 保存测试结果 - 使用安全的序列化
        test_results_safe = convert_to_serializable(test_metrics)
        with open(os.path.join(output_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(test_results_safe, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 50)
        print("增强版训练完成！")
        print("=" * 50)
        print(f"所有结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()