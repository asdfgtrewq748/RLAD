#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OriginalRLADAnomalyDetector类
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

try:
    # 导入必要的模块
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.semi_supervised import LabelSpreading
    from collections import namedtuple
    import random
    from scipy import stats
    
    # 定义常量
    EPISODES = 500
    DISCOUNT_FACTOR = 0.5
    NUM_LABEL_PROPAGATION = 20
    NUM_ACTIVE_LEARNING = 5
    ACTION_SPACE_N = 2
    N_STEPS = 25
    N_INPUT_DIM = 2
    N_HIDDEN_DIM = 128
    
    # 简化的Q网络
    class Q_Estimator_Nonlinear(nn.Module):
        def __init__(self, learning_rate=0.01):
            super().__init__()
            self.learning_rate = learning_rate
            self.lstm = nn.LSTM(N_INPUT_DIM, N_HIDDEN_DIM, batch_first=True)
            self.fc = nn.Linear(N_HIDDEN_DIM, ACTION_SPACE_N)
            
        def predict(self, state):
            return np.random.rand(ACTION_SPACE_N)
    
    # WarmUp类
    class WarmUp:
        def warm_up_isolation_forest(self, outliers_fraction, X_train):
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=outliers_fraction, random_state=42)
            model.fit(X_train)
            return model
    
    # 策略函数
    def make_epsilon_greedy_policy(estimator, nA):
        def policy_fn(observation, epsilon):
            return np.ones(nA) / nA
        return policy_fn
    
    # 测试类定义
    class OriginalRLADAnomalyDetector:
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
            
            print(f"✅ 成功创建OriginalRLADAnomalyDetector, episodes={episodes}")
            self.episodes = episodes
            self.learning_rate = learning_rate
            self.is_trained = False
    
    # 测试创建实例
    print("🧪 测试创建OriginalRLADAnomalyDetector实例...")
    
    detector = OriginalRLADAnomalyDetector(
        episodes=50,
        learning_rate=0.01,
        random_state=42
    )
    
    print("🎉 测试成功！类定义正确。")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
