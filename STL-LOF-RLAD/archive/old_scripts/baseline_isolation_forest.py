#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest异常检测对比实验
================================

基于Isolation Forest算法的时间序列异常检测方法
与RLADv3.2进行性能对比

Isolation Forest原理：
- 通过随机分割数据空间来隔离异常点
- 异常点更容易被隔离，需要更少的分割次数
- 无监督学习，不需要标注的异常样本
- 适合高维数据和大规模数据集

作者: AI Assistant
创建时间: 2025-08-26
版本: v1.0
"""

import os
import sys
import json
import random
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    average_precision_score, classification_report
)
from sklearn.model_selection import ParameterGrid

# 设置中文字体和科学图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 14
sns.set_style("whitegrid")

# 忽略警告
warnings.filterwarnings("ignore")

# =================================
# 辅助函数
# =================================

def set_seed(seed=42):
    """设置随机种子保证可重现性"""
    random.seed(seed)
    np.random.seed(seed)

def convert_to_serializable(obj):
    """将numpy等对象转换为可JSON序列化的格式"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
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
# 核心指标可视化类
# =================================

class CoreMetricsVisualizer:
    def __init__(self, output_dir="./output_visuals"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = {
            'primary': '#2E86AB', 'secondary': '#A23B72', 'tertiary': '#F18F01',
            'accent': '#C73E1D', 'neutral': '#59A5D8', 'black': '#333333'
        }

    def _set_scientific_style(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)

    def plot_final_metrics_bar(self, precision, recall, f1_score, auc_roc, save_path=None):
        metrics, values = ['AUC-ROC', 'F1-Score', 'Recall', 'Precision'], [auc_roc, f1_score, recall, precision]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(metrics, values, color=self.colors['primary'], height=0.6)
        self._set_scientific_style(ax, 'Isolation Forest Method Performance', 'Score', 'Metric')
        ax.set_xlim(0, 1.0)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                   va='center', ha='left', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save_path is None: 
            save_path = os.path.join(self.output_dir, 'isolation_forest_metrics_summary.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Isolation Forest metrics summary plot saved to: {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={"size": 14}, linecolor='white', linewidths=1)
        self._set_scientific_style(ax, 'Isolation Forest Confusion Matrix', 'Predicted Label', 'True Label')
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'], va='center', rotation=90)
        
        plt.tight_layout()
        if save_path is None: 
            save_path = os.path.join(self.output_dir, 'isolation_forest_confusion_matrix.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Isolation Forest confusion matrix plot saved to: {save_path}")

    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        if len(np.unique(y_true)) < 2: 
            print("⚠️ 无法绘制ROC曲线：只有一个类别")
            return 0.5
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=2, 
                label=f'Isolation Forest ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=self.colors['black'], lw=1.5, 
                linestyle='--', label='Random Classifier')
        
        self._set_scientific_style(ax, 'Isolation Forest ROC Curve', 
                                 'False Positive Rate', 'True Positive Rate')
        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right", frameon=False)
        
        plt.tight_layout()
        if save_path is None: 
            save_path = os.path.join(self.output_dir, 'isolation_forest_roc_curve.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Isolation Forest ROC curve plot saved to: {save_path}")
        return roc_auc

    def plot_anomaly_scores_distribution(self, scores, y_true, threshold, save_path=None):
        """绘制异常分数分布"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        
        ax.hist(normal_scores, bins=50, alpha=0.7, color=self.colors['neutral'], 
                label=f'Normal (n={len(normal_scores)})', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, color=self.colors['accent'], 
                label=f'Anomaly (n={len(anomaly_scores)})', density=True)
        
        # 添加阈值线
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Threshold = {threshold:.3f}')
        
        self._set_scientific_style(ax, 'Isolation Forest Anomaly Scores Distribution', 
                                 'Anomaly Score', 'Density')
        ax.legend()
        
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'isolation_forest_scores_distribution.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Isolation Forest scores distribution plot saved to: {save_path}")

    def plot_feature_importance(self, feature_names, importances, save_path=None):
        """绘制特征重要性（如果有多个特征）"""
        if len(feature_names) <= 1:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 排序
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        bars = ax.bar(range(len(sorted_features)), sorted_importances, 
                     color=self.colors['primary'])
        
        self._set_scientific_style(ax, 'Feature Importance in Isolation Forest', 
                                 'Features', 'Importance')
        ax.set_xticks(range(len(sorted_features)))
        ax.set_xticklabels(sorted_features, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, importance in zip(bars, sorted_importances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{importance:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'isolation_forest_feature_importance.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Isolation Forest feature importance plot saved to: {save_path}")

# =================================
# Isolation Forest异常检测器
# =================================

class IsolationForestAnomalyDetector:
    """基于Isolation Forest的异常检测器"""
    
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, 
                 max_features=1.0, bootstrap=False, random_state=42):
        """
        初始化Isolation Forest异常检测器
        
        参数:
        - n_estimators: 树的数量
        - max_samples: 每棵树使用的样本数
        - contamination: 预期的异常比例
        - max_features: 每棵树使用的特征比例
        - bootstrap: 是否使用bootstrap采样
        - random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # 创建Isolation Forest模型
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=-1  # 使用所有CPU核心
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.threshold = None
        
        print(f"🌲 Isolation Forest初始化完成:")
        print(f"   - 树的数量: {n_estimators}")
        print(f"   - 最大样本数: {max_samples}")
        print(f"   - 预期异常比例: {contamination}")
        print(f"   - 特征比例: {max_features}")
    
    def _prepare_features(self, X):
        """准备特征矩阵"""
        if X.ndim == 3:
            # 如果是3D数据 (n_samples, window_size, n_features)
            # 展平为2D (n_samples, window_size * n_features)
            n_samples, window_size, n_features = X.shape
            X_flat = X.reshape(n_samples, window_size * n_features)
        elif X.ndim == 2:
            # 如果已经是2D数据
            X_flat = X.copy()
        else:
            raise ValueError(f"不支持的数据维度: {X.ndim}")
        
        return X_flat
    
    def fit(self, X_train):
        """训练Isolation Forest模型"""
        print("🌲 开始训练Isolation Forest模型...")
        
        # 准备特征
        X_train_flat = self._prepare_features(X_train)
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        # 训练模型
        self.model.fit(X_train_scaled)
        
        # 计算训练集上的异常分数用于确定阈值
        train_scores = self.model.decision_function(X_train_scaled)
        
        # 使用分位数方法确定阈值
        self.threshold = np.percentile(train_scores, self.contamination * 100)
        
        self.is_fitted = True
        
        print(f"✅ Isolation Forest训练完成")
        print(f"   - 训练样本数: {len(X_train_scaled)}")
        print(f"   - 特征维度: {X_train_scaled.shape[1]}")
        print(f"   - 异常阈值: {self.threshold:.6f}")
        
        return self
    
    def predict_scores(self, X):
        """预测异常分数"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        
        # 准备特征
        X_flat = self._prepare_features(X)
        
        # 标准化
        X_scaled = self.scaler.transform(X_flat)
        
        # 计算异常分数
        scores = self.model.decision_function(X_scaled)
        
        # Isolation Forest的分数越小表示越异常
        # 为了与其他方法保持一致，我们取负值
        anomaly_scores = -scores
        
        return anomaly_scores
    
    def predict(self, X):
        """预测异常标签"""
        scores = self.predict_scores(X)
        
        # 使用阈值进行预测
        predictions = (scores > -self.threshold).astype(int)
        
        return predictions
    
    def fit_predict(self, X):
        """训练并预测"""
        self.fit(X)
        return self.predict(X)

# =================================
# 增强版Isolation Forest检测器
# =================================

class EnhancedIsolationForestDetector:
    """增强版Isolation Forest检测器，支持参数调优和集成"""
    
    def __init__(self, auto_tune=True, ensemble_size=5, random_state=42):
        self.auto_tune = auto_tune
        self.ensemble_size = ensemble_size
        self.random_state = random_state
        self.best_params = None
        self.models = []
        self.scaler = RobustScaler()  # 使用RobustScaler更适合异常检测
        
    def _get_param_grid(self):
        """获取参数网格"""
        return {
            'n_estimators': [50, 100, 200],
            'max_samples': ['auto', 0.5, 0.7, 1.0],
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'max_features': [0.5, 0.7, 1.0]
        }
    
    def _evaluate_params(self, X_train, X_val, y_val, params):
        """评估参数组合"""
        try:
            detector = IsolationForestAnomalyDetector(
                random_state=self.random_state,
                **params
            )
            
            detector.fit(X_train)
            scores = detector.predict_scores(X_val)
            predictions = detector.predict(X_val)
            
            # 过滤标注样本
            labeled_mask = (y_val != -1)
            if not np.any(labeled_mask):
                return 0.0
            
            y_val_labeled = y_val[labeled_mask]
            pred_labeled = predictions[labeled_mask]
            
            if len(np.unique(y_val_labeled)) < 2:
                return 0.0
            
            # 使用F1分数作为评估指标
            f1 = f1_score(y_val_labeled, pred_labeled, average='weighted', zero_division=0.0)
            
            return f1
            
        except Exception as e:
            print(f"⚠️ 参数评估失败: {e}")
            return 0.0
    
    def fit(self, X_train, X_val=None, y_val=None):
        """训练增强版模型"""
        print("🌲 开始训练增强版Isolation Forest...")
        
        # 参数调优
        if self.auto_tune and X_val is not None and y_val is not None:
            print("🔍 进行参数调优...")
            
            param_grid = self._get_param_grid()
            best_score = 0.0
            best_params = None
            
            # 网格搜索
            for params in ParameterGrid(param_grid):
                score = self._evaluate_params(X_train, X_val, y_val, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            self.best_params = best_params if best_params else {
                'n_estimators': 100,
                'max_samples': 'auto',
                'contamination': 0.1,
                'max_features': 1.0
            }
            
            print(f"✅ 最佳参数: {self.best_params}")
            print(f"✅ 最佳验证F1分数: {best_score:.4f}")
        
        else:
            # 使用默认参数
            self.best_params = {
                'n_estimators': 100,
                'max_samples': 'auto',
                'contamination': 0.1,
                'max_features': 1.0
            }
        
        # 创建集成模型
        print(f"🔄 创建 {self.ensemble_size} 个模型的集成...")
        
        for i in range(self.ensemble_size):
            model = IsolationForestAnomalyDetector(
                random_state=self.random_state + i,
                **self.best_params
            )
            model.fit(X_train)
            self.models.append(model)
        
        print(f"✅ 增强版Isolation Forest训练完成")
        
        return self
    
    def predict_scores(self, X):
        """集成预测异常分数"""
        if not self.models:
            raise ValueError("模型尚未训练")
        
        all_scores = []
        for model in self.models:
            scores = model.predict_scores(X)
            all_scores.append(scores)
        
        # 取平均分数
        ensemble_scores = np.mean(all_scores, axis=0)
        
        return ensemble_scores
    
    def predict(self, X):
        """集成预测异常标签"""
        if not self.models:
            raise ValueError("模型尚未训练")
        
        all_predictions = []
        for model in self.models:
            predictions = model.predict(X)
            all_predictions.append(predictions)
        
        # 多数投票
        all_predictions = np.array(all_predictions)
        ensemble_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
        
        return ensemble_predictions

# =================================
# 数据加载和预处理
# =================================

def load_and_preprocess_data(data_path, feature_column, window_size=288, stride=20):
    """加载和预处理数据"""
    print(f"📊 加载数据: {data_path}")
    data = pd.read_csv(data_path)
    
    if feature_column not in data.columns:
        raise ValueError(f"特征列 '{feature_column}' 不存在于数据中")
    
    feature_data = data[feature_column].values
    
    # 创建滑动窗口
    print(f"🔄 创建滑动窗口 (窗口大小: {window_size}, 步长: {stride})")
    windows = []
    
    for i in range(0, len(feature_data) - window_size + 1, stride):
        window = feature_data[i:i + window_size]
        windows.append(window.reshape(-1, 1))
    
    X = np.array(windows)
    
    print(f"✅ 数据加载完成: {len(X)} 个窗口, 窗口形状: {X[0].shape}")
    return X, feature_data

def create_pseudo_labels(X, contamination_rate=0.08):
    """创建伪标签用于评估"""
    print("🔄 创建伪标签用于评估...")
    
    # 使用简单的Isolation Forest创建伪标签
    temp_detector = IsolationForestAnomalyDetector(
        contamination=contamination_rate,
        random_state=42
    )
    
    temp_detector.fit(X)
    scores = temp_detector.predict_scores(X)
    
    # 使用分位数方法创建标签
    threshold = np.percentile(scores, (1 - contamination_rate) * 100)
    pseudo_labels = (scores > threshold).astype(int)
    
    print(f"📊 伪标签分布: 正常={np.sum(pseudo_labels == 0)}, 异常={np.sum(pseudo_labels == 1)}")
    print(f"📈 异常比例: {np.mean(pseudo_labels):.3f}")
    
    return pseudo_labels

def split_data(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """时间序列数据分割"""
    n_samples = len(X)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    print(f"📊 数据分割完成:")
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   验证集: {len(X_val)} 样本")
    print(f"   测试集: {len(X_test)} 样本")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# =================================
# 评估函数
# =================================

def evaluate_model(detector, X_test, y_test):
    """评估模型性能"""
    print("🔍 评估Isolation Forest模型性能...")
    
    # 预测
    y_pred = detector.predict(X_test)
    y_scores = detector.predict_scores(X_test)
    
    # 计算指标
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0.0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0.0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0.0)
    
    try:
        auc_roc = roc_auc_score(y_test, y_scores)
    except ValueError:
        auc_roc = 0.5
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }
    
    print(f"📈 性能指标:")
    print(f"   精确率: {precision:.4f}")
    print(f"   召回率: {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    
    # 详细分类报告
    print("\n📋 详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    return metrics, y_pred, y_scores

# =================================
# 主函数
# =================================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Isolation Forest异常检测对比实验")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='clean_data.csv',
                       help='数据文件路径')
    parser.add_argument('--feature_column', type=str, default='103#',
                       help='特征列名')
    parser.add_argument('--window_size', type=int, default=288,
                       help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=20,
                       help='滑动窗口步长')
    
    # Isolation Forest参数
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='树的数量')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='预期异常比例')
    parser.add_argument('--max_samples', type=str, default='auto',
                       help='每棵树的最大样本数')
    parser.add_argument('--max_features', type=float, default=1.0,
                       help='每棵树的特征比例')
    
    # 实验参数
    parser.add_argument('--enhanced', action='store_true', default=True,
                       help='是否使用增强版模型（参数调优+集成）')
    parser.add_argument('--ensemble_size', type=int, default=5,
                       help='集成模型数量')
    parser.add_argument('--auto_tune', action='store_true', default=True,
                       help='是否自动调优参数')
    
    # 其他参数
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
        args.output_dir = f"output_isolation_forest_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {args.output_dir}")
    
    try:
        # 1. 加载数据
        X, original_data = load_and_preprocess_data(
            args.data_path, 
            args.feature_column, 
            args.window_size, 
            args.stride
        )
        
        # 2. 创建伪标签
        y = create_pseudo_labels(X, contamination_rate=args.contamination)
        
        # 3. 分割数据
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
        
        # 4. 创建检测器
        if args.enhanced:
            print("🚀 创建增强版Isolation Forest检测器...")
            detector = EnhancedIsolationForestDetector(
                auto_tune=args.auto_tune,
                ensemble_size=args.ensemble_size,
                random_state=args.seed
            )
            
            # 训练模型
            detector.fit(X_train, X_val, y_val)
            
        else:
            print("🚀 创建标准Isolation Forest检测器...")
            
            # 转换max_samples参数
            max_samples = args.max_samples
            if max_samples != 'auto':
                try:
                    max_samples = float(max_samples)
                except ValueError:
                    max_samples = 'auto'
            
            detector = IsolationForestAnomalyDetector(
                n_estimators=args.n_estimators,
                max_samples=max_samples,
                contamination=args.contamination,
                max_features=args.max_features,
                random_state=args.seed
            )
            
            # 训练模型（只使用正常样本）
            X_train_normal = X_train[y_train == 0] if len(X_train[y_train == 0]) > 0 else X_train
            detector.fit(X_train_normal)
        
        # 5. 评估模型
        metrics, y_pred, y_scores = evaluate_model(detector, X_test, y_test)
        
        # 6. 可视化结果
        print("\n📊 生成可视化结果...")
        visualizer = CoreMetricsVisualizer(args.output_dir)
        
        # 绘制性能指标
        visualizer.plot_final_metrics_bar(
            metrics['precision'], metrics['recall'], 
            metrics['f1'], metrics['auc_roc']
        )
        
        # 绘制混淆矩阵
        visualizer.plot_confusion_matrix(y_test, y_pred)
        
        # 绘制ROC曲线
        visualizer.plot_roc_curve(y_test, y_scores)
        
        # 绘制分数分布
        if hasattr(detector, 'threshold'):
            threshold = -detector.threshold if hasattr(detector, 'threshold') else np.median(y_scores)
        else:
            threshold = np.median(y_scores)
        
        visualizer.plot_anomaly_scores_distribution(y_scores, y_test, threshold)
        
        # 7. 保存结果
        print("\n💾 保存实验结果...")
        
        results_summary = {
            'experiment_info': {
                'method': 'Isolation Forest',
                'enhanced': args.enhanced,
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'n_estimators': args.n_estimators,
                    'contamination': args.contamination,
                    'max_samples': args.max_samples,
                    'max_features': args.max_features,
                    'ensemble_size': args.ensemble_size if args.enhanced else 1,
                    'auto_tune': args.auto_tune if args.enhanced else False
                }
            },
            'data_info': {
                'data_path': args.data_path,
                'feature_column': args.feature_column,
                'window_size': args.window_size,
                'stride': args.stride,
                'total_samples': len(X),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'test_normal_samples': int(np.sum(y_test == 0)),
                'test_anomaly_samples': int(np.sum(y_test == 1))
            },
            'performance_metrics': {
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1']),
                'auc_roc': float(metrics['auc_roc'])
            },
            'best_parameters': detector.best_params if args.enhanced and hasattr(detector, 'best_params') else None,
            'comparison_baseline': {
                'method_description': 'Isolation Forest for Time Series Anomaly Detection',
                'approach': 'Tree-based isolation of anomalous patterns',
                'advantages': [
                    'No need for labeled anomaly data',
                    'Efficient for large datasets',
                    'Good performance on high-dimensional data',
                    'Robust to noise and outliers'
                ],
                'limitations': [
                    'Assumes anomalies are rare',
                    'Performance depends on contamination parameter',
                    'May not capture complex temporal patterns',
                    'Less interpretable than statistical methods'
                ]
            }
        }
        
        # 转换为可序列化格式
        results_summary = convert_to_serializable(results_summary)
        
        # 保存结果
        results_file = os.path.join(args.output_dir, 'isolation_forest_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Isolation Forest实验结果已保存到: {results_file}")
        
        # 8. 生成Markdown报告
        print("📝 生成实验报告...")
        
        report_content = f"""
# Isolation Forest异常检测实验报告

## 实验概述
- **方法**: Isolation Forest
- **增强版**: {'是' if args.enhanced else '否'}
- **实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据来源**: {args.data_path}
- **特征列**: {args.feature_column}

## 实验参数
- **窗口大小**: {args.window_size}
- **滑动步长**: {args.stride}
- **树的数量**: {args.n_estimators}
- **预期异常比例**: {args.contamination}
- **最大样本数**: {args.max_samples}
- **特征比例**: {args.max_features}
{'- **集成大小**: ' + str(args.ensemble_size) if args.enhanced else ''}
{'- **自动调优**: ' + ('是' if args.auto_tune else '否') if args.enhanced else ''}

## 数据统计
- **总样本数**: {len(X)}
- **训练集**: {len(X_train)}
- **验证集**: {len(X_val)}
- **测试集**: {len(X_test)}
- **测试集正常样本**: {np.sum(y_test == 0)}
- **测试集异常样本**: {np.sum(y_test == 1)}

## 性能指标
- **精确率**: {metrics['precision']:.4f}
- **召回率**: {metrics['recall']:.4f}
- **F1分数**: {metrics['f1']:.4f}
- **AUC-ROC**: {metrics['auc_roc']:.4f}

## 方法优势
- ✅ 无需标注的异常样本
- ✅ 对高维数据表现良好
- ✅ 计算效率高，适合大数据集
- ✅ 对噪声和离群值鲁棒

## 方法局限
- ⚠️ 假设异常样本稀少
- ⚠️ 性能依赖contamination参数设置
- ⚠️ 可能无法捕获复杂的时序模式
- ⚠️ 相比统计方法可解释性较差

## 与RLADv3.2对比
Isolation Forest作为经典的无监督异常检测方法，为RLADv3.2提供了重要的性能基准。
通过对比可以评估强化学习方法相对于传统机器学习方法的优势。

## 结论
Isolation Forest在时间序列异常检测任务中表现{'良好' if metrics['f1'] > 0.7 else '一般'}，
F1分数达到{metrics['f1']:.4f}，为后续方法对比提供了可靠的基准。
"""
        
        report_file = os.path.join(args.output_dir, 'isolation_forest_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 实验报告已保存到: {report_file}")
        
        print(f"\n✅ Isolation Forest对比实验完成!")
        print(f"📁 所有结果已保存到: {args.output_dir}")
        print(f"\n🎯 最终性能指标:")
        print(f"   精确率: {metrics['precision']:.4f}")
        print(f"   召回率: {metrics['recall']:.4f}")
        print(f"   F1分数: {metrics['f1']:.4f}")
        print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
