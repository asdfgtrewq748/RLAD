# PART 1/2

"""
RLAD结果可视化分析工具
用于分析和可视化RLAD模型的训练效果、异常检测结果和性能对比
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve, 
                           average_precision_score)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

warnings.filterwarnings("ignore")

class RLADVisualizationAnalyzer:
    """RLAD结果可视化分析器"""
    
    def __init__(self, result_dir: str):
        """
        初始化分析器
        
        Args:
            result_dir: RLAD训练结果目录路径
        """
        self.result_dir = Path(result_dir)
        self.output_dir = self.result_dir / "visualization_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载结果数据
        self.config = self.load_config()
        self.training_history = self.load_training_history()
        self.test_results = self.load_test_results()
        self.manual_annotations = self.load_manual_annotations()
        
        # 初始化数据
        self.raw_data = None
        self.model = None
        self.predictions = None
        
        print(f"分析器初始化完成，输出目录: {self.output_dir}")
        
    def load_config(self) -> Dict:
        """加载配置文件"""
        config_path = self.result_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("警告: 未找到config.json文件")
            return {}
    
    def load_training_history(self) -> Dict:
        """加载训练历史"""
        history_path = self.result_dir / "training_history_final.json"
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("警告: 未找到training_history_final.json文件")
            return {}
    
    def load_test_results(self) -> Dict:
        """加载测试结果"""
        test_path = self.result_dir / "test_results_final.json"
        if test_path.exists():
            with open(test_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("警告: 未找到test_results_final.json文件")
            return {}
    
    def load_manual_annotations(self) -> List:
        """加载人工标注记录"""
        annotations_path = self.result_dir / "manual_annotations.json"
        if annotations_path.exists():
            with open(annotations_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("警告: 未找到manual_annotations.json文件")
            return []
    
    def load_raw_data(self) -> pd.DataFrame:
        """加载原始数据"""
        if 'data_path' in self.config:
            data_path = self.config['data_path']
            if os.path.exists(data_path):
                self.raw_data = pd.read_csv(data_path)
                print(f"原始数据加载成功，形状: {self.raw_data.shape}")
                return self.raw_data
            else:
                print(f"警告: 原始数据文件不存在: {data_path}")
        return None
    
    def create_comprehensive_analysis(self):
        """创建综合分析报告"""
        print("开始创建综合分析报告...")
        
        # 1. 训练过程分析
        self.plot_training_analysis()
        
        # 2. 模型性能分析
        self.plot_performance_analysis()
        
        # 3. 人工标注分析
        self.plot_annotation_analysis()
        
        # 4. 异常检测结果分析
        self.plot_anomaly_detection_analysis()
        
        # 5. 对比分析
        self.plot_comparison_analysis()
        
        # 6. 生成分析报告
        self.generate_analysis_report()
        
        print(f"综合分析完成，结果保存在: {self.output_dir}")
    
    def plot_training_analysis(self):
        """绘制训练过程分析图"""
        if not self.training_history:
            print("跳过训练分析（无训练历史数据）")
            return
            
        print("创建训练过程分析图...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RLAD训练过程分析', fontsize=16, fontweight='bold')
        
        # 提取训练数据
        episodes = self.training_history.get('episodes', [])
        train_loss = self.training_history.get('train_loss', [])
        val_f1 = self.training_history.get('val_f1', [])
        val_precision = self.training_history.get('val_precision', [])
        val_recall = self.training_history.get('val_recall', [])
        epsilon = self.training_history.get('epsilon', [])
        human_annotations = self.training_history.get('human_annotations_count', [])
        human_accuracy = self.training_history.get('human_labeled_accuracy', [])
        anomaly_f1 = self.training_history.get('anomaly_f1', [])
        normal_f1 = self.training_history.get('normal_f1', [])
        
        # 1. 训练损失
        if train_loss:
            axes[0, 0].plot(episodes, train_loss, 'b-', linewidth=2, label='训练损失')
            axes[0, 0].set_title('训练损失变化', fontweight='bold')
            axes[0, 0].set_xlabel('训练轮数')
            axes[0, 0].set_ylabel('损失值')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # 2. 验证集性能指标
        if val_f1:
            axes[0, 1].plot(episodes, val_f1, 'g-', linewidth=2, label='F1分数')
            if val_precision:
                axes[0, 1].plot(episodes, val_precision, 'r--', linewidth=2, label='精确率')
            if val_recall:
                axes[0, 1].plot(episodes, val_recall, 'b--', linewidth=2, label='召回率')
            axes[0, 1].set_title('验证集性能指标', fontweight='bold')
            axes[0, 1].set_xlabel('训练轮数')
            axes[0, 1].set_ylabel('指标值')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            axes[0, 1].set_ylim(0, 1)
        
        # 3. Epsilon衰减
        if epsilon:
            axes[0, 2].plot(episodes, epsilon, 'purple', linewidth=2, label='Epsilon值')
            axes[0, 2].set_title('探索率衰减', fontweight='bold')
            axes[0, 2].set_xlabel('训练轮数')
            axes[0, 2].set_ylabel('Epsilon值')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
        
        # 4. 分类别F1分数
        if anomaly_f1 and normal_f1:
            axes[1, 0].plot(episodes, anomaly_f1, 'r-', linewidth=2, label='异常类F1')
            axes[1, 0].plot(episodes, normal_f1, 'g-', linewidth=2, label='正常类F1')
            axes[1, 0].set_title('分类别F1分数', fontweight='bold')
            axes[1, 0].set_xlabel('训练轮数')
            axes[1, 0].set_ylabel('F1分数')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0, 1)
        
        # 5. 人工标注进展
        if human_annotations:
            axes[1, 1].plot(episodes, human_annotations, 'orange', linewidth=2, marker='o', label='标注样本数')
            axes[1, 1].set_title('人工标注进展', fontweight='bold')
            axes[1, 1].set_xlabel('训练轮数')
            axes[1, 1].set_ylabel('累计标注样本数')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        # 6. 人工标注准确率
        if human_accuracy:
            axes[1, 2].plot(episodes, human_accuracy, 'brown', linewidth=2, marker='s', label='人工标注准确率')
            axes[1, 2].set_title('人工标注样本模型准确率', fontweight='bold')
            axes[1, 2].set_xlabel('训练轮数')
            axes[1, 2].set_ylabel('准确率')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()
            axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_performance_analysis(self):
        """绘制模型性能分析图"""
        if not self.test_results:
            print("跳过性能分析（无测试结果数据）")
            return
            
        print("创建模型性能分析图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RLAD模型性能分析', fontsize=16, fontweight='bold')
        
        # 1. 性能指标总览
        metrics = ['精确率', '召回率', 'F1分数']
        values = [
            self.test_results.get('precision', 0),
            self.test_results.get('recall', 0),
            self.test_results.get('f1', 0)
        ]
        
        bars = axes[0, 0].bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        axes[0, 0].set_title('整体性能指标', fontweight='bold')
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].set_ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 分类别性能对比
        precision_per_class = self.test_results.get('precision_per_class', [])
        recall_per_class = self.test_results.get('recall_per_class', [])
        f1_per_class = self.test_results.get('f1_per_class', [])
        
        if precision_per_class and recall_per_class and f1_per_class:
            class_names = ['正常类', '异常类'][:len(precision_per_class)]
            x = np.arange(len(class_names))
            width = 0.25
            
            axes[0, 1].bar(x - width, precision_per_class, width, label='精确率', alpha=0.8, color='#3498db')
            axes[0, 1].bar(x, recall_per_class, width, label='召回率', alpha=0.8, color='#e74c3c')
            axes[0, 1].bar(x + width, f1_per_class, width, label='F1分数', alpha=0.8, color='#2ecc71')
            
            axes[0, 1].set_title('分类别性能对比', fontweight='bold')
            axes[0, 1].set_ylabel('分数')
            axes[0, 1].set_xlabel('类别')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(class_names)
            axes[0, 1].legend()
            axes[0, 1].set_ylim(0, 1)
        
        # 3. 性能雷达图
        if len(values) >= 3:
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values_radar = values + [values[0]]  # 闭合图形
            angles += angles[:1]
            
            ax_radar = plt.subplot(2, 2, 3, projection='polar')
            ax_radar.plot(angles, values_radar, 'o-', linewidth=2, color='#e74c3c')
            ax_radar.fill(angles, values_radar, alpha=0.25, color='#e74c3c')
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(metrics)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('性能雷达图', fontweight='bold', pad=20)
            ax_radar.grid(True)
        
        # 4. 训练历史性能趋势
        if self.training_history and 'val_f1' in self.training_history:
            episodes = self.training_history.get('episodes', [])
            val_f1 = self.training_history.get('val_f1', [])
            anomaly_f1 = self.training_history.get('anomaly_f1', [])
            normal_f1 = self.training_history.get('normal_f1', [])
            
            if val_f1:
                axes[1, 1].plot(episodes, val_f1, 'g-', linewidth=2, label='总体F1')
            if anomaly_f1:
                axes[1, 1].plot(episodes, anomaly_f1, 'r--', linewidth=2, label='异常类F1')
            if normal_f1:
                axes[1, 1].plot(episodes, normal_f1, 'b--', linewidth=2, label='正常类F1')
            
            axes[1, 1].set_title('训练过程性能趋势', fontweight='bold')
            axes[1, 1].set_xlabel('训练轮数')
            axes[1, 1].set_ylabel('F1分数')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_annotation_analysis(self):
        """绘制人工标注分析图"""
        if not self.manual_annotations:
            print("跳过人工标注分析（无标注数据）")
            return
            
        print("创建人工标注分析图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('人工标注分析', fontsize=16, fontweight='bold')
        
        # 提取标注数据
        annotations_df = pd.DataFrame(self.manual_annotations)
        
        if len(annotations_df) == 0:
            print("无人工标注数据可分析")
            return
        
        # 1. 标注标签分布
        if 'label' in annotations_df.columns:
            label_counts = annotations_df['label'].value_counts()
            label_names = ['正常' if x == 0 else '异常' for x in label_counts.index]
            colors = ['#2ecc71' if x == 0 else '#e74c3c' for x in label_counts.index]
            
            axes[0, 0].pie(label_counts.values, labels=label_names, autopct='%1.1f%%', 
                          colors=colors, startangle=90)
            axes[0, 0].set_title('人工标注标签分布', fontweight='bold')
        
        # 2. 标注时间线
        if 'timestamp' in annotations_df.columns:
            annotations_df['timestamp'] = pd.to_datetime(annotations_df['timestamp'])
            annotations_df = annotations_df.sort_values('timestamp')
            
            # 累计标注数量
            annotations_df['cumulative'] = range(1, len(annotations_df) + 1)
            
            axes[0, 1].plot(annotations_df['timestamp'], annotations_df['cumulative'], 
                           'o-', linewidth=2, markersize=6, color='#3498db')
            axes[0, 1].set_title('标注时间线', fontweight='bold')
            axes[0, 1].set_xlabel('时间')
            axes[0, 1].set_ylabel('累计标注数量')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. AI预测 vs 人工标注对比
        if 'auto_predicted_label' in annotations_df.columns and 'label' in annotations_df.columns:
            # 过滤掉auto_predicted_label为None的行
            comparison_df = annotations_df.dropna(subset=['auto_predicted_label'])
            
            if len(comparison_df) > 0:
                agreement = (comparison_df['auto_predicted_label'] == comparison_df['label']).mean()
                disagreement = 1 - agreement
                
                axes[1, 0].pie([agreement, disagreement], 
                              labels=[f'一致 ({agreement:.1%})', f'不一致 ({disagreement:.1%})'],
                              autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
                axes[1, 0].set_title('AI预测与人工标注一致性', fontweight='bold')
        
        # 4. 窗口统计信息分析
        if 'window_stats' in annotations_df.columns:
            stats_list = []
            for stats in annotations_df['window_stats']:
                if stats and isinstance(stats, dict):
                    stats_list.append(stats)
            
            if stats_list:
                stats_df = pd.DataFrame(stats_list)
                
                # 绘制均值分布的箱线图
                if 'mean' in stats_df.columns:
                    normal_means = []
                    anomaly_means = []
                    
                    for i, row in annotations_df.iterrows():
                        if row.get('window_stats') and isinstance(row['window_stats'], dict):
                            mean_val = row['window_stats'].get('mean')
                            if mean_val is not None:
                                if row['label'] == 0:
                                    normal_means.append(mean_val)
                                else:
                                    anomaly_means.append(mean_val)
                    
                    if normal_means and anomaly_means:
                        data_to_plot = [normal_means, anomaly_means]
                        axes[1, 1].boxplot(data_to_plot, labels=['正常', '异常'])
                        axes[1, 1].set_title('不同类别窗口数据均值分布', fontweight='bold')
                        axes[1, 1].set_ylabel('数据均值')
                        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "annotation_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_anomaly_detection_analysis(self):
        """绘制异常检测结果分析"""
        print("创建异常检测结果分析图...")
        
        # 尝试加载和重新处理数据进行分析
        self.load_raw_data()
        
        if self.raw_data is None:
            print("无法加载原始数据，跳过异常检测分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('异常检测结果分析', fontsize=16, fontweight='bold')
        
        # 1. 原始数据时间序列展示
        # 自动选择第一个数值列作为主要特征
        numeric_columns = self.raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            main_feature = numeric_columns[0]
            
            # 绘制原始时间序列
            axes[0, 0].plot(self.raw_data.index, self.raw_data[main_feature], 
                           'b-', linewidth=1, alpha=0.7, label='原始数据')
            
            # 添加异常阈值线
            Q1 = self.raw_data[main_feature].quantile(0.25)
            Q3 = self.raw_data[main_feature].quantile(0.75)
            IQR = Q3 - Q1
            threshold = Q3 + 1.5 * IQR
            
            axes[0, 0].axhline(y=threshold, color='r', linestyle='--', 
                              alpha=0.8, label=f'异常阈值 (Q3+1.5*IQR)')
            axes[0, 0].set_title(f'原始数据时间序列 - {main_feature}', fontweight='bold')
            axes[0, 0].set_xlabel('时间点')
            axes[0, 0].set_ylabel('数值')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 数据分布分析
        if len(numeric_columns) > 0:
            data_values = self.raw_data[main_feature].dropna()
            
            axes[0, 1].hist(data_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].axvline(x=data_values.mean(), color='r', linestyle='-', 
                              label=f'均值: {data_values.mean():.2f}')
            axes[0, 1].axvline(x=threshold, color='orange', linestyle='--', 
                              label=f'异常阈值: {threshold:.2f}')
            axes[0, 1].set_title('数据值分布', fontweight='bold')
            axes[0, 1].set_xlabel('数值')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 滑动窗口异常检测模拟
        if len(numeric_columns) > 0:
            window_size = self.config.get('window_size', 288)
            stride = self.config.get('stride', 12)
            
            # 计算滑动窗口的最大值
            window_maxes = []
            window_indices = []
            
            for i in range(0, len(self.raw_data) - window_size + 1, stride):
                window_data = self.raw_data[main_feature].iloc[i:i+window_size]
                window_maxes.append(window_data.max())
                window_indices.append(i + window_size // 2)  # 窗口中心位置
            
            # 标记异常窗口
            anomaly_windows = np.array(window_maxes) > threshold
            
            axes[1, 0].plot(window_indices, window_maxes, 'b-', linewidth=1, alpha=0.7, label='窗口最大值')
            axes[1, 0].scatter(np.array(window_indices)[anomaly_windows], 
                              np.array(window_maxes)[anomaly_windows], 
                              color='red', s=20, alpha=0.8, label='异常窗口')
            axes[1, 0].axhline(y=threshold, color='orange', linestyle='--', 
                              alpha=0.8, label='异常阈值')
            axes[1, 0].set_title('滑动窗口异常检测结果', fontweight='bold')
            axes[1, 0].set_xlabel('窗口中心位置')
            axes[1, 0].set_ylabel('窗口最大值')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 异常统计
            total_windows = len(window_maxes)
            anomaly_count = np.sum(anomaly_windows)
            normal_count = total_windows - anomaly_count
            
            labels = ['正常窗口', '异常窗口']
            sizes = [normal_count, anomaly_count]
            colors = ['#2ecc71', '#e74c3c']
            
            axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            axes[1, 1].set_title(f'窗口异常检测统计\n(总窗口数: {total_windows})', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "anomaly_detection_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
# PART 2/2

    def plot_comparison_analysis(self):
        """绘制对比分析图"""
        print("创建对比分析图...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RLAD模型对比分析', fontsize=16, fontweight='bold')
        
        # 1. 训练策略对比（人工标注 vs 自动标注）
        if self.training_history and 'human_annotations_count' in self.training_history:
            episodes = self.training_history.get('episodes', [])
            human_count = self.training_history.get('human_annotations_count', [])
            val_f1 = self.training_history.get('val_f1', [])
            
            if human_count and val_f1:
                # 创建双y轴图
                ax1 = axes[0, 0]
                ax2 = ax1.twinx()
                
                line1 = ax1.plot(episodes, human_count, 'o-', color='#e74c3c', 
                                linewidth=2, label='人工标注数量')
                line2 = ax2.plot(episodes, val_f1, 's-', color='#2ecc71', 
                                linewidth=2, label='验证F1分数')
                
                ax1.set_xlabel('训练轮数')
                ax1.set_ylabel('人工标注数量', color='#e74c3c')
                ax2.set_ylabel('验证F1分数', color='#2ecc71')
                ax1.set_title('人工标注与模型性能关系', fontweight='bold')
                
                # 合并图例
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
                
                ax1.grid(True, alpha=0.3)
        
        # 2. 不同类别检测性能对比
        precision_per_class = self.test_results.get('precision_per_class', [])
        recall_per_class = self.test_results.get('recall_per_class', [])
        f1_per_class = self.test_results.get('f1_per_class', [])
        
        if precision_per_class and recall_per_class and f1_per_class:
            # 确保有两个类别的数据
            if len(precision_per_class) >= 2:
                metrics = ['精确率', '召回率', 'F1分数']
                normal_scores = [precision_per_class[0], recall_per_class[0], f1_per_class[0]]
                anomaly_scores = [precision_per_class[1] if len(precision_per_class) > 1 else 0,
                                recall_per_class[1] if len(recall_per_class) > 1 else 0,
                                f1_per_class[1] if len(f1_per_class) > 1 else 0]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                bars1 = axes[0, 1].bar(x - width/2, normal_scores, width, 
                                      label='正常类', color='#2ecc71', alpha=0.8)
                bars2 = axes[0, 1].bar(x + width/2, anomaly_scores, width, 
                                      label='异常类', color='#e74c3c', alpha=0.8)
                
                axes[0, 1].set_title('正常类 vs 异常类性能对比', fontweight='bold')
                axes[0, 1].set_ylabel('分数')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(metrics)
                axes[0, 1].legend()
                axes[0, 1].set_ylim(0, 1)
                
                # 添加数值标签
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 训练效率分析
        if self.training_history:
            episodes = self.training_history.get('episodes', [])
            train_loss = self.training_history.get('train_loss', [])
            val_f1 = self.training_history.get('val_f1', [])
            
            if episodes and val_f1:
                # 计算训练效率：每轮的性能提升
                if len(val_f1) > 1:
                    improvement_rate = np.diff(val_f1)
                    axes[0, 2].plot(episodes[1:], improvement_rate, 'o-', 
                                   linewidth=2, color='#9b59b6', label='F1改进率')
                    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    axes[0, 2].set_title('训练效率分析', fontweight='bold')
                    axes[0, 2].set_xlabel('训练轮数')
                    axes[0, 2].set_ylabel('F1分数改进率')
                    axes[0, 2].legend()
                    axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 人工标注质量分析
        if self.manual_annotations:
            annotations_df = pd.DataFrame(self.manual_annotations)
            
            # AI预测准确性分析
            if 'auto_predicted_label' in annotations_df.columns and 'label' in annotations_df.columns:
                comparison_df = annotations_df.dropna(subset=['auto_predicted_label'])
                
                if len(comparison_df) > 0:
                    # 混淆矩阵
                    y_true = comparison_df['label'].values
                    y_pred = comparison_df['auto_predicted_label'].values
                    
                    cm = confusion_matrix(y_true, y_pred)
                    
                    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
                    axes[1, 0].set_title('AI预测混淆矩阵\n(与人工标注对比)', fontweight='bold')
                    
                    # 添加数值标签
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                                           ha="center", va="center",
                                           color="white" if cm[i, j] > thresh else "black",
                                           fontsize=14, fontweight='bold')
                    
                    axes[1, 0].set_ylabel('人工标注真实值')
                    axes[1, 0].set_xlabel('AI预测值')
                    axes[1, 0].set_xticks([0, 1])
                    axes[1, 0].set_yticks([0, 1])
                    axes[1, 0].set_xticklabels(['正常', '异常'])
                    axes[1, 0].set_yticklabels(['正常', '异常'])
        
        # 5. 模型收敛性分析
        if self.training_history and 'val_f1' in self.training_history:
            episodes = self.training_history.get('episodes', [])
            val_f1 = self.training_history.get('val_f1', [])
            
            if len(val_f1) > 5:  # 至少需要5个点来分析趋势
                # 计算滑动平均
                window = min(5, len(val_f1) // 3)
                val_f1_ma = pd.Series(val_f1).rolling(window=window).mean()
                
                axes[1, 1].plot(episodes, val_f1, 'o-', alpha=0.6, color='lightblue', 
                               label='原始F1分数')
                axes[1, 1].plot(episodes, val_f1_ma, '-', linewidth=3, color='#3498db', 
                               label=f'{window}点滑动平均')
                
                # 标记最佳性能点
                best_idx = np.argmax(val_f1)
                axes[1, 1].scatter(episodes[best_idx], val_f1[best_idx], 
                                  color='red', s=100, zorder=5, label='最佳性能点')
                
                axes[1, 1].set_title('模型收敛性分析', fontweight='bold')
                axes[1, 1].set_xlabel('训练轮数')
                axes[1, 1].set_ylabel('验证F1分数')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim(0, 1)
        
        # 6. 综合性能雷达图对比
        if self.test_results:
            # 创建性能指标对比
            metrics = ['精确率', '召回率', 'F1分数']
            current_scores = [
                self.test_results.get('precision', 0),
                self.test_results.get('recall', 0),
                self.test_results.get('f1', 0)
            ]
            
            # 假设的基线性能（可以根据实际情况调整）
            baseline_scores = [0.8, 0.7, 0.75]  # 一般传统方法的性能
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            current_scores += current_scores[:1]  # 闭合图形
            baseline_scores += baseline_scores[:1]
            angles += angles[:1]
            
            ax_radar = plt.subplot(2, 3, 6, projection='polar')
            ax_radar.plot(angles, current_scores, 'o-', linewidth=2, 
                         color='#e74c3c', label='RLAD模型')
            ax_radar.fill(angles, current_scores, alpha=0.25, color='#e74c3c')
            ax_radar.plot(angles, baseline_scores, 'o-', linewidth=2, 
                         color='#95a5a6', label='传统基线')
            ax_radar.fill(angles, baseline_scores, alpha=0.25, color='#95a5a6')
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(metrics)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('RLAD vs 传统方法\n性能对比', fontweight='bold', pad=20)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax_radar.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_analysis_report(self):
        """生成详细的分析报告"""
        print("生成分析报告...")
        
        report_content = []
        report_content.append("# RLAD模型训练与异常检测分析报告")
        report_content.append(f"## 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # 1. 基本信息
        report_content.append("## 1. 基本配置信息")
        if self.config:
            report_content.append(f"- 数据路径: {self.config.get('data_path', 'N/A')}")
            report_content.append(f"- 特征列: {self.config.get('actual_feature_column_used', 'N/A')}")
            report_content.append(f"- 窗口大小: {self.config.get('window_size', 'N/A')}")
            report_content.append(f"- 步长: {self.config.get('stride', 'N/A')}")
            report_content.append(f"- 学习率: {self.config.get('lr', 'N/A')}")
            report_content.append(f"- 训练轮数: {self.config.get('num_episodes', 'N/A')}")
            report_content.append(f"- 标注频率: {self.config.get('annotation_frequency', 'N/A')}")
        report_content.append("")
        
        # 2. 训练结果总结
        report_content.append("## 2. 训练结果总结")
        if self.training_history:
            episodes = self.training_history.get('episodes', [])
            if episodes:
                report_content.append(f"- 实际训练轮数: {max(episodes)}")
                
            val_f1 = self.training_history.get('val_f1', [])
            if val_f1:
                report_content.append(f"- 最终验证F1分数: {val_f1[-1]:.4f}")
                report_content.append(f"- 最佳验证F1分数: {max(val_f1):.4f}")
            
            human_count = self.training_history.get('human_annotations_count', [])
            if human_count:
                report_content.append(f"- 总人工标注数量: {max(human_count) if human_count else 0}")
        report_content.append("")
        
        # 3. 测试性能
        report_content.append("## 3. 最终测试性能")
        if self.test_results:
            report_content.append(f"- 整体精确率: {self.test_results.get('precision', 0):.4f}")
            report_content.append(f"- 整体召回率: {self.test_results.get('recall', 0):.4f}")
            report_content.append(f"- 整体F1分数: {self.test_results.get('f1', 0):.4f}")
            
            precision_per_class = self.test_results.get('precision_per_class', [])
            recall_per_class = self.test_results.get('recall_per_class', [])
            f1_per_class = self.test_results.get('f1_per_class', [])
            
            if precision_per_class and len(precision_per_class) >= 2:
                report_content.append(f"- 正常类精确率: {precision_per_class[0]:.4f}")
                report_content.append(f"- 异常类精确率: {precision_per_class[1]:.4f}")
            
            if recall_per_class and len(recall_per_class) >= 2:
                report_content.append(f"- 正常类召回率: {recall_per_class[0]:.4f}")
                report_content.append(f"- 异常类召回率: {recall_per_class[1]:.4f}")
            
            if f1_per_class and len(f1_per_class) >= 2:
                report_content.append(f"- 正常类F1分数: {f1_per_class[0]:.4f}")
                report_content.append(f"- 异常类F1分数: {f1_per_class[1]:.4f}")
        report_content.append("")
        
        # 4. 人工标注分析
        report_content.append("## 4. 人工标注分析")
        if self.manual_annotations:
            annotations_df = pd.DataFrame(self.manual_annotations)
            
            report_content.append(f"- 总标注样本数: {len(annotations_df)}")
            
            if 'label' in annotations_df.columns:
                label_counts = annotations_df['label'].value_counts()
                normal_count = label_counts.get(0, 0)
                anomaly_count = label_counts.get(1, 0)
                report_content.append(f"- 标注为正常的样本: {normal_count}")
                report_content.append(f"- 标注为异常的样本: {anomaly_count}")
                report_content.append(f"- 异常样本比例: {anomaly_count/(normal_count+anomaly_count)*100:.1f}%")
            
            # AI预测一致性
            if ('auto_predicted_label' in annotations_df.columns and 
                'label' in annotations_df.columns):
                comparison_df = annotations_df.dropna(subset=['auto_predicted_label'])
                if len(comparison_df) > 0:
                    agreement = (comparison_df['auto_predicted_label'] == comparison_df['label']).mean()
                    report_content.append(f"- AI预测与人工标注一致率: {agreement:.1%}")
        else:
            report_content.append("- 未进行人工标注")
        report_content.append("")
        
        # 5. 模型优势分析
        report_content.append("## 5. 模型优势分析")
        if self.test_results:
            f1_score = self.test_results.get('f1', 0)
            if f1_score > 0.9:
                report_content.append("- ✅ 模型达到优秀性能水平 (F1 > 0.9)")
            elif f1_score > 0.8:
                report_content.append("- ✅ 模型达到良好性能水平 (F1 > 0.8)")
            else:
                report_content.append("- ⚠️ 模型性能有待提升 (F1 ≤ 0.8)")
            
            precision = self.test_results.get('precision', 0)
            recall = self.test_results.get('recall', 0)
            
            if precision > 0.95:
                report_content.append("- ✅ 模型具有很高的精确率，误报率低")
            if recall > 0.95:
                report_content.append("- ✅ 模型具有很高的召回率，漏报率低")
            
            # 分析精确率和召回率的平衡
            if abs(precision - recall) < 0.05:
                report_content.append("- ✅ 精确率和召回率平衡良好")
            elif precision > recall:
                report_content.append("- ⚠️ 精确率高于召回率，模型较为保守")
            else:
                report_content.append("- ⚠️ 召回率高于精确率，模型较为激进")
        report_content.append("")
        
        # 6. 改进建议
        report_content.append("## 6. 改进建议")
        
        if self.manual_annotations:
            human_count = len(self.manual_annotations)
            if human_count < 50:
                report_content.append("- 建议增加人工标注样本数量以提高模型性能")
        
        if self.training_history:
            episodes = self.training_history.get('episodes', [])
            if episodes and max(episodes) < 50:
                report_content.append("- 可以考虑增加训练轮数以获得更好的收敛效果")
        
        if self.test_results:
            f1_per_class = self.test_results.get('f1_per_class', [])
            if len(f1_per_class) >= 2:
                if f1_per_class[1] < f1_per_class[0] - 0.1:  # 异常类性能明显低于正常类
                    report_content.append("- 异常类检测性能相对较低，建议增加异常样本的标注")
                elif f1_per_class[0] < f1_per_class[1] - 0.1:  # 正常类性能明显低于异常类
                    report_content.append("- 正常类检测性能相对较低，建议增加正常样本的标注")
        
        report_content.append("- 可以考虑调整网络结构参数（如隐藏层大小、层数等）")
        report_content.append("- 可以尝试不同的学习率和优化器设置")
        report_content.append("- 建议对更多样化的数据进行测试以验证模型泛化能力")
        report_content.append("")
        
        # 7. 结论
        report_content.append("## 7. 结论")
        if self.test_results:
            f1_score = self.test_results.get('f1', 0)
            if f1_score > 0.9:
                report_content.append("RLAD模型在液压支架异常检测任务上表现优秀，")
                report_content.append("结合强化学习和人工标注的方法有效提升了检测性能。")
            else:
                report_content.append("RLAD模型在当前设置下取得了一定的性能，")
                report_content.append("建议根据改进建议进一步优化模型配置。")
        
        report_content.append("")
        report_content.append("---")
        report_content.append("*此报告由RLAD可视化分析工具自动生成*")
        
        # 保存报告
        report_path = self.output_dir / "analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"分析报告已保存到: {report_path}")
        
        # 同时保存为txt格式
        txt_report_path = self.output_dir / "analysis_report.txt"
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        return report_content
    
    def create_summary_dashboard(self):
        """创建总结仪表板"""
        print("创建总结仪表板...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 主标题
        fig.suptitle('RLAD液压支架异常检测 - 综合仪表板', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. 关键指标展示 (占用2x2区域)
        ax_metrics = fig.add_subplot(gs[0:2, 0:2])
        
        if self.test_results:
            metrics = ['精确率', '召回率', 'F1分数']
            values = [
                self.test_results.get('precision', 0),
                self.test_results.get('recall', 0),
                self.test_results.get('f1', 0)
            ]
            
            # 创建大的指标显示
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            bars = ax_metrics.barh(metrics, values, color=colors, alpha=0.8)
            
            ax_metrics.set_xlim(0, 1)
            ax_metrics.set_title('核心性能指标', fontsize=16, fontweight='bold', pad=20)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                ax_metrics.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{value:.4f}', va='center', fontweight='bold', fontsize=14)
        
        # 2. 训练进度
        ax_training = fig.add_subplot(gs[0, 2:])
        if self.training_history and 'val_f1' in self.training_history:
            episodes = self.training_history.get('episodes', [])
            val_f1 = self.training_history.get('val_f1', [])
            
            ax_training.plot(episodes, val_f1, 'o-', linewidth=3, markersize=8, color='#2ecc71')
            ax_training.set_title('训练进度 (验证F1分数)', fontweight='bold')
            ax_training.set_xlabel('训练轮数')
            ax_training.set_ylabel('F1分数')
            ax_training.grid(True, alpha=0.3)
            ax_training.set_ylim(0, 1)
        
        # 3. 类别性能对比
        ax_class = fig.add_subplot(gs[1, 2:])
        if self.test_results:
            f1_per_class = self.test_results.get('f1_per_class', [])
            if len(f1_per_class) >= 2:
                classes = ['正常类', '异常类']
                values = f1_per_class[:2]
                colors = ['#2ecc71', '#e74c3c']
                
                bars = ax_class.bar(classes, values, color=colors, alpha=0.8)
                ax_class.set_title('分类别F1性能', fontweight='bold')
                ax_class.set_ylabel('F1分数')
                ax_class.set_ylim(0, 1)
                
                for bar, value in zip(bars, values):
                    ax_class.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 人工标注统计
        ax_annotation = fig.add_subplot(gs[2, 0])
        if self.manual_annotations:
            annotations_df = pd.DataFrame(self.manual_annotations)
            if 'label' in annotations_df.columns:
                label_counts = annotations_df['label'].value_counts()
                labels = ['正常' if x == 0 else '异常' for x in label_counts.index]
                colors = ['#2ecc71' if x == 0 else '#e74c3c' for x in label_counts.index]
                
                ax_annotation.pie(label_counts.values, labels=labels, autopct='%1.1f%%',
                                 colors=colors, startangle=90)
                ax_annotation.set_title(f'人工标注分布\n(总计: {len(annotations_df)}样本)', 
                                       fontweight='bold')
        
        # 5. AI预测一致性
        ax_agreement = fig.add_subplot(gs[2, 1])
        if self.manual_annotations:
            annotations_df = pd.DataFrame(self.manual_annotations)
            if ('auto_predicted_label' in annotations_df.columns and 
                'label' in annotations_df.columns):
                comparison_df = annotations_df.dropna(subset=['auto_predicted_label'])
                if len(comparison_df) > 0:
                    agreement = (comparison_df['auto_predicted_label'] == comparison_df['label']).mean()
                    disagreement = 1 - agreement
                    
                    ax_agreement.pie([agreement, disagreement], 
                                   labels=[f'一致\n{agreement:.1%}', f'不一致\n{disagreement:.1%}'],
                                   colors=['#2ecc71', '#e74c3c'], startangle=90, autopct='')
                    ax_agreement.set_title('AI预测一致性', fontweight='bold')
        
        # 6. 模型配置信息
        ax_config = fig.add_subplot(gs[2, 2:])
        ax_config.axis('off')
        
        config_text = []
        if self.config:
            config_text.append(f"窗口大小: {self.config.get('window_size', 'N/A')}")
            config_text.append(f"步长: {self.config.get('stride', 'N/A')}")
            config_text.append(f"学习率: {self.config.get('lr', 'N/A')}")
            config_text.append(f"隐藏层大小: {self.config.get('hidden_size', 'N/A')}")
            
        if self.training_history:
            episodes = self.training_history.get('episodes', [])
            if episodes:
                config_text.append(f"训练轮数: {max(episodes)}")
            
            human_count = self.training_history.get('human_annotations_count', [])
            if human_count:
                config_text.append(f"人工标注: {max(human_count)}样本")
        
        config_text.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        ax_config.text(0.05, 0.95, '模型配置信息', fontsize=14, fontweight='bold', 
                      transform=ax_config.transAxes, va='top')
        
        for i, text in enumerate(config_text):
            ax_config.text(0.05, 0.85 - i*0.12, f"• {text}", fontsize=11, 
                          transform=ax_config.transAxes, va='top')
        
        plt.savefig(self.output_dir / "summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
def find_result_directories():
    """自动查找结果目录"""
    current_dir = Path.cwd()
    result_dirs = []
    
    # 查找可能的结果目录
    patterns = [
        "output_gui/rlad_gui_*",
        "output_gui\\rlad_gui_*", 
        "output/rlad_*",
        "output\\rlad_*"
    ]
    
    for pattern in patterns:
        matching_dirs = list(current_dir.glob(pattern))
        result_dirs.extend(matching_dirs)
    
    # 过滤出包含必要文件的目录
    valid_dirs = []
    for dir_path in result_dirs:
        if (dir_path / "config.json").exists():
            valid_dirs.append(dir_path)
    
    return valid_dirs
def main():
    """主函数 - 改进版"""
    parser = argparse.ArgumentParser(description='RLAD结果可视化分析工具')
    parser.add_argument('--result_dir', type=str,
                       help='RLAD训练结果目录路径')
    parser.add_argument('--create_dashboard', action='store_true',
                       help='创建综合仪表板')
    parser.add_argument('--auto_find', action='store_true',
                       help='自动查找结果目录')
    
    args = parser.parse_args()
    
    # 自动查找模式
    if args.auto_find or not args.result_dir:
        print("正在自动查找结果目录...")
        valid_dirs = find_result_directories()
        
        if not valid_dirs:
            print("未找到有效的结果目录")
            print("请确保目录中包含以下文件:")
            print("- config.json")
            print("- training_history_final.json")
            print("- test_results_final.json")
            return
        
        print(f"找到 {len(valid_dirs)} 个有效的结果目录:")
        for i, dir_path in enumerate(valid_dirs, 1):
            print(f"{i}. {dir_path}")
        
        if len(valid_dirs) == 1:
            result_dir = valid_dirs[0]
            print(f"自动选择: {result_dir}")
        else:
            while True:
                try:
                    choice = input(f"请选择要分析的目录 (1-{len(valid_dirs)}): ")
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(valid_dirs):
                        result_dir = valid_dirs[choice_idx]
                        break
                    else:
                        print("无效的选择，请重新输入")
                except ValueError:
                    print("请输入数字")
    else:
        result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        print(f"错误: 结果目录不存在: {result_dir}")
        return
    
    print(f"开始分析RLAD结果: {result_dir}")
    
    try:
        # 创建分析器
        analyzer = RLADVisualizationAnalyzer(str(result_dir))
        
        # 执行综合分析
        analyzer.create_comprehensive_analysis()
        
        # 创建仪表板
        analyzer.create_summary_dashboard()
        
        print(f"\n✅ 分析完成！所有可视化结果已保存到: {analyzer.output_dir}")
        print("\n📊 生成的文件包括:")
        print("- training_analysis.png: 训练过程分析")
        print("- performance_analysis.png: 模型性能分析") 
        print("- annotation_analysis.png: 人工标注分析")
        print("- anomaly_detection_analysis.png: 异常检测结果分析")
        print("- comparison_analysis.png: 对比分析")
        print("- summary_dashboard.png: 综合仪表板")
        print("- analysis_report.md/txt: 详细分析报告")
        
        # 打开结果目录
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            subprocess.run(f'explorer "{analyzer.output_dir}"', shell=True)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(f'open "{analyzer.output_dir}"', shell=True)
        else:  # Linux
            subprocess.run(f'xdg-open "{analyzer.output_dir}"', shell=True)
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) == 1:
        print("RLAD可视化分析工具")
        print("\n使用方法:")
        print("python RLAD_visualization_analysis.py --result_dir <结果目录路径> [--create_dashboard]")
        
        # 修改为您的实际路径
        example_dir = r"C:\Users\Liu HaoTian\Desktop\Python files\deeplearning\example\timeseries\examples\RLAD\output_gui\rlad_gui_1__20250619_010018"
        
        if os.path.exists(example_dir):
            print(f"\n发现目录，开始分析: {example_dir}")
            analyzer = RLADVisualizationAnalyzer(example_dir)
            analyzer.create_comprehensive_analysis()
            analyzer.create_summary_dashboard()
            print(f"✅ 分析完成！结果保存在: {analyzer.output_dir}")
            
            # 自动打开结果文件夹
            import subprocess
            subprocess.run(f'explorer "{analyzer.output_dir}"', shell=True)
        else:
            print(f"\n❌ 未找到目录: {example_dir}")
            print("请检查路径是否正确")
    else:
        main()