#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试高级可视化功能
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 确保matplotlib使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加主脚本的路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_visualizations():
    """测试三种可视化方法"""
    
    print("🚀 开始测试高级可视化功能...")
    
    # 导入必要的类
    try:
        exec(open('RLADv3_2_TRUE_copy copy.py').read(), globals())
        print("✅ 成功导入RLAD模块")
    except Exception as e:
        print(f"❌ 导入模块失败: {e}")
        return
    
    # 创建测试数据
    np.random.seed(42)
    
    # 生成1000个时间步的压力信号（模拟真实的工业数据）
    t = np.linspace(0, 100, 1000)
    
    # 基础信号：包含趋势、季节性和噪声
    trend = 0.01 * t + 50  # 缓慢上升趋势
    seasonal = 5 * np.sin(2 * np.pi * t / 24) + 2 * np.sin(2 * np.pi * t / 12)  # 日周期和半日周期
    noise = np.random.normal(0, 1, len(t))
    
    # 添加一些异常点
    anomaly_indices = [100, 150, 200, 300, 450, 600, 750, 850]
    anomaly_signal = np.zeros_like(t)
    for idx in anomaly_indices:
        if idx < len(t):
            anomaly_signal[idx:idx+10] = np.random.normal(0, 8, min(10, len(t)-idx))
    
    original_data = trend + seasonal + noise + anomaly_signal
    
    # 创建窗口索引和标签
    window_size = 50
    stride = 25
    window_indices = list(range(0, len(original_data) - window_size + 1, stride))
    
    # 生成真实标签（基于异常点位置）
    true_labels = []
    for start_idx in window_indices:
        end_idx = start_idx + window_size
        # 如果窗口中包含异常点，标记为异常
        contains_anomaly = any(idx >= start_idx and idx < end_idx for idx in anomaly_indices)
        true_labels.append(1 if contains_anomaly else 0)
    
    # 生成模拟的异常分数（与真实标签相关但有噪声）
    scores = []
    for label in true_labels:
        if label == 1:
            # 异常窗口：高分数 + 噪声
            score = np.random.uniform(0.6, 1.0)
        else:
            # 正常窗口：低分数 + 噪声
            score = np.random.uniform(0.0, 0.4)
        scores.append(score)
    
    scores = np.array(scores)
    
    # 创建输出目录
    output_dir = "./test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建可视化器实例
    try:
        visualizer = AdvancedRLADVisualizer(output_dir)
        print("✅ 成功创建可视化器")
        
        # 测试三种可视化方法
        print("\n📊 测试双子图版本...")
        aligned_path = visualizer.plot_anomaly_case_study_with_stl(
            original_data, window_indices, true_labels, scores, window_size,
            os.path.join(output_dir, "test_aligned.pdf")
        )
        
        print("🔄 测试融合版本...")
        unified_path = visualizer.plot_anomaly_case_study_unified(
            original_data, window_indices, true_labels, scores, window_size,
            os.path.join(output_dir, "test_unified.pdf")
        )
        
        print("🎯 测试三合一高级版本...")
        advanced_path = visualizer.plot_anomaly_case_study_advanced(
            original_data, window_indices, true_labels, scores, window_size,
            os.path.join(output_dir, "test_advanced.pdf")
        )
        
        print(f"\n✅ 所有测试完成！可视化文件保存在: {output_dir}")
        print(f"   📊 对齐双子图: {aligned_path}")
        print(f"   🔄 融合单图: {unified_path}")
        print(f"   🎯 三合一高级版本: {advanced_path}")
        
        # 生成数据统计报告
        print(f"\n📈 测试数据统计:")
        print(f"   时间序列长度: {len(original_data)}")
        print(f"   窗口数量: {len(window_indices)}")
        print(f"   异常窗口数: {sum(true_labels)}")
        print(f"   正常窗口数: {len(true_labels) - sum(true_labels)}")
        print(f"   平均异常分数: {np.mean(scores):.3f}")
        print(f"   异常分数标准差: {np.std(scores):.3f}")
        
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualizations()
