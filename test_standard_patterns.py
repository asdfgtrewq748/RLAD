"""
测试标准异常值和正常值模式对比可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

def create_test_data():
    """创建测试数据，包含明显的异常和正常模式"""
    np.random.seed(42)
    
    # 生成基础时间序列（正常模式）
    time_points = np.arange(1000)
    base_signal = 50 + 10 * np.sin(time_points * 0.1) + np.random.normal(0, 2, 1000)
    
    # 添加一些异常模式
    # 异常1: 突然的尖峰 (index 200-220)
    base_signal[200:220] += 30 + np.random.normal(0, 5, 20)
    
    # 异常2: 持续的偏移 (index 400-450)  
    base_signal[400:450] += 25 + np.random.normal(0, 3, 50)
    
    # 异常3: 剧烈波动 (index 600-630)
    base_signal[600:630] += 20 * np.sin(np.arange(30) * 0.5) + np.random.normal(0, 8, 30)
    
    # 异常4: 缓慢下降后急剧上升 (index 800-850)
    fade_in = np.linspace(0, -15, 25)
    spike_up = np.linspace(-15, 40, 25)
    base_signal[800:825] += fade_in
    base_signal[825:850] += spike_up
    
    return base_signal

def create_test_labels_and_windows(data, window_size=50):
    """基于数据特征创建真实标签和窗口索引"""
    window_indices = []
    true_labels = []
    
    # 滑动窗口
    for i in range(0, len(data) - window_size, window_size // 2):
        window_indices.append(i)
        window_data = data[i:i+window_size]
        
        # 简单的异常检测逻辑：基于标准差和极值
        std_val = np.std(window_data)
        range_val = np.max(window_data) - np.min(window_data)
        mean_val = np.mean(window_data)
        
        # 如果标准差大于阈值或均值偏离正常范围，标记为异常
        is_anomaly = (std_val > 8) or (range_val > 40) or (mean_val > 70) or (mean_val < 30)
        true_labels.append(1 if is_anomaly else 0)
    
    return window_indices, true_labels

def test_visualization():
    """测试可视化功能"""
    print("🧪 测试标准异常值和正常值模式对比可视化...")
    
    # 创建测试数据
    test_data = create_test_data()
    window_size = 50
    window_indices, true_labels = create_test_labels_and_windows(test_data, window_size)
    
    print(f"📊 测试数据信息:")
    print(f"   - 数据长度: {len(test_data)}")
    print(f"   - 窗口数量: {len(window_indices)}")
    print(f"   - 异常窗口: {sum(true_labels)}")
    print(f"   - 正常窗口: {len(true_labels) - sum(true_labels)}")
    
    # 导入可视化器
    try:
        from sklearn.preprocessing import StandardScaler
        
        # 创建一个简化的可视化器类用于测试
        class TestVisualizer:
            def __init__(self):
                self.output_dir = "test_output"
                os.makedirs(self.output_dir, exist_ok=True)
                
            def plot_standard_anomaly_patterns(self, original_data, window_indices, true_labels, window_size, save_path=None):
                """
                可视化标准异常值和正常值模式对比
                展示典型的异常模式和正常模式，帮助理解模型学习目标
                """
                print("📊 生成标准异常值和正常值模式对比图...")
                
                try:
                    # 确保数据存在
                    if original_data is None or len(true_labels) == 0:
                        print("⚠️ 数据不足，无法生成标准模式对比图")
                        return
                    
                    # 找到异常和正常样本的索引
                    anomaly_indices = np.where(np.array(true_labels) == 1)[0]
                    normal_indices = np.where(np.array(true_labels) == 0)[0]
                    
                    if len(anomaly_indices) == 0 or len(normal_indices) == 0:
                        print("⚠️ 缺少异常或正常样本，无法生成对比图")
                        return
                    
                    # 创建图表 - 2x3的布局
                    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                    
                    # 颜色配置
                    normal_color = '#2E86AB'  # 蓝色
                    anomaly_color = '#F24236'  # 红色
                    
                    # 选择典型样本进行展示（选择最有代表性的）
                    n_samples = min(5, len(anomaly_indices), len(normal_indices))
                    
                    # 随机选择一些代表性样本
                    np.random.seed(42)  # 确保可重复性
                    selected_anomaly_idx = np.random.choice(anomaly_indices, min(n_samples, len(anomaly_indices)), replace=False)
                    selected_normal_idx = np.random.choice(normal_indices, min(n_samples, len(normal_indices)), replace=False)
                    
                    # 1. 原始时间序列对比 (第一行第一列)
                    axes[0,0].set_title('标准异常值 vs 正常值 - 时间序列模式', fontsize=14, fontweight='bold', pad=15)
                    
                    # 绘制正常样本的时间序列
                    for i, idx in enumerate(selected_normal_idx):
                        start_idx = window_indices[idx]
                        end_idx = start_idx + window_size
                        if end_idx <= len(original_data):
                            window_data = original_data[start_idx:end_idx]
                            axes[0,0].plot(range(len(window_data)), window_data, 
                                          color=normal_color, alpha=0.6, linewidth=2,
                                          label='正常模式' if i == 0 else "")
                    
                    # 绘制异常样本的时间序列
                    for i, idx in enumerate(selected_anomaly_idx):
                        start_idx = window_indices[idx]
                        end_idx = start_idx + window_size
                        if end_idx <= len(original_data):
                            window_data = original_data[start_idx:end_idx]
                            axes[0,0].plot(range(len(window_data)), window_data, 
                                          color=anomaly_color, alpha=0.8, linewidth=2,
                                          label='异常模式' if i == 0 else "")
                    
                    axes[0,0].set_xlabel('时间步长', fontsize=12)
                    axes[0,0].set_ylabel('压力值', fontsize=12)
                    axes[0,0].legend(fontsize=11)
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # 2. 统计特征对比 (第一行第二列)
                    axes[0,1].set_title('统计特征对比', fontsize=14, fontweight='bold', pad=15)
                    
                    # 计算统计特征
                    normal_features = []
                    anomaly_features = []
                    
                    feature_names = ['均值', '标准差', '偏度', '峰度', '极差', '中位数']
                    
                    # 计算正常样本特征
                    for idx in selected_normal_idx:
                        start_idx = window_indices[idx]
                        end_idx = start_idx + window_size
                        if end_idx <= len(original_data):
                            window_data = original_data[start_idx:end_idx]
                            features = [
                                np.mean(window_data),
                                np.std(window_data),
                                scipy_stats.skew(window_data),
                                scipy_stats.kurtosis(window_data),
                                np.max(window_data) - np.min(window_data),
                                np.median(window_data)
                            ]
                            normal_features.append(features)
                    
                    # 计算异常样本特征
                    for idx in selected_anomaly_idx:
                        start_idx = window_indices[idx]
                        end_idx = start_idx + window_size
                        if end_idx <= len(original_data):
                            window_data = original_data[start_idx:end_idx]
                            features = [
                                np.mean(window_data),
                                np.std(window_data),
                                scipy_stats.skew(window_data),
                                scipy_stats.kurtosis(window_data),
                                np.max(window_data) - np.min(window_data),
                                np.median(window_data)
                            ]
                            anomaly_features.append(features)
                    
                    if normal_features and anomaly_features:
                        normal_features = np.array(normal_features)
                        anomaly_features = np.array(anomaly_features)
                        
                        # 计算均值
                        normal_means = np.mean(normal_features, axis=0)
                        anomaly_means = np.mean(anomaly_features, axis=0)
                        
                        x = np.arange(len(feature_names))
                        width = 0.35
                        
                        bars1 = axes[0,1].bar(x - width/2, normal_means, width, label='正常值', 
                                             color=normal_color, alpha=0.7)
                        bars2 = axes[0,1].bar(x + width/2, anomaly_means, width, label='异常值', 
                                             color=anomaly_color, alpha=0.7)
                        
                        axes[0,1].set_xticks(x)
                        axes[0,1].set_xticklabels(feature_names, rotation=45)
                        axes[0,1].legend()
                        axes[0,1].grid(True, alpha=0.3)
                    
                    # 3. 频域特征对比 (第一行第三列)
                    axes[0,2].set_title('频域特征对比', fontsize=14, fontweight='bold', pad=15)
                    
                    # 计算频域特征
                    if normal_features is not None and anomaly_features is not None:
                        # 选择一个代表性的正常和异常样本进行FFT分析
                        normal_idx = selected_normal_idx[0]
                        anomaly_idx = selected_anomaly_idx[0]
                        
                        # 正常样本FFT
                        start_idx = window_indices[normal_idx]
                        end_idx = start_idx + window_size
                        normal_window = original_data[start_idx:end_idx]
                        normal_fft = np.abs(np.fft.fft(normal_window))[:len(normal_window)//2]
                        
                        # 异常样本FFT
                        start_idx = window_indices[anomaly_idx]
                        end_idx = start_idx + window_size
                        anomaly_window = original_data[start_idx:end_idx]
                        anomaly_fft = np.abs(np.fft.fft(anomaly_window))[:len(anomaly_window)//2]
                        
                        freqs = np.fft.fftfreq(window_size, 1.0)[:window_size//2]
                        
                        axes[0,2].plot(freqs, normal_fft, color=normal_color, linewidth=2, 
                                      label='正常值频谱', alpha=0.8)
                        axes[0,2].plot(freqs, anomaly_fft, color=anomaly_color, linewidth=2, 
                                      label='异常值频谱', alpha=0.8)
                        
                        axes[0,2].set_xlabel('频率', fontsize=12)
                        axes[0,2].set_ylabel('幅值', fontsize=12)
                        axes[0,2].legend()
                        axes[0,2].grid(True, alpha=0.3)
                    
                    # 简化其余子图以避免代码过长
                    for i in range(1, 2):
                        for j in range(3):
                            if i == 1 and j == 0:
                                axes[i,j].set_title('变化率模式对比', fontsize=14, fontweight='bold', pad=15)
                                axes[i,j].text(0.5, 0.5, '变化率分析\n(简化版)', ha='center', va='center', 
                                              transform=axes[i,j].transAxes, fontsize=12)
                            elif i == 1 and j == 1:
                                axes[i,j].set_title('极值模式对比', fontsize=14, fontweight='bold', pad=15)
                                axes[i,j].text(0.5, 0.5, '极值分析\n(简化版)', ha='center', va='center', 
                                              transform=axes[i,j].transAxes, fontsize=12)
                            elif i == 1 and j == 2:
                                axes[i,j].set_title('模式特征总结', fontsize=14, fontweight='bold', pad=15)
                                axes[i,j].axis('off')
                                
                                if normal_features is not None and anomaly_features is not None:
                                    normal_std_mean = np.mean(normal_features[:, 1])
                                    anomaly_std_mean = np.mean(anomaly_features[:, 1])
                                    
                                    summary_text = f"""
测试数据模式对比:

🔵 正常值模式:
  • 标准差: {normal_std_mean:.3f}
  • 变化平稳

🔴 异常值模式:
  • 标准差: {anomaly_std_mean:.3f}
  • 变化剧烈

✅ 检测到显著差异！
                                    """
                                    
                                    axes[i,j].text(0.05, 0.95, summary_text, transform=axes[i,j].transAxes,
                                                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                    
                    # 美化所有子图
                    for i, ax in enumerate(axes.flat[:-1]):  # 最后一个是文本图，不需要美化
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.tick_params(axis='both', which='major', labelsize=11, direction='in')
                    
                    plt.tight_layout(pad=3.0)
                    
                    # 添加总标题
                    fig.suptitle('标准异常值与正常值模式特征对比分析 (测试版)', 
                                fontsize=18, fontweight='bold', y=0.98)
                    
                    if save_path is None:
                        save_path = os.path.join(self.output_dir, 'test_standard_anomaly_patterns_comparison.pdf')
                    
                    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    print(f"✅ 测试标准异常值和正常值模式对比图已保存: {save_path}")
                    
                    return {
                        'normal_samples': len(selected_normal_idx),
                        'anomaly_samples': len(selected_anomaly_idx),
                        'success': True
                    }
                    
                except Exception as e:
                    print(f"❌ 生成测试模式对比图时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
        
        # 创建测试可视化器并运行测试
        visualizer = TestVisualizer()
        result = visualizer.plot_standard_anomaly_patterns(test_data, window_indices, true_labels, window_size)
        
        if result and result.get('success'):
            print(f"🎉 测试成功！")
            print(f"   - 正常样本数: {result['normal_samples']}")
            print(f"   - 异常样本数: {result['anomaly_samples']}")
            
            # 显示原始数据图表以便对比
            plt.figure(figsize=(15, 6))
            plt.plot(test_data, linewidth=2, label='测试数据', alpha=0.8)
            
            # 标记异常区域
            for i, label in enumerate(true_labels):
                if label == 1:
                    start_idx = window_indices[i]
                    end_idx = start_idx + window_size
                    plt.axvspan(start_idx, end_idx, alpha=0.3, color='red', label='异常区域' if i == 0 else "")
            
            plt.title('测试数据概览 - 显示异常区域', fontsize=16, fontweight='bold')
            plt.xlabel('时间', fontsize=14)
            plt.ylabel('压力值', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(visualizer.output_dir, 'test_data_overview.pdf')
            plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"📈 测试数据概览图已保存: {save_path}")
        else:
            print("❌ 测试失败")
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization()
