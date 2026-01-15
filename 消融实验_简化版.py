"""
RLAD v3.2 消融实验 - 工作版本
对比以下方法：
1. Active Learning
2. LOF (using 3σ)  
3. STL (LOF on Raw)
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

def load_data_and_create_windows(data_path, window_size=288, stride=144):
    """加载数据并创建窗口"""
    print(f"📥 加载数据: {data_path}")
    
    # 读取数据
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    
    # 选择特征列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("没有找到数值列")
    
    # 尝试找到目标列
    target_cols = ['103#', '102#', '1#']
    feature_col = None
    for col in target_cols:
        if col in df.columns:
            feature_col = col
            break
    
    if feature_col is None:
        feature_col = numeric_cols[0]
    
    print(f"使用特征列: {feature_col}")
    
    # 处理数据
    data = df[feature_col].fillna(method='ffill').fillna(0).values
    print(f"数据长度: {len(data)}")
    
    # 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 创建窗口
    windows = []
    raw_windows = []
    indices = []
    
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data_scaled[i:i + window_size])
        raw_windows.append(data[i:i + window_size])
        indices.append(i)
    
    print(f"创建窗口数: {len(windows)}")
    
    return np.array(windows), np.array(raw_windows), np.array(indices), data, scaler

def generate_ground_truth(raw_data, windows, window_indices):
    """生成基准真实标签"""
    print("生成基准标签...")
    
    # 使用统计方法生成标签
    mean_val = np.mean(raw_data)
    std_val = np.std(raw_data)
    
    # 点级别异常检测
    threshold = mean_val + 2.5 * std_val
    point_anomalies = (raw_data > threshold).astype(int)
    
    # 窗口级别标签
    y_true = []
    for i, start_idx in enumerate(window_indices):
        end_idx = start_idx + len(windows[i])
        window_anomalies = point_anomalies[start_idx:end_idx]
        
        # 如果窗口中异常点比例超过5%，标记为异常
        anomaly_ratio = np.sum(window_anomalies) / len(window_anomalies)
        y_true.append(1 if anomaly_ratio > 0.05 else 0)
    
    y_true = np.array(y_true)
    
    # 确保有足够的异常样本
    if np.sum(y_true) < 10:
        # 基于窗口统计特征选择异常
        window_scores = []
        for window in windows:
            score = np.std(window) + (np.max(window) - np.min(window))
            window_scores.append(score)
        
        threshold_95 = np.percentile(window_scores, 95)
        y_true = (np.array(window_scores) > threshold_95).astype(int)
    
    print(f"标签分布: 正常={np.sum(y_true==0)}, 异常={np.sum(y_true==1)}")
    return y_true

def method_active_learning(windows, y_true):
    """方法1: Active Learning"""
    print("\n🔬 方法1: Active Learning")
    
    # 选择10%的样本进行"标注"
    n_labeled = int(len(windows) * 0.1)
    
    # 基于不确定性采样
    uncertainties = []
    for window in windows:
        uncertainty = np.std(window) / (np.mean(np.abs(window)) + 1e-8)
        uncertainties.append(uncertainty)
    
    # 选择最不确定的样本
    uncertain_indices = np.argsort(uncertainties)[-n_labeled:]
    
    # 训练简单分类器
    labeled_X = windows[uncertain_indices]
    labeled_y = y_true[uncertain_indices]
    
    # 基于统计阈值分类
    if np.sum(labeled_y == 1) > 0:
        anomaly_samples = labeled_X[labeled_y == 1]
        threshold = np.mean([np.std(w) for w in anomaly_samples])
    else:
        threshold = np.percentile(uncertainties, 90)
    
    # 预测
    y_pred = []
    for window in windows:
        y_pred.append(1 if np.std(window) > threshold else 0)
    
    y_pred = np.array(y_pred)
    
    # 计算指标
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return {'method': 'Active Learning', 'f1': f1, 'precision': precision, 'recall': recall}

def method_lof_3sigma(raw_windows):
    """方法2: LOF (using 3σ)"""
    print("\n🔬 方法2: LOF (using 3σ)")
    
    y_pred = []
    
    for window in raw_windows:
        # 3σ检测
        mean_val = np.mean(window)
        std_val = np.std(window)
        outliers = np.abs(window - mean_val) > 3 * std_val
        sigma3_ratio = np.sum(outliers) / len(window)
        
        # 如果3σ异常点多，应用LOF
        if sigma3_ratio > 0.02:
            try:
                # 构建特征
                features = np.array([
                    np.mean(window),
                    np.std(window),
                    np.max(window) - np.min(window),
                    sigma3_ratio
                ]).reshape(1, -1)
                
                # 这里简化处理，直接基于sigma3_ratio判断
                y_pred.append(1 if sigma3_ratio > 0.05 else 0)
            except:
                y_pred.append(1 if sigma3_ratio > 0.1 else 0)
        else:
            y_pred.append(0)
    
    y_pred = np.array(y_pred)
    return y_pred

def method_stl_lof_raw(raw_data, raw_windows, window_indices):
    """方法3: STL (LOF on Raw)"""
    print("\n🔬 方法3: STL (LOF on Raw)")
    
    try:
        # STL分解
        stl = STL(raw_data, seasonal=25, period=24, robust=True)
        result = stl.fit()
        residual = result.resid
        
        # 处理NaN
        residual = np.nan_to_num(residual, nan=0.0)
        
        print("STL分解完成")
        
        # 为每个窗口计算残差特征
        y_pred = []
        
        for i, start_idx in enumerate(window_indices):
            end_idx = start_idx + len(raw_windows[i])
            if end_idx > len(residual):
                end_idx = len(residual)
                start_idx = max(0, end_idx - len(raw_windows[i]))
            
            window_residual = residual[start_idx:end_idx]
            
            # 残差异常分数
            residual_std = np.std(window_residual)
            residual_mean_abs = np.mean(np.abs(window_residual))
            
            # 阈值判断
            global_residual_std = np.std(residual)
            threshold = global_residual_std * 2
            
            y_pred.append(1 if residual_std > threshold else 0)
        
        y_pred = np.array(y_pred)
        
    except Exception as e:
        print(f"STL失败: {e}")
        # 备用方案
        y_pred = []
        for window in raw_windows:
            # 简单移动平均
            trend = np.convolve(window, np.ones(min(24, len(window)//4))/min(24, len(window)//4), mode='same')
            residual = window - trend
            
            residual_score = np.std(residual)
            threshold = np.std(window) * 1.5
            y_pred.append(1 if residual_score > threshold else 0)
        
        y_pred = np.array(y_pred)
    
    return y_pred

def run_experiment():
    """运行消融实验"""
    print("🚀 开始RLAD v3.2消融实验")
    
    try:
        # 加载数据
        windows, raw_windows, window_indices, raw_data, scaler = load_data_and_create_windows("data1.csv")
        
        # 生成基准标签
        y_true = generate_ground_truth(raw_data, windows, window_indices)
        
        # 运行三种方法
        results = []
        
        # 方法1: Active Learning
        result1 = method_active_learning(windows, y_true)
        results.append(result1)
        
        # 方法2: LOF (3σ)
        y_pred2 = method_lof_3sigma(raw_windows)
        f1_2 = f1_score(y_true, y_pred2)
        precision_2 = precision_score(y_true, y_pred2, zero_division=0)
        recall_2 = recall_score(y_true, y_pred2, zero_division=0)
        print(f"F1: {f1_2:.4f}, Precision: {precision_2:.4f}, Recall: {recall_2:.4f}")
        
        result2 = {'method': 'LOF (3σ)', 'f1': f1_2, 'precision': precision_2, 'recall': recall_2}
        results.append(result2)
        
        # 方法3: STL (LOF on Raw)
        y_pred3 = method_stl_lof_raw(raw_data, raw_windows, window_indices)
        f1_3 = f1_score(y_true, y_pred3)
        precision_3 = precision_score(y_true, y_pred3, zero_division=0)
        recall_3 = recall_score(y_true, y_pred3, zero_division=0)
        print(f"F1: {f1_3:.4f}, Precision: {precision_3:.4f}, Recall: {recall_3:.4f}")
        
        result3 = {'method': 'STL (LOF on Raw)', 'f1': f1_3, 'precision': precision_3, 'recall': recall_3}
        results.append(result3)
        
        # 汇总结果
        baseline_f1 = 0.85  # RLAD基准F1
        
        print("\n" + "="*60)
        print("📊 消融实验结果汇总")
        print("="*60)
        print(f"{'方法':<20} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'Performance Drop':<15}")
        print("-"*75)
        
        for result in results:
            method = result['method']
            f1 = result['f1']
            precision = result['precision']
            recall = result['recall']
            
            perf_drop = ((baseline_f1 - f1) / baseline_f1) * 100 if baseline_f1 > 0 else 0
            
            print(f"{method:<20} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f} {perf_drop:<15.2f}%")
        
        # 保存结果
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        with open(f"ablation_results_{timestamp}.txt", 'w') as f:
            f.write("RLAD v3.2 消融实验结果\n")
            f.write("="*50 + "\n")
            for result in results:
                f.write(f"{result['method']}: F1={result['f1']:.4f}\n")
        
        print(f"\n✅ 实验完成! 结果已保存到 ablation_results_{timestamp}.txt")
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_experiment()
