"""
简化的消融实验测试
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

def main():
    print("🚀 开始简化消融实验")
    
    # 读取数据
    try:
        df = pd.read_csv('data1.csv')
        print(f"✅ 数据读取成功: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("❌ 没有数值列")
            return
            
        # 使用第一个数值列
        feature_col = numeric_cols[0]
        print(f"使用特征列: {feature_col}")
        
        # 获取数据
        data = df[feature_col].fillna(method='ffill').fillna(0).values
        print(f"数据长度: {len(data)}")
        print(f"数据范围: [{np.min(data):.2f}, {np.max(data):.2f}]")
        
        # 生成基准标签（使用简单的阈值方法）
        threshold = np.mean(data) + 2 * np.std(data)
        y_true = (data > threshold).astype(int)
        print(f"异常样本数: {np.sum(y_true)} / {len(y_true)}")
        
        if np.sum(y_true) == 0:
            # 如果没有异常，人工创建一些
            indices = np.random.choice(len(data), size=len(data)//20, replace=False)
            y_true[indices] = 1
            print(f"人工创建异常样本数: {np.sum(y_true)}")
        
        # 方法1: 简单LOF
        print("\n🔬 方法1: 简单LOF")
        try:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
            lof_pred = lof.fit_predict(data.reshape(-1, 1))
            y_pred1 = (lof_pred == -1).astype(int)
            
            f1_1 = f1_score(y_true, y_pred1)
            precision_1 = precision_score(y_true, y_pred1, zero_division=0)
            recall_1 = recall_score(y_true, y_pred1, zero_division=0)
            
            print(f"F1: {f1_1:.4f}, Precision: {precision_1:.4f}, Recall: {recall_1:.4f}")
            
        except Exception as e:
            print(f"方法1失败: {e}")
            f1_1 = 0
        
        # 方法2: 3σ规则
        print("\n🔬 方法2: 3σ规则")
        try:
            mean_val = np.mean(data)
            std_val = np.std(data)
            y_pred2 = (np.abs(data - mean_val) > 3 * std_val).astype(int)
            
            f1_2 = f1_score(y_true, y_pred2)
            precision_2 = precision_score(y_true, y_pred2, zero_division=0)
            recall_2 = recall_score(y_true, y_pred2, zero_division=0)
            
            print(f"F1: {f1_2:.4f}, Precision: {precision_2:.4f}, Recall: {recall_2:.4f}")
            
        except Exception as e:
            print(f"方法2失败: {e}")
            f1_2 = 0
        
        # 方法3: 统计方法
        print("\n🔬 方法3: 统计方法")
        try:
            # 使用IQR方法
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            y_pred3 = ((data < lower_bound) | (data > upper_bound)).astype(int)
            
            f1_3 = f1_score(y_true, y_pred3)
            precision_3 = precision_score(y_true, y_pred3, zero_division=0)
            recall_3 = recall_score(y_true, y_pred3, zero_division=0)
            
            print(f"F1: {f1_3:.4f}, Precision: {precision_3:.4f}, Recall: {recall_3:.4f}")
            
        except Exception as e:
            print(f"方法3失败: {e}")
            f1_3 = 0
        
        # 汇总结果
        print("\n📊 结果汇总:")
        print(f"{'方法':<15} {'F1-Score':<10}")
        print("-" * 25)
        print(f"{'简单LOF':<15} {f1_1:<10.4f}")
        print(f"{'3σ规则':<15} {f1_2:<10.4f}")
        print(f"{'统计方法':<15} {f1_3:<10.4f}")
        
        print("\n✅ 简化实验完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
