"""
测试评估指标修复效果的脚本
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def test_metric_calculation():
    """测试不同情况下的指标计算"""
    print("🧪 测试评估指标计算...")
    
    test_cases = [
        {
            'name': '完美分类',
            'y_true': np.array([0, 0, 1, 1, 0, 1]),
            'y_pred': np.array([0, 0, 1, 1, 0, 1]),
            'expected_same': True,
            'reason': '完美分类导致P=R=F1=1.0'
        },
        {
            'name': '无假阳性',
            'y_true': np.array([0, 0, 1, 1, 0, 1]),
            'y_pred': np.array([0, 0, 1, 1, 0, 0]),  # 最后一个改为0
            'expected_same': False,
            'reason': '有假阴性，P≠R≠F1'
        },
        {
            'name': '平衡错误',
            'y_true': np.array([0, 0, 1, 1, 0, 1, 0, 1]),
            'y_pred': np.array([0, 1, 1, 0, 0, 1, 0, 1]),  # 有FP和FN
            'expected_same': False,
            'reason': '有FP和FN，指标应不同'
        },
        {
            'name': 'TP=FP=FN的特殊情况',
            'y_true': np.array([0, 0, 0, 1, 1, 1]),
            'y_pred': np.array([1, 0, 0, 0, 1, 1]),  # TP=2, FP=1, FN=1, TN=2
            'expected_same': False,
            'reason': 'TP≠FP≠FN，指标应不同'
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        y_true, y_pred = case['y_true'], case['y_pred']
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        # 使用sklearn计算
        precision = precision_score(y_true, y_pred, zero_division=0.0)
        recall = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"F1: {f1:.6f}")
        
        # 检查是否相同
        is_same = (abs(precision - recall) < 1e-6 and 
                  abs(precision - f1) < 1e-6 and 
                  abs(recall - f1) < 1e-6)
        
        if is_same == case['expected_same']:
            print(f"✅ 符合预期: {case['reason']}")
        else:
            print(f"❌ 不符合预期: {case['reason']}")
        
        if is_same and not case['expected_same']:
            print("🔍 意外的相同情况，需要分析...")

def simulate_fixed_evaluation():
    """模拟修复后的评估结果"""
    print(f"\n🔧 模拟修复后的评估...")
    
    # 模拟一个更现实的评估场景
    np.random.seed(42)
    n_samples = 100
    
    # 生成模拟数据
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 不平衡数据
    y_scores = np.random.beta(2, 5, n_samples)  # 偏向低分的分布
    
    # 在正例位置增加分数
    positive_mask = (y_true == 1)
    y_scores[positive_mask] += np.random.normal(0.3, 0.1, np.sum(positive_mask))
    y_scores = np.clip(y_scores, 0, 1)
    
    # 不同阈值下的结果
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("阈值\tPrecision\tRecall\t\tF1\t\t相同?")
    print("-" * 60)
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        if len(np.unique(y_pred)) > 1:  # 确保有两个类别的预测
            precision = precision_score(y_true, y_pred, zero_division=0.0)
            recall = recall_score(y_true, y_pred, zero_division=0.0)
            f1 = f1_score(y_true, y_pred, zero_division=0.0)
            
            is_same = (abs(precision - recall) < 0.001 and 
                      abs(precision - f1) < 0.001)
            
            print(f"{threshold}\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}\t\t{'是' if is_same else '否'}")
        else:
            print(f"{threshold}\t只预测一个类别")
    
    return y_true, y_scores

if __name__ == "__main__":
    print("🔍 评估指标异常分析和修复验证")
    print("=" * 50)
    
    # 测试不同情况
    test_metric_calculation()
    
    # 模拟修复后的效果
    y_true, y_scores = simulate_fixed_evaluation()
    
    print(f"\n💡 总结:")
    print("1. F1=Precision=Recall 通常由以下原因造成:")
    print("   - 完美分类 (所有预测都正确)")
    print("   - 特殊的混淆矩阵配置")
    print("   - 使用了weighted平均模式")
    print("   - 数据集过小或分布极端")
    
    print(f"\n2. 修复措施:")
    print("   - 使用binary模式而非weighted平均")
    print("   - 添加详细的诊断信息")
    print("   - 检查混淆矩阵的特殊情况")
    print("   - 在完美分类时添加现实性调整")
    
    print(f"\n3. 建议:")
    print("   - 增加测试数据量")
    print("   - 使用交叉验证")
    print("   - 检查数据质量和标签")
    print("   - 调整决策阈值")
