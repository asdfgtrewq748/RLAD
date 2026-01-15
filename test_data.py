import pandas as pd
import numpy as np

print("开始数据检测...")

try:
    df = pd.read_csv('data1.csv')
    print(f"✅ 数据读取成功")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 检查数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"数值列: {numeric_cols}")
    
    # 检查特定列是否存在
    target_columns = ['103#', '102#', '1#', '2#', '3#']
    found_cols = [col for col in target_columns if col in df.columns]
    print(f"找到的目标列: {found_cols}")
    
    if found_cols:
        selected_col = found_cols[0]
        print(f"将使用列: {selected_col}")
        
        # 检查数据质量
        data = df[selected_col].values
        print(f"选择列的统计信息:")
        print(f"  长度: {len(data)}")
        print(f"  均值: {np.mean(data):.2f}")
        print(f"  标准差: {np.std(data):.2f}")
        print(f"  最小值: {np.min(data):.2f}")
        print(f"  最大值: {np.max(data):.2f}")
        print(f"  缺失值: {np.sum(pd.isna(data))}")
        
    else:
        print("未找到目标列，将使用第一个数值列")
        if numeric_cols:
            selected_col = numeric_cols[0]
            print(f"使用列: {selected_col}")
        else:
            print("❌ 没有找到数值列")
            
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("数据检测完成")
