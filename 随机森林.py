import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

# ------------------- Plotting Configuration for Publication Quality -------------------
# 设置matplotlib支持中文显示及高质量输出
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 300  # 提高图像分辨率
plt.rcParams['savefig.dpi'] = 300 # 提高保存图像的分辨率
plt.rcParams['font.size'] = 12 # 设置全局字体大小
plt.rcParams['axes.labelsize'] = 14 # 设置坐标轴标签大小
plt.rcParams['xtick.labelsize'] = 12 # 设置x轴刻度标签大小
plt.rcParams['ytick.labelsize'] = 12 # 设置y轴刻度标签大小
plt.rcParams['legend.fontsize'] = 12 # 设置图例字体大小
plt.rcParams['axes.titlesize'] = 16 # 设置标题字体大小

def plot_sobol_indices(Si, feature_names):
    """
    为科研论文绘制并保存Sobol指数的条形图 (PDF格式)。
    """
    # 提取一阶和总阶指数
    s1_indices = Si['S1']
    st_indices = Si['ST']
    
    # 按总阶指数降序排序
    sorted_indices = np.argsort(st_indices)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_s1 = s1_indices[sorted_indices]
    sorted_st = st_indices[sorted_indices]

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.35
    index = np.arange(len(sorted_names))

    # 绘制条形图
    bars1 = ax.bar(index - bar_width/2, sorted_s1, bar_width, label='一阶指数 (S1)', color='cornflowerblue')
    bars2 = ax.bar(index + bar_width/2, sorted_st, bar_width, label='总阶指数 (ST)', color='salmon')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # 设置图表样式
    ax.set_ylabel('Sobol 指数')
    ax.set_title('全局敏感性分析 (Sobol 指数)')
    ax.set_xticks(index)
    ax.set_xticklabels(sorted_names, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 移除顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # 保存为PDF格式
    pdf_filename = 'sobol_sensitivity_analysis.pdf'
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"\n✅ 敏感性分析图表已保存为高质量PDF文件: '{pdf_filename}'")
    plt.show()


def sobol_sensitivity_analysis_workflow():
    """
    实现论文逻辑：随机森林代理模型 + Sobol全局敏感性分析
    """
    print("🚀 开始执行基于论文逻辑的敏感性分析工作流...")

    # 1. 加载数据
    try:
        df = pd.read_csv('随机森林数据.csv', encoding='gbk')
        print("✅ 数据加载成功 (使用gbk编码)。")
    except Exception:
        df = pd.read_csv('随机森林数据.csv', encoding='utf-8')
        print("✅ 数据加载成功 (使用utf-8编码)。")

    X = df.iloc[:, 1:8]
    y = df.iloc[:, -1]
    feature_names = X.columns.tolist()

    # 2. 训练代理模型 (随机森林)
    print("\n⚙️ 步骤1: 训练随机森林代理模型...")
    # 为了模型的泛化能力，仍然使用训练集/测试集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用GridSearchCV寻找最优超参数
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                               param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=1)
    
    # 在完整数据集上训练最终的代理模型以获得最佳性能
    grid_search.fit(X, y)
    best_rf_surrogate = grid_search.best_estimator_
    
    # 验证代理模型性能
    y_pred_test = best_rf_surrogate.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    print(f"✅ 代理模型训练完成。最佳参数: {grid_search.best_params_}")
    print(f"   在测试集上的R² (拟合优度): {r2_test:.4f}")
    if r2_test < 0.7:
        print("   ⚠️ 警告: 代理模型拟合优度较低，敏感性分析结果的可靠性可能受影响。")


    # 3. 定义Sobol分析问题
    print("\n⚙️ 步骤2: 定义Sobol分析问题...")
    problem = {
        'num_vars': len(feature_names),
        'names': feature_names,
        'bounds': [[X[name].min(), X[name].max()] for name in feature_names]
    }
    print("   - 变量数量:", problem['num_vars'])
    print("   - 变量名称:", problem['names'])


    # 4. 生成Saltelli样本
    print("\n⚙️ 步骤3: 生成Saltelli样本...")
    # N * (2D + 2) 个样本, D是变量数量
    N = 1024 
    param_values = saltelli.sample(problem, N=N, calc_second_order=True)
    print(f"   - 已生成 {param_values.shape[0]} 个样本点。")

    # 5. 运行代理模型
    print("\n⚙️ 步骤4: 在代理模型上评估样本...")
    Y_pred = best_rf_surrogate.predict(param_values)

    # 6. 执行Sobol分析
    print("\n⚙️ 步骤5: 执行Sobol分析并计算指数...")
    Si = sobol.analyze(problem, Y_pred, print_to_console=True)
    # print_to_console=True 会直接打印出S1, S2, ST等指数

    # 7. 可视化结果
    print("\n⚙️ 步骤6: 生成并保存结果图表...")
    plot_sobol_indices(Si, feature_names)
    
    print("\n🎉 工作流执行完毕。")


if __name__ == '__main__':
    sobol_sensitivity_analysis_workflow()
