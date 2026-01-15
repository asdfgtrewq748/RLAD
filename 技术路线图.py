# -*- coding: utf-8 -*-
# 导入 graphviz 库，用于创建流程图
# 如果您尚未安装，请使用命令: pip install graphviz
# 此外，请确保您的系统中已经安装了 Graphviz 的图形渲染引擎
# (可以从官网 https://graphviz.org/download/ 下载)
from graphviz import Digraph

# --- 1. 初始化图形对象 ---
dot = Digraph(
    comment='STL-LOF-RLAD 美化版技术路线图',
    graph_attr={
        'rankdir': 'TB',          # 布局方向：从上到下
        'splines': 'ortho',       # 连线样式：直角正交
        'nodesep': '0.8',         # 节点间距
        'ranksep': '1.2',         # 层级间距
        'fontname': 'SimHei',     # 全局字体
        'fontsize': '20',         # 全局字体大小
        'bgcolor': '#F7F7F7'      # 背景色
    }
)

# --- 2. 定义节点和边的通用样式 ---
dot.attr('node', style='rounded,filled,drop-shadow', shape='box', fontname='SimHei', fontsize='12', margin='0.4,0.3')
dot.attr('edge', fontname='SimHei', fontsize='10')

# --- 3. 定义流程节点 (内容已丰富) ---

# (a) 数据准备与预处理
dot.node('A',
         '(a) 数据准备与预处理\nData Preparation & Preprocessing\n\n'
         '▸ 数据源: 钱家营煤矿液压支架数据\l'
         '▸ 标准化: Z-score 归一化\l'
         '▸ 样本构建: 288步长滑动窗口\l',
         fillcolor='#E3F2FD', color='#90CAF9')

# (b) 无监督信号生成
dot.node('B',
         '(b) 无监督信号生成\nUnsupervised Signal Generation\n\n'
         '▸ 分解: STL 去除宏观模式 (趋势/季节性)\l'
         '▸ 检测: LOF 识别残差序列中的局部密度异常\l'
         '▸ 输出: 生成高质量初始伪标签\l',
         fillcolor='#E8F5E9', color='#A5D6A7')

# (c) 模型训练与精化 (使用子图分组)
with dot.subgraph(name='cluster_C') as c:
    c.attr(label='(c) 风险感知模型训练与精化\nRisk-Aware Model Training & Refinement',
           style='rounded,filled', color='#B3E5FC', bgcolor='#FFFFFF', fontname='SimHei', fontsize='14', penwidth='2')
    c.node('C1',
           '▸ 智能体架构: DQN + Bi-LSTM + Attention\l'
           '▸ 训练策略: 优先经验回放 (PER)\l'
           '▸ 优化目标: 非对称奖励函数 (高惩罚漏报)\l',
           fillcolor='#FFFDE7', color='#FFF59D')
    c.node('C2',
           '▸ 查询策略: 边际采样 (Margin Sampling)\l'
           '▸ 人机协同: 提交不确定样本由专家标注\l',
           fillcolor='#FFFDE7', color='#FFF59D')
    # 训练与主动学习的反馈循环
    c.edge('C1', 'C2', style='dashed', label='  查询')
    c.edge('C2', 'C1', style='dashed', label='反馈优化  ')

# (决策) 模型性能评估
dot.node('Decision',
         '模型性能评估\nModel Performance Evaluation\n\n'
         '(F1-Score, AUC, Recall)',
         shape='diamond', fillcolor='#FFEBEE', color='#EF9A9A')

# (d) 自动化检测与应用
dot.node('D',
         '(d) 自动化检测与应用\nAutomated Detection & Application\n\n'
         '▸ 部署训练完成的智能体策略\l'
         '▸ 实现自动化、无阈值的异常检测\l'
         '▸ 应用于预测性维护 (PdM)，提升系统安全性\l',
         fillcolor='#E1F5FE', color='#81D4FA')

# (最终分析 - 使用不可见的结构节点进行对齐)
with dot.subgraph(name='cluster_E') as c:
    c.attr(style='invis') # 隐藏子图边框
    c.node('E1', '对比分析\n(与基线模型性能对比)', shape='oval', fillcolor='#F3E5F5', color='#CE93D8')
    c.node('E2', '消融研究\n(验证各模块贡献度)', shape='oval', fillcolor='#F3E5F5', color='#CE93D8')
    c.node('E3', '案例研究\n(可视化异常检测效果)', shape='oval', fillcolor='#F3E5F5', color='#CE93D8')
    # 保持分析节点在同一行
    dot.edge('E1', 'E2', style='invis')
    dot.edge('E2', 'E3', style='invis')
    c.attr(rank='same')


# --- 4. 连接整体流程 ---
dot.edge('A', 'B')
dot.edge('B', 'C1')
dot.edge('C1', 'Decision', lhead='cluster_C') # 从C1出发，但视觉上从C集群引出

# 决策分支
dot.edge('Decision', 'D', label='  合适 (Suitable)')
dot.edge('Decision', 'B', label='不合适 (Unsuitable)\n(调整伪标签或模型参数)  ', style='dashed')

# 部署后的分析流程
dot.edge('D', 'E2', ltail='D', lhead='E2') # 从D的中心连接到E2


# --- 5. 渲染并保存图形 ---
output_filename = 'stl_lof_rlad_roadmap_enhanced'
try:
    dot.render(output_filename, view=False, cleanup=True)
    print(f"美化版流程图已成功生成: {output_filename}.png")
except Exception as e:
    print(f"生成失败，请确保已正确安装 Graphviz 并且配置了系统路径。错误信息: {e}")

