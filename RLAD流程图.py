# RLAD v3.2 Model Flowchart Generation Script
# This script uses the Graphviz library to create a flowchart visualizing the model's logic.
#
# To run this script, you need to install Graphviz:
# 1. Install the Graphviz software: https://graphviz.org/download/
#    (Make sure to add it to your system's PATH during installation on Windows)
# 2. Install the Python library: pip install graphviz

import graphviz
from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='RLAD v3.2 Model Flowchart')
dot.attr('graph', 
         rankdir='TB',  # Top to Bottom layout
         splines='ortho', # Use orthogonal lines for a cleaner look
         nodesep='0.8', # Separation between nodes
         ranksep='1.2', # Separation between ranks (layers)
         fontname='Helvetica', # Use a common, clean font
         label='基于STL+LOF与强化学习的液压支架工作阻力异常检测模型 (RLAD v3.2) 流程图',
         fontsize='22',
         labelloc='t'
        )

# Define global styles for nodes and edges
dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='12')
dot.attr('edge', fontname='Helvetica', fontsize='10')

# === Color Palette (inspired by the reference image) ===
color_A = '#E8EAF6' # Light Purple/Blue for Preprocessing
color_B = '#E0F2F1' # Light Green for Interactive Loop
color_C = '#E3F2FD' # Light Blue for Model Architecture
color_D = '#FFF9C4' # Light Yellow for Evaluation
color_data = '#FFCDD2' # Light Red for Data
color_human = '#D1C4E9' # Light Purple for Human Interaction
color_arrow = '#424242'

# =============================================================================
# (A) Data Preprocessing & Pseudo-Labeling
# =============================================================================
with dot.subgraph(name='cluster_A') as c:
    c.attr(label='(A) 数据预处理与伪标签生成', style='filled', color=color_A, penwidth='2', fontsize='16')
    
    # Nodes
    c.node('raw_data', '原始工作阻力数据\n(CSV 文件)', shape='cylinder', fillcolor=color_data)
    c.node('data_loader', '数据加载与特征选择\n(load_hydraulic_data_with_stl_lof)', fillcolor='white')
    # CORRECTED: Removed the second positional argument 'STL+LOF 异常点检测'
    c.node('stl_lof', shape='Mrecord', fillcolor='white', 
           label='{STL+LOF 异常点检测 | {<f0> STL 分解 (趋势, 季节, 残差) | <f1> LOF + 统计阈值 + 趋势变化 | <f2> 生成逐点异常标签}}')
    c.node('windowing', '滑动窗口切分\n(窗口大小, 步长)', fillcolor='white')
    c.node('window_label', '窗口级伪标签生成\n(基于窗口内异常点比例与密度)', fillcolor='white')
    c.node('split', '数据集划分\n(训练集, 验证集, 测试集)\n(包含未标注样本)', shape='diamond', fillcolor='white')

    # Edges
    c.edge('raw_data', 'data_loader', label='输入', color=color_arrow)
    c.edge('data_loader', 'stl_lof', label='时序数据', color=color_arrow)
    c.edge('stl_lof:f2', 'windowing', label='逐点标签', color=color_arrow)
    c.edge('data_loader', 'windowing', label='标准化数据', color=color_arrow)
    c.edge('windowing', 'window_label', color=color_arrow)
    c.edge('window_label', 'split', color=color_arrow)

# =============================================================================
# (C) Enhanced RLAD Agent Architecture (Defined early for layout)
# =============================================================================
with dot.subgraph(name='cluster_C') as c:
    c.attr(label='(C) 增强型RLAD智能体架构 (EnhancedRLADAgent)', style='filled', color=color_C, penwidth='2', fontsize='16')
    
    # Nodes
    c.node('input_window', '输入: 时间序列窗口', shape='parallelogram', fillcolor='white')
    # CORRECTED: Removed the second positional argument
    c.node('cnn', shape='Mrecord', fillcolor='white',
           label='{1D CNN 特征提取器 | (局部模式捕捉) | Conv1D → ReLU → BN → Dropout | Conv1D → ReLU → BN | ...}')
    # CORRECTED: Removed the second positional argument
    c.node('lstm', shape='Mrecord', fillcolor='white',
           label='{双向 LSTM | (时序依赖建模) | <f0> 前向 | <f1> 后向}')
    c.node('attention', '多头自注意力机制\n(关键时间步聚焦)', fillcolor='white')
    c.node('add_norm', '残差连接 & 层归一化', shape='invhouse', fillcolor='white')
    c.node('pooling', '全局池化\n(最大池化 + 平均池化)', fillcolor='white')
    # CORRECTED: Removed the second positional argument
    c.node('classifier', shape='Mrecord', fillcolor='white',
           label='{分类器 (MLP) | (输出Q值) | Linear → ReLU → Dropout | Linear → Q-Values}')
    c.node('output_q', '输出: Q(s, a)\n(正常/异常)', shape='parallelogram', fillcolor='white')

    # Edges
    c.edge('input_window', 'cnn', color=color_arrow)
    c.edge('cnn', 'lstm', color=color_arrow)
    c.edge('lstm', 'attention', label='序列特征', color=color_arrow)
    c.edge('attention', 'add_norm', color=color_arrow)
    c.edge('lstm', 'add_norm', color=color_arrow) # Residual connection
    c.edge('add_norm', 'pooling', color=color_arrow)
    c.edge('pooling', 'classifier', label='特征向量', color=color_arrow)
    c.edge('classifier', 'output_q', color=color_arrow)

# =============================================================================
# (B) Interactive Reinforcement Learning Loop
# =============================================================================
with dot.subgraph(name='cluster_B') as c:
    c.attr(label='(B) 交互式强化学习训练环', style='filled', color=color_B, penwidth='2', fontsize='16')
    
    # Nodes
    c.node('start_loop', '开始训练循环', shape='ellipse', fillcolor='white')
    c.node('active_learning', '主动学习: 查询不确定样本', shape='Mdiamond', fillcolor='white')
    c.node('human_annotation', '人类专家交互式标注\n(AnnotationGUI)', shape='cds', fillcolor=color_human)
    c.node('update_labels', '更新训练集标签', fillcolor='white')
    c.node('select_action', 'RL智能体选择动作\n(ε-greedy策略)', fillcolor='white')
    c.node('compute_reward', '计算奖励\n(enhanced_compute_reward)', fillcolor='white')
    c.node('store_experience', '存入优先经验回放池\n(PrioritizedReplayBuffer)', shape='cylinder', fillcolor='white')
    c.node('sample_experience', '从回放池采样经验', fillcolor='white')
    c.node('train_step', '训练智能体 (DQN)\n(enhanced_train_dqn_step)', fillcolor='white')
    c.node('update_target', '更新目标网络', fillcolor='white')
    c.node('end_loop', '结束循环\n(达到最大轮次或早停)', shape='ellipse', fillcolor='white')

    # Edges for the loop
    c.edge('start_loop', 'active_learning', label='每 N 轮', color=color_arrow)
    c.edge('active_learning', 'human_annotation', label='请求标注', color=color_arrow)
    c.edge('human_annotation', 'update_labels', label='提供标签', color=color_arrow)
    c.edge('update_labels', 'select_action', style='dashed', color=color_arrow)
    c.edge('active_learning', 'select_action', label='跳过标注', style='dashed', color=color_arrow)
    
    c.edge('select_action', 'compute_reward', label='动作', color=color_arrow)
    c.edge('compute_reward', 'store_experience', label='奖励', color=color_arrow)
    c.edge('store_experience', 'sample_experience', label='Push', color=color_arrow)
    c.edge('sample_experience', 'train_step', label='Sample (s,a,r,s\')', color=color_arrow)
    c.edge('train_step', 'update_target', label='每 T 步', color=color_arrow)
    
    # Feedback loop
    c.edge('train_step', 'active_learning', label='更新模型权重', style='dashed', arrowhead='icurve', color=color_arrow)
    c.edge('update_target', 'end_loop', color=color_arrow)

# =============================================================================
# (D) Model Evaluation & Output
# =============================================================================
with dot.subgraph(name='cluster_D') as c:
    c.attr(label='(D) 模型评估与输出', style='filled', color=color_D, penwidth='2', fontsize='16')
    
    # Nodes
    c.node('load_best', '加载最佳模型', shape='folder', fillcolor='white')
    c.node('find_threshold', '在验证集上寻找最优阈值', fillcolor='white')
    c.node('evaluate', '在测试集上评估\n(enhanced_evaluate_model)', fillcolor='white')
    c.node('metrics', '性能指标\n(F1, Precision, Recall, AUC)', shape='note', fillcolor='white')
    c.node('visualize', '生成可视化图表\n(混淆矩阵, ROC, t-SNE等)', shape='image', fillcolor='white')
    c.node('pointwise', '逐点异常标记\n(mark_anomalies_pointwise)', fillcolor='white')
    c.node('final_output', '最终输出\n(CSV预测文件, 图表, 指标)', shape='document', fillcolor='white')

    # Edges
    c.edge('load_best', 'find_threshold', color=color_arrow)
    c.edge('find_threshold', 'evaluate', label='最优阈值', color=color_arrow)
    c.edge('evaluate', 'metrics', color=color_arrow)
    c.edge('evaluate', 'visualize', color=color_arrow)
    c.edge('evaluate', 'pointwise', label='窗口预测', color=color_arrow)
    c.edge('pointwise', 'final_output', color=color_arrow)
    c.edge('metrics', 'final_output', color=color_arrow)
    c.edge('visualize', 'final_output', color=color_arrow)


# =============================================================================
# Connecting the main clusters
# =============================================================================
dot.edge('split', 'start_loop', label='训练数据', color=color_arrow, lhead='cluster_B')
dot.edge('split', 'find_threshold', label='验证数据', color=color_arrow)
dot.edge('split', 'evaluate', label='测试数据', color=color_arrow)

# Connect the agent to the training loop
dot.edge('select_action', 'input_window', style='dashed', label='状态(s)', color=color_arrow, lhead='cluster_C')
dot.edge('output_q', 'select_action', style='dashed', label='Q值', color=color_arrow, ltail='cluster_C')

dot.edge('end_loop', 'load_best', label='训练完成', color=color_arrow)

# =============================================================================
# Render the graph
# =============================================================================
try:
    # Render the flowchart and save it as a PNG file
    output_filename = 'RLAD_v3.2_Flowchart'
    dot.render(output_filename, format='png', view=False, cleanup=True)
    print(f"✅ Flowchart successfully generated and saved as '{output_filename}.png'")
except Exception as e:
    print(f"❌ Error generating flowchart: {e}")
    print("Please ensure Graphviz is installed and configured in your system's PATH.")
