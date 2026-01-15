import graphviz
import os

def create_flowchart_image_style():
    """
    根据用户提供的SCI论文配图风格，生成一个高度美化的流程图。
    - 核心特点：灰色块状分区，橙色数据流，蓝色关键模块，大量留白。
    """
    # --- 1. 样式配置 (参考图配色方案) ---
    C_FONT = 'Helvetica'
    C_ARROW = '#D98880'      # 参考图中的柔和橙色 (箭头)
    C_NODE_BLUE = '#85C1E9'   # 参考图中的柔和蓝色 (关键模块)
    C_BG_GRAY = '#F4F6F7'      # 参考图中的浅灰色背景
    C_BORDER_DARK = '#566573' # 深色边框

    # --- 2. 图形初始化 ---
    dot = graphviz.Digraph('RLAD_Flowchart_Image_Style', comment='SCI-Pro Flowchart')
    dot.attr('graph',
             rankdir='LR',
             splines='spline', # 使用曲线箭头更柔和
             nodesep='0.8',    # 节点间距
             ranksep='1.5',    # 层级间距，增加留白
             fontname=C_FONT,
             fontsize='22',
             label='一个融合分解-密度与强化引导主动学习的异常检测框架',
             labelloc='t',
             bgcolor='white')

    # --- 3. 全局节点和边的默认样式 ---
    dot.attr('node', style='filled', fontname=C_FONT, fontsize='15',
             shape='rect', fillcolor='white', color=C_BORDER_DARK, peripheries='1')
    dot.attr('edge', fontname=C_FONT, fontsize='12', color=C_ARROW,
             arrowhead='normal', penwidth='1.5') # 统一设置橙色、加粗的箭头

    # --- 4. 定义节点 ---

    # 使用HTML-like标签来模拟参考图中的输入数据矩阵
    dot.node('data_input', label=r'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR><TD PORT="header" COLSPAN="3" BGCOLOR="#E5E7E9"><b>Y</b> (Raw Time Series)</TD></TR>
        <TR><TD>y₁,₁</TD><TD>...</TD><TD>y₁,ₜ</TD></TR>
        <TR><TD>...</TD><TD>...</TD><TD>...</TD></TR>
        <TR><TD>yₙ,₁</TD><TD>...</TD><TD>yₙ,ₜ</TD></TR>
        </TABLE>>''', shape='plaintext')

    # 阶段一：放置在灰色背景的"块"中
    with dot.subgraph(name='cluster_phase1') as c:
        c.attr(label='Phase 1: Signal Decomposition', style='filled', color=C_BG_GRAY,
               fontname=C_FONT, fontsize='18')
        c.node('stl_decomp', 'STL Decomposition', fillcolor=C_NODE_BLUE)
        c.node('residual', '<b>R<sub>t</sub></b> (Residual Series)', peripheries='2') # 用双边框突出

    # 阶段二：同样放置在灰色块中
    with dot.subgraph(name='cluster_phase2') as c:
        c.attr(label='Phase 2: Pseudo-Label Generation', style='filled', color=C_BG_GRAY,
               fontname=C_FONT, fontsize='18')
        c.node('lof_score', 'LOF Anomaly Scoring', fillcolor=C_NODE_BLUE)
        c.node('pseudo_label', '<b>L<sub>p</sub></b> (Pseudo-Label Set)', peripheries='2')

    # ---------------- 阶段三：强化学习 ----------------
    with dot.subgraph(name='cluster_phase3') as c:
        c.attr(label='Phase 3: Reinforcement-Guided Active Learning', style='filled', color=C_BG_GRAY,
               fontname=C_FONT, fontsize='18')

        # Agent也作为一个内部的块
        with c.subgraph(name='cluster_agent') as agent:
            agent.attr(label='RLAD Agent', style='rounded,filled', color='white')
            agent.node('bilstm', 'Bi-LSTM Encoder', fillcolor=C_NODE_BLUE)
            agent.node('q_network', 'DQN & Policy', fillcolor=C_NODE_BLUE)
            agent.edge('bilstm', 'q_network')

        c.node('replay_buffer', 'Experience Replay\n(with Expert Labels)')
        c.node('expert', 'Domain Expert', shape='ellipse')

    # --- 5. 定义数据流 (Edges) ---

    # 主要数据流 (实线橙色)
    dot.edge('data_input:header', 'stl_decomp')
    dot.edge('stl_decomp', 'residual')
    dot.edge('residual', 'lof_score')
    dot.edge('lof_score', 'pseudo_label')
    dot.edge('pseudo_label', 'replay_buffer', label=' Inject Initial Data')

    # 强化学习循环
    dot.edge('replay_buffer', 'bilstm', label=' Sampled Batch s', lhead='cluster_agent')
    dot.edge('q_network', 'replay_buffer', label=' Action a', ltail='cluster_agent')

    # 主动学习循环 (虚线橙色，参考图风格)
    dot.edge('q_network', 'expert', label=' Uncertainty Query', style='dashed', ltail='cluster_agent')
    dot.edge('expert', 'replay_buffer', style='dashed')


    # --- 6. 渲染并保存文件 ---
    try:
        output_filename = 'flowchart_sci_style_new'
        dot.render(output_filename, format='svg', view=False, cleanup=True)
        print(f"✅ 成功！已生成参考图风格的流程图：'{output_filename}.svg'")
    except graphviz.backend.execute.CalledProcessError as e:
        print(f"❌ 渲染失败: {e}")
        print(f"   请确保已正确安装Graphviz并将其bin目录添加至系统PATH。")

if __name__ == '__main__':
    create_flowchart_image_style()