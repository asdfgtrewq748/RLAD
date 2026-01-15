import graphviz
import os

def create_minimalist_sci_flowchart_en_final():
    """
    Generates a minimalist, professional flowchart suitable for top-tier scientific papers.
    This version fixes the AttributeError bug by correctly referencing nodes by their string name.
    """
    # --- 1. Style & Color Palette Configuration ---
    C_FONT = 'Helvetica'
    C_BORDER = '#444444'
    C_HIGHLIGHT1 = '#E6F3E6'
    C_HIGHLIGHT2 = '#FFF2E6'

    # --- 2. Graph Initialization ---
    dot = graphviz.Digraph('STL_System_Architecture_Minimalist_EN',
                           comment='Minimalist SCI-Pro Flowchart (English)')
    dot.attr('graph',
             rankdir='LR',
             splines='ortho',
             nodesep='1.0',
             ranksep='2.0',
             fontname=C_FONT,
             fontsize='18',
             label='A Framework for Anomaly Detection Fusing Decomposition, Density Methods, and Reinforcement-Guided Active Learning',
             labelloc='t',
             bgcolor='white')

    dot.attr('node', style='filled', fontname=C_FONT, fontsize='14',
             shape='rect', fillcolor='white', color=C_BORDER)
    dot.attr('edge', fontname=C_FONT, fontsize='11', color='#555555')

    # --- 3. Phase 1: Signal Decomposition ---
    with dot.subgraph(name='cluster_phase1') as c:
        c.attr(label='Phase 1: Signal Decomposition', style='dashed', color=C_BORDER, fontname=C_FONT, fontsize='16')
        c.node('data_input', '<b>Y</b> = {y₁, y₂, ..., yₙ}\n(Raw Time Series)')
        c.node('stl_decomp', 'STL Decomposition')
        c.node('residual', '<b>R<sub>t</sub></b>\n(Residual Series)', fillcolor=C_HIGHLIGHT2)

    # --- 4. Phase 2: Pseudo-Label Generation ---
    with dot.subgraph(name='cluster_phase2') as c:
        c.attr(label='Phase 2: Pseudo-Label Generation', style='dashed', color=C_BORDER, fontname=C_FONT, fontsize='16')
        c.node('lof_score', 'LOF Anomaly Scoring')
        c.node('pseudo_label', '<b>L<sub>p</sub></b>\n(Pseudo-Label Set)', fillcolor=C_HIGHLIGHT2)

    # --- 5. Phase 3: Reinforcement-Guided Active Learning ---
    with dot.subgraph(name='cluster_phase3') as c:
        c.attr(label='Phase 3: Reinforcement-Guided Active Learning', style='dashed', color=C_BORDER, fontname=C_FONT, fontsize='16')

        with c.subgraph(name='cluster_agent') as agent:
            agent.attr(label='RLAD Agent', style='dashed', color=C_BORDER)
            agent.node('bilstm', 'Bi-LSTM\n(State Encoder)', fillcolor=C_HIGHLIGHT1)
            agent.node('q_network', 'DQN & Policy\nQ(s,a;θ)', fillcolor=C_HIGHLIGHT1)
            agent.edge('bilstm', 'q_network', label=' Encoded State')
        
        c.node('replay_buffer', 'Prioritized Experience Replay\n(Replay Buffer)')
        c.node('expert_query', 'Active Learning Query\n(Uncertainty Sampling)')
        c.node('expert', 'Domain Expert')

        # FIX: Edges now use string names ('q_network', 'replay_buffer') instead of variables.
        c.edge('replay_buffer', 'bilstm', label=' Sampled Batch s', lhead='cluster_agent')
        c.edge('q_network', 'replay_buffer', label=' Action a', ltail='cluster_agent')
        c.edge('q_network', 'expert_query', style='dashed', ltail='cluster_agent')
        
        dot.edge('q_network', 'replay_buffer', label=" Store Experience (s,a,r,s')",
                 ltail='cluster_agent', style='dashed', dir='back', constraint='false')
        
        c.edge('expert_query', 'expert', style='dashed')
        c.edge('expert', 'replay_buffer', label=' Expert Label', style='dashed')

    # --- 6. Connecting the Main Phases ---
    dot.edge('data_input', 'stl_decomp')
    dot.edge('stl_decomp', 'residual', style='bold', penwidth='1.5')
    dot.edge('residual', 'lof_score', style='bold', penwidth='1.5')
    dot.edge('lof_score', 'pseudo_label', style='bold', penwidth='1.5')
    dot.edge('pseudo_label', 'replay_buffer', label=' Inject Initial Data', style='bold', penwidth='1.5')

    # --- 7. Render and Save the Files ---
    try:
        output_filename = 'stl_flowchart_minimalist_en_final'
        dot.render(output_filename, format='svg', view=False, cleanup=True)
        dot.render(output_filename, format='pdf', view=False, cleanup=True)
        print(f"✅ Success! Generated final flowchart: '{output_filename}.svg'")
        print(f"   Also generated '{output_filename}.pdf' (recommended for final paper submission)")

    except graphviz.backend.execute.ExecutableNotFound:
        print("❌ Error: Graphviz executable 'dot' not found.")
        print("Please ensure you have installed Graphviz (https://graphviz.org/download/)")
        print("and that its 'bin' directory is in your system's PATH environment variable.")

if __name__ == '__main__':
    create_minimalist_sci_flowchart_en_final()