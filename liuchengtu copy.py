import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path  # 修复：导入缺失的 Path 类
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl
import os

# --- 用户配置 ---
# 将下载好的图标文件放置在此脚本所在的目录
# 图标目录设置为当前脚本所在文件夹
ICON_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 优化：将保存目录设置为脚本所在文件夹下的 'output' 子目录，更具可移植性
SAVE_DIR = os.path.join(ICON_DIR, "output")

# --- 样式配置 (科研风格) ---
# 设置全局字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

# 颜色方案 (保持原有优秀配色)
COLOR_PREPROCESS = '#4682B4'  # 钢蓝色
COLOR_STL_LOF = '#D2691E'     # 棕褐色
COLOR_RL = '#228B22'          # 森林绿
COLOR_INFERENCE = '#4B0082'   # 靛蓝色
COLOR_ARROW = '#555555'       # 灰色
COLOR_BG = '#fdfefe'          # 更干净的淡白色背景
COLOR_BOX_EDGE = '#333333'    # 深灰色边框
COLOR_TEXT = '#333333'        # 深灰色文本

def get_icon(path, zoom=0.08):
    """加载图标，如果找不到则打印警告。"""
    if not os.path.exists(path):
        print(f"警告: 图标文件未找到 '{path}'. 将不会显示该图标。")
        return None
    try:
        img = plt.imread(path)
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"错误: 加载图标 '{path}' 失败: {e}")
        return None

def create_stl_lof_rlad_flowchart(save_path="STL_LOF_RLAD_Flowchart_Optimized.png", dpi=300):
    """创建带有图标的、经过优化的STL-LOF-RLAD系统架构流程图。"""
    
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLOR_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    fig.suptitle('STL-LOF-RLAD 系统架构流程图', fontsize=22, fontweight='bold', y=0.97, color=COLOR_TEXT)
    
    # === 1. 数据注入与预处理模块 ===
    preprocess_elements = [
        {'name': '原始压力\n时间序列', 'x': 8, 'y': 75, 'w': 14, 'h': 8, 'icon': 'icon_timeseries.png'},
        {'name': '数据标准化', 'x': 31, 'y': 75, 'w': 14, 'h': 8, 'icon': 'icon_normalize.png'},
        {'name': '滑动窗口分割', 'x': 54, 'y': 75, 'w': 14, 'h': 8, 'icon': 'icon_window.png'},
        {'name': '窗口化样本', 'x': 77, 'y': 75, 'w': 14, 'h': 8, 'icon': 'icon_samples.png'}
    ]
    draw_module(ax, "1. 数据注入与预处理", preprocess_elements, COLOR_PREPROCESS, 50, 90)
    
    # === 2. 无监督信号生成模块 ===
    stl_lof_elements = [
        {'name': 'STL分解', 'x': 8, 'y': 52, 'w': 14, 'h': 8, 'icon': 'icon_stl.png'},
        {'name': '残差提取', 'x': 31, 'y': 52, 'w': 14, 'h': 8, 'icon': 'icon_residual.png'},
        {'name': 'LOF算法', 'x': 54, 'y': 52, 'w': 14, 'h': 8, 'icon': 'icon_lof.png'},
        {'name': '生成伪标签', 'x': 77, 'y': 52, 'w': 14, 'h': 8, 'icon': 'icon_label.png'}
    ]
    draw_module(ax, "2. 无监督信号生成 (STL-LOF)", stl_lof_elements, COLOR_STL_LOF, 50, 67)
    
    # === 3. 强化引导的主动学习模块 ===
    rl_elements = [
        {'name': 'RL智能体\n(BiLSTM+Att)', 'x': 8, 'y': 29, 'w': 18, 'h': 9, 'icon': 'icon_agent.png'},
        {'name': '交互环境', 'x': 35, 'y': 29, 'w': 14, 'h': 9, 'icon': 'icon_env.png'},
        {'name': '主动学习\n标注系统', 'x': 58, 'y': 29, 'w': 14, 'h': 9, 'icon': 'icon_active.png'},
        {'name': '人类专家', 'x': 81, 'y': 29, 'w': 14, 'h': 9, 'icon': 'icon_expert.png'}
    ]
    draw_module(ax, "3. 强化引导的主动学习", rl_elements, COLOR_RL, 50, 45)
    
    # === 4. 推理与定位模块 ===
    inference_elements = [
        {'name': '新数据窗口', 'x': 8, 'y': 6, 'w': 14, 'h': 8, 'icon': 'icon_new_data.png'},
        {'name': '智能体预测', 'x': 31, 'y': 6, 'w': 14, 'h': 8, 'icon': 'icon_predict.png'},
        {'name': '窗口级结果', 'x': 54, 'y': 6, 'w': 14, 'h': 8, 'icon': 'icon_window_result.png'},
        {'name': '点级异常定位', 'x': 77, 'y': 6, 'w': 14, 'h': 8, 'icon': 'icon_locate.png'}
    ]
    draw_module(ax, "4. 推理与定位", inference_elements, COLOR_INFERENCE, 50, 21)
    
    # === 主要流程连接箭头 ===
    # 从“窗口化样本”指向“STL分解”和“交互环境”的起始位置
    draw_vertical_arrow(ax, 84, 75, 61, COLOR_ARROW) 
    # 从“生成伪标签”指向“交互环境”的起始位置
    draw_vertical_arrow(ax, 84, 52, 39, COLOR_ARROW) 
    # 从“RL智能体”指向“新数据窗口”
    draw_vertical_arrow(ax, 17, 29, 15, COLOR_ARROW) 
    
    # === 强化学习与专家反馈的特殊连接 ===
    draw_feedback_loop(ax)
    
    # === 保存图片 ===
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    final_path = os.path.join(SAVE_DIR, save_path)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(final_path, dpi=dpi, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"优化版流程图已保存至: {final_path}")
    plt.close()

def draw_module(ax, title, elements, color, title_x, title_y):
    """绘制一个完整的模块，包括标题、元素框、图标和内部箭头。"""
    ax.text(title_x, title_y, title, color=color, fontsize=16, fontweight='bold', ha='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', edgecolor=color, linewidth=1.5))
    
    for i, element in enumerate(elements):
        rect = patches.FancyBboxPatch(
            (element['x'], element['y']), element['w'], element['h'],
            linewidth=1.5, edgecolor=color, facecolor=mpl.colors.to_rgba(color, 0.1),
            boxstyle='round,pad=0.5', zorder=2)
        ax.add_patch(rect)
        
        icon_path = os.path.join(ICON_DIR, element.get('icon', ''))
        icon = get_icon(icon_path, zoom=0.09) # 稍微增大图标
        if icon:
            ab = AnnotationBbox(icon, (element['x'] + element['w']/2, element['y'] + element['h'] * 0.68),
                                frameon=False, box_alignment=(0.5, 0.5), zorder=4)
            ax.add_artist(ab)
        
        ax.text(element['x'] + element['w']/2, element['y'] + element['h'] * 0.22,
                element['name'], ha='center', va='center', fontsize=10.5,
                fontweight='normal', color=COLOR_TEXT, zorder=3, linespacing=1.3)
        
        if i < len(elements) - 1:
            next_element = elements[i + 1]
            draw_arrow(ax, element['x'] + element['w'], element['y'] + element['h']/2,
                       next_element['x'], next_element['y'] + next_element['h']/2, color)

def draw_arrow(ax, x1, y1, x2, y2, color):
    """绘制模块内部的箭头。"""
    arrow = patches.FancyArrowPatch(
        (x1, y1), (x2, y2), connectionstyle="arc3,rad=0.0",
        arrowstyle='-|>', mutation_scale=20, linewidth=2,
        edgecolor=color, facecolor=color, zorder=1)
    ax.add_patch(arrow)

def draw_vertical_arrow(ax, x, y1, y2, color):
    """绘制模块之间的垂直虚线箭头。"""
    arrow = patches.FancyArrowPatch(
        (x, y1), (x, y2), connectionstyle="arc3,rad=0.0",
        arrowstyle='-|>', mutation_scale=25, linewidth=2.5,
        linestyle=(0, (4, 4)), color=color, zorder=1) # 使用元组定义虚线样式
    ax.add_patch(arrow)

def draw_feedback_loop(ax):
    """优化：绘制从'人类专家'到'RL智能体'的反馈回路，使其更清晰美观。"""
    feedback_color = '#C71585' # 洋红色，更醒目
    style = dict(facecolor='none', edgecolor=feedback_color, linewidth=2,
                 linestyle='--', arrowstyle='-|>')

    # 从“人类专家”出发，向下绕行
    p1 = (88, 29)
    p2 = (88, 25)
    p3 = (17, 25)
    p4 = (17, 29)
    
    # 使用 Path 构建更平滑的路径
    path_data = [
        (p1, Path.MOVETO),
        (p2, Path.LINETO),
        (p3, Path.LINETO),
        (p4, Path.LINETO),
    ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    
    patch = patches.PathPatch(path, **style, mutation_scale=20, zorder=1)
    ax.add_patch(patch)

    ax.text(52.5, 23, "专家标注反馈 (修正奖励与状态)", ha='center', fontsize=10,
            color=feedback_color, fontweight='bold', fontstyle='italic',
            bbox=dict(facecolor=COLOR_BG, alpha=0.8, edgecolor='none', pad=0))

if __name__ == "__main__":
    # 确保保存目录存在
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"已创建目录: {SAVE_DIR}")
        
    create_stl_lof_rlad_flowchart()