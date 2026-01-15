# main_gui.py
import sys
import time
import os
import json
import traceback
from datetime import datetime
import pandas as pd
import multiprocessing  # 1. 导入 multiprocessing 模块
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox,
                             QFormLayout, QFileDialog, QSpinBox, QCheckBox, QProgressBar, QDoubleSpinBox,
                             QListWidget, QComboBox, QScrollArea)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices

# ------------------- Plotting Configuration for Publication Quality -------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.titlesize'] = 16

def plot_sobol_indices(Si, feature_names, output_path):
    """
    为科研论文绘制并保存Sobol指数的条形图 (PDF格式)。
    """
    s1_indices = Si['S1']
    st_indices = Si['ST']
    
    sorted_indices = np.argsort(st_indices)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_s1 = s1_indices[sorted_indices]
    sorted_st = st_indices[sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.35
    index = np.arange(len(sorted_names))

    bars1 = ax.bar(index - bar_width/2, sorted_s1, bar_width, label='一阶指数 (S1)', color='cornflowerblue')
    bars2 = ax.bar(index + bar_width/2, sorted_st, bar_width, label='总阶指数 (ST)', color='salmon')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylabel('Sobol 指数')
    ax.set_title('全局敏感性分析 (Sobol 指数)')
    ax.set_xticks(index)
    ax.set_xticklabels(sorted_names, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class AnalysisWorker(QObject):
    """
    分析工作线程，用于在后台执行耗时任务，避免GUI冻结。
    """
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    finished = pyqtSignal(str, str, object) # report_path, plot_path, Si_dict

    def __init__(self, file_path, params):
        super().__init__()
        self.file_path = file_path
        self.params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _check_cancel(self):
        if self._cancelled:
            raise RuntimeError("用户取消了分析任务。")

    def run(self):
        """
        执行完整的Sobol敏感性分析工作流。
        """
        try:
            self.progress.emit("🚀 开始执行敏感性分析工作流...")
            t0 = time.time()
            self.progress_percent.emit(2)
            
            # 1. 加载数据
            self.progress.emit(f"   - 正在加载数据: {self.file_path}")
            try:
                df = pd.read_csv(self.file_path, encoding='gbk')
                self.progress.emit("   - 数据加载成功 (GBK编码)。")
            except Exception:
                df = pd.read_csv(self.file_path, encoding='utf-8')
                self.progress.emit("   - 数据加载成功 (UTF-8编码)。")
            self._check_cancel()

            # 选择特征与目标：优先使用用户选择；否则使用原切片；再否则使用数值列回退
            try:
                selected_target = self.params.get('selected_target')
                selected_features = self.params.get('selected_features')
                if selected_target and selected_features:
                    # 校验列存在
                    missing = [c for c in [selected_target] + selected_features if c not in df.columns]
                    if missing:
                        raise ValueError(f"所选列不存在: {missing}")
                    X = df.loc[:, selected_features]
                    y = df.loc[:, selected_target]
                elif df.shape[1] >= 9:
                    X = df.iloc[:, 1:8]
                    y = df.iloc[:, -1]
                else:
                    numeric_df = df.select_dtypes(include=[np.number]).copy()
                    if numeric_df.shape[1] < 2:
                        raise ValueError("数据中数值列少于2列，无法进行建模。")
                    X = numeric_df.iloc[:, :-1]
                    y = numeric_df.iloc[:, -1]
                feature_names = list(X.columns)
            except Exception as e:
                raise RuntimeError(f"选择特征/目标列失败: {e}")
            n_samples = X.shape[0]
            self.progress.emit(f"   - 当前样本量: {n_samples}")
            small_data = n_samples <= 40
            if small_data:
                self.progress.emit("   - 触发少样本优化：将使用留一交叉验证(LOOCV)进行模型选择，并跳过独立测试集切分。")
            
            time.sleep(0.5)
            self.progress_percent.emit(10)
            self._check_cancel()

            # 2. 训练代理模型
            self.progress.emit("\n⚙️ 步骤1: 训练随机森林代理模型...")
            if small_data:
                X_train, y_train = X, y
                X_test, y_test = None, None
                # 对于小样本，LOOCV计算量太大，改为5折交叉验证
                cv_strategy = 5 
                self.progress.emit("   - 少样本优化：使用5折交叉验证。")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                cv_strategy = 3
            
            # --- FIX: Robust parameter parsing ---
            def parse_max_features(vals_str):
                parsed = []
                for v in vals_str.split(','):
                    v_strip = v.strip().lower()
                    if v_strip in ['sqrt', 'log2']:
                        parsed.append(v_strip)
                    else:
                        try:
                            parsed.append(float(v_strip))
                        except ValueError:
                            self.progress.emit(f"⚠️ 无法解析 max_features 值 '{v}'，已跳过。")
                return parsed

            param_grid = {
                'n_estimators': [int(n.strip()) for n in self.params['n_estimators'].split(',')],
                'max_depth': [int(md.strip()) if md.strip().lower() != 'none' else None for md in self.params['max_depth'].split(',')],
                'min_samples_split': [int(mss.strip()) for mss in self.params['min_samples_split'].split(',')],
                'min_samples_leaf': [int(msl.strip()) for msl in self.params['min_samples_leaf'].split(',')],
                'max_features': parse_max_features(self.params['max_features'])
            }
            
            # 少样本时，适度抬高 min_samples_leaf 以降低过拟合风险
            if small_data:
                param_grid['min_samples_leaf'] = sorted({max(2, v) for v in param_grid['min_samples_leaf']})

            self.progress.emit(f"   - 使用参数网格进行GridSearchCV: {param_grid}")
            scoring_metric = 'neg_mean_squared_error' if small_data else 'r2'
            # 2. 将 n_jobs 设置为 1，避免在GUI线程外创建过多进程
            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42, n_jobs=1),
                                               param_grid=param_grid, cv=cv_strategy, n_jobs=1, scoring=scoring_metric, verbose=2)
            
            t1 = time.time()
            grid_search.fit(X_train, y_train)
            best_rf_surrogate = grid_search.best_estimator_
            
            self.progress.emit(f"✅ 代理模型训练完成。最佳参数: {grid_search.best_params_}")
            if small_data:
                try:
                    # OOB score is a good indicator for small datasets
                    if hasattr(best_rf_surrogate, 'oob_score_') and best_rf_surrogate.oob_score_:
                         self.progress.emit(f"   袋外(OOB) R² 分数: {best_rf_surrogate.oob_score_:.4f}")
                    cv_predictions = cross_val_predict(best_rf_surrogate, X_train, y_train, cv=cv_strategy, n_jobs=1)
                    cv_r2 = r2_score(y_train, cv_predictions)
                    cv_rmse = float(np.sqrt(np.mean((y_train.to_numpy() - cv_predictions) ** 2)))
                    self.progress.emit(f"   留一交叉验证(LOOCV) R²: {cv_r2:.4f} | RMSE: {cv_rmse:.4f}")
                    if cv_r2 < 0.7:
                        self.progress.emit("   ⚠️ 警告: 在少样本条件下代理模型拟合优度较低，可考虑更保守的参数或更简单的模型。")
                except Exception as eval_err:
                    self.progress.emit(f"   ⚠️ LOOCV 评估阶段出现异常: {eval_err}")
                    self.progress.emit("   ⚠️ 建议检查数据质量或暂时禁用增强选项。")
            else:
                y_pred_test = best_rf_surrogate.predict(X_test)
                r2_test = r2_score(y_test, y_pred_test)
                self.progress.emit(f"   在测试集上的R² (拟合优度): {r2_test:.4f}")
                if r2_test < 0.7:
                    self.progress.emit("   ⚠️ 警告: 代理模型拟合优度较低，结果可靠性可能受影响。")
            self.progress.emit(f"   - 训练耗时: {time.time() - t1:.2f} 秒。")

            # 2.1 可选：数据增强（仅对训练数据进行，不泄漏）
            if bool(self.params.get('augment_enabled', False)):
                try:
                    self.progress.emit("\n🧪 启用数据增强：抖动 + 伪标签 训练...")
                    K = int(self.params.get('augment_repeat', 5))
                    noise_pct = float(self.params.get('augment_noise_pct', 10.0)) / 100.0
                    aug_weight = float(self.params.get('augment_weight', 0.3))
                    # 统计量基于训练集
                    mu = X_train.mean(axis=0).to_numpy()
                    sigma = X_train.std(axis=0, ddof=0).to_numpy()
                    # 避免零方差
                    sigma = np.where(np.abs(sigma) < 1e-12, np.maximum(1e-12, np.abs(mu) * 1e-6), sigma)
                    n_train, n_feat = X_train.shape
                    X_aug_list = []
                    for _ in range(K):
                        noise = np.random.normal(loc=0.0, scale=sigma * noise_pct, size=(n_train, n_feat))
                        X_aug_list.append(X_train.to_numpy() + noise)
                    X_aug = np.vstack(X_aug_list)
                    # 使用当前最佳代理模型生成伪标签
                    y_aug = best_rf_surrogate.predict(X_aug)
                    # 合并数据
                    X_train_final = np.vstack([X_train.to_numpy(), X_aug])
                    y_train_final = np.concatenate([y_train.to_numpy(), y_aug])
                    sample_weight = np.concatenate([np.ones(n_train), np.full(X_aug.shape[0], aug_weight)])
                    # 使用相同超参重新拟合最终模型
                    best_params = grid_search.best_params_.copy()
                    # 2. 将 n_jobs 设置为 1
                    final_rf = RandomForestRegressor(random_state=42, n_jobs=1, oob_score=small_data, **best_params)
                    t2 = time.time()
                    final_rf.fit(X_train_final, y_train_final, sample_weight=sample_weight)
                    self.progress.emit(f"   - 数据增强后训练完成，用时: {time.time() - t2:.2f} 秒。")
                    self.progress.emit(f"   - 训练样本数: 原始 {n_train} 条，合成 {X_aug.shape[0]} 条，总计 {X_train_final.shape[0]} 条。")
                    # 评价：小样本则报告 OOB 分数（若可用）；否则在测试集上评估
                    if small_data and hasattr(final_rf, 'oob_score_'):
                        self.progress.emit(f"   - OOB R²: {final_rf.oob_score_:.4f} (仅供参考)")
                    elif not small_data:
                        r2_test_refit = r2_score(y_test, final_rf.predict(X_test))
                        self.progress.emit(f"   - 重新拟合后在测试集上的R²: {r2_test_refit:.4f}")
                    # 覆盖后续使用的代理模型
                    best_rf_surrogate = final_rf
                except Exception as e:
                    self.progress.emit(f"⚠️ 数据增强阶段发生错误，已回退到未增强模型: {e}")
            
            time.sleep(0.5)
            self.progress_percent.emit(35)
            self._check_cancel()

            # 3. 定义Sobol分析问题
            self.progress.emit("\n⚙️ 步骤2: 定义Sobol分析问题...")
            problem = {
                'num_vars': len(feature_names),
                'names': feature_names,
                'bounds': []
            }
            # 避免 min==max 导致采样报错，必要时微扩展边界
            for name in feature_names:
                vmin, vmax = float(np.nanmin(X[name])), float(np.nanmax(X[name]))
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    raise RuntimeError(f"特征 {name} 的边界包含非数值/无穷值。")
                if vmin == vmax:
                    eps = max(1e-8, abs(vmin) * 1e-6)
                    vmin -= eps
                    vmax += eps
                problem['bounds'].append([vmin, vmax])
            self.progress.emit(f"   - 变量数量: {problem['num_vars']}, 名称: {problem['names']}")
            
            time.sleep(0.5)
            self.progress_percent.emit(45)
            self._check_cancel()

            # 4. 生成Saltelli样本
            self.progress.emit("\n⚙️ 步骤3: 生成Saltelli样本...")
            N = int(self.params['sobol_n'])
            calc_second = bool(self.params.get('calc_second_order', False))
            param_values = saltelli.sample(problem, N=N, calc_second_order=calc_second)
            self.progress.emit(f"   - 已生成 {param_values.shape[0]} 个样本点 (N={N}, calc_second_order={calc_second})。")
            
            time.sleep(0.5)
            self.progress_percent.emit(60)
            self._check_cancel()

            # 5. 运行代理模型
            self.progress.emit("\n⚙️ 步骤4: 在代理模型上评估样本...")
            Y_pred = best_rf_surrogate.predict(param_values)
            self.progress.emit("   - 样本评估完成。")
            
            time.sleep(0.5)
            self.progress_percent.emit(70)
            self._check_cancel()

            # 6. 执行Sobol分析
            self.progress.emit("\n⚙️ 步骤5: 执行Sobol分析并计算指数...")
            Si = sobol.analyze(problem, Y_pred, print_to_console=False) # Set to False for GUI
            
            # Manually create report string
            s1 = Si.get('S1')
            s1_conf = Si.get('S1_conf')
            st = Si.get('ST')
            st_conf = Si.get('ST_conf')

            report_lines = []
            report_lines.append("--- Sobol Analysis Results ---")
            report_lines.append(f"Parameters: {problem['names']}")
            report_lines.append("")
            report_lines.append("S1 Indices (with 95% CI):")
            for i, name in enumerate(problem['names']):
                ci = f" ± {s1_conf[i]:.4f}" if s1_conf is not None else ""
                report_lines.append(f"  {name}: {s1[i]:.4f}{ci}")
            report_lines.append("")
            report_lines.append("ST Indices (Total Order, with 95% CI):")
            for i, name in enumerate(problem['names']):
                ci = f" ± {st_conf[i]:.4f}" if st_conf is not None else ""
                report_lines.append(f"  {name}: {st[i]:.4f}{ci}")
            report_lines.append("")
            report_lines.append(f"Elapsed time: {time.time() - t0:.2f} s")
            report_str = "\n".join(report_lines)
            
            self.progress.emit(report_str)
            self.progress.emit("✅ Sobol分析计算完成。")
            
            time.sleep(0.5)
            self.progress_percent.emit(80)
            self._check_cancel()

            # 7. 可视化结果
            self.progress.emit("\n⚙️ 步骤6: 生成并保存结果图表...")
            # 输出目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = os.path.join(os.getcwd(), f'sobol_results_{timestamp}')
            _ensure_dir(out_dir)
            plot_path = os.path.join(out_dir, 'sobol_sensitivity_analysis_gui.pdf')
            plot_sobol_indices(Si, feature_names, plot_path)
            self.progress.emit(f"✅ 敏感性分析图表已保存为: '{plot_path}'")
            
            # 8. 保存报告
            report_path = os.path.join(out_dir, 'sobol_analysis_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            self.progress.emit(f"✅ 分析报告已保存为: '{report_path}'")

            # 9. 保存CSV数据，便于后续复用
            try:
                df_sobol = pd.DataFrame({
                    'feature': problem['names'],
                    'S1': s1,
                    'S1_conf': s1_conf,
                    'ST': st,
                    'ST_conf': st_conf
                })
                csv_path = os.path.join(out_dir, 'sobol_indices.csv')
                df_sobol.to_csv(csv_path, index=False, encoding='utf-8-sig')
                # 也保存原始Si为json（去除非可序列化类型）
                si_json_path = os.path.join(out_dir, 'sobol_indices_raw.json')
                with open(si_json_path, 'w', encoding='utf-8') as jf:
                    json.dump({k: (np.array(v).tolist() if isinstance(v, (list, np.ndarray)) else v)
                               for k, v in Si.items()}, jf, ensure_ascii=False, indent=2)
                self.progress.emit(f"✅ 指数数据已保存为: '{csv_path}', '{si_json_path}'")
            except Exception as e:
                self.progress.emit(f"⚠️ 保存指数数据失败: {e}")

            self.progress.emit("\n🎉 工作流执行完毕。")
            self.progress_percent.emit(100)
            self.finished.emit(report_path, plot_path, Si)

        except Exception as e:
            tb = traceback.format_exc()
            self.progress.emit(f"\n❌ 分析过程中发生错误: {e}\n{tb}")
            self.finished.emit("", "", None)

# --- Modern StyleSheet (inspired by Xiaomi/Huawei) ---
STYLESHEET = """
QMainWindow {
    background-color: #f0f2f5;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    margin-top: 10px;
    font-size: 14px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px 15px;
    background-color: #ffffff;
    border-radius: 12px;
    color: #333333;
}
QLabel {
    color: #333333;
    font-size: 13px;
}
QLineEdit, QTextEdit, QSpinBox, QComboBox {
    background-color: #f7f7f7;
    border: 1px solid #dcdcdc;
    border-radius: 6px;
    padding: 8px;
    font-size: 13px;
    color: #333333;
}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #007aff;
}
QPushButton {
    background-color: #007aff;
    color: white;
    font-size: 14px;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}
QPushButton:hover {
    background-color: #005ecb;
}
QPushButton:pressed {
    background-color: #004a9e;
}
QPushButton:disabled {
    background-color: #dcdcdc;
    color: #a0a0a0;
}
QTextEdit {
    font-family: 'Courier New', monospace;
}
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("全局敏感性分析工具")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet(STYLESHEET)
        
        self.worker = None
        self.thread = None
        self.output_plot_path = ""
        self.output_report_path = ""
        self.last_output_dir = ""

        self.init_ui()

    def init_ui(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- File Selection Group ---
        file_group = QGroupBox("1. 选择数据文件")
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("请选择一个 .csv 文件")
        self.file_path_edit.setReadOnly(True)
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(self.browse_file)
        browse_button.setToolTip("选择包含特征与目标列的 CSV 数据文件")
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_button)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        info_label = QLabel("提示：请依次完成①选择数据 → ②加载列并设置特征/目标 → ③调整模型参数 → ④可选数据增强 → ⑤开始分析。")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color:#555555;")
        main_layout.addWidget(info_label)

        # --- Feature/Target Selection Group ---
        ft_group = QGroupBox("2. 选择特征/目标列")
        ft_layout = QVBoxLayout()
        ft_controls = QHBoxLayout()
        self.only_numeric_checkbox = QCheckBox("仅显示数值列")
        self.only_numeric_checkbox.setChecked(True)
        self.only_numeric_checkbox.setToolTip("仅显示数值列以避免选择非数值字段")
        self.load_cols_button = QPushButton("加载列")
        self.load_cols_button.clicked.connect(self.load_columns)
        self.load_cols_button.setToolTip("从选定的 CSV 中读取列名")
        ft_controls.addWidget(self.only_numeric_checkbox)
        ft_controls.addWidget(self.load_cols_button)
        ft_layout.addLayout(ft_controls)

        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("目标列:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.setToolTip("加载列后选择目标变量")
        target_layout.addWidget(self.target_combo)
        ft_layout.addLayout(target_layout)

        self.features_list = QListWidget()
        self.features_list.setSelectionMode(self.features_list.SelectionMode.ExtendedSelection)
        self.features_list.setEnabled(False)
        self.features_list.setToolTip("加载列后选择一个或多个特征列")
        ft_layout.addWidget(QLabel("特征列（多选）："))
        ft_layout.addWidget(self.features_list)

        features_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_clear_selection = QPushButton("清空")
        self.btn_select_all.setEnabled(False)
        self.btn_clear_selection.setEnabled(False)
        self.btn_select_all.clicked.connect(self.select_all_features)
        self.btn_clear_selection.clicked.connect(self.clear_feature_selection)
        features_btn_layout.addWidget(self.btn_select_all)
        features_btn_layout.addWidget(self.btn_clear_selection)
        features_btn_layout.addStretch(1)
        ft_layout.addLayout(features_btn_layout)

        self.selection_summary_label = QLabel("尚未加载列")
        self.selection_summary_label.setStyleSheet("color:#666666;font-size:12px;")
        ft_layout.addWidget(self.selection_summary_label)

        ft_group.setLayout(ft_layout)
        main_layout.addWidget(ft_group)

        # --- Parameters Group ---
        params_group = QGroupBox("3. 设置模型参数")
        params_layout = QFormLayout()
        params_layout.setSpacing(10)

        self.rf_n_estimators = QLineEdit("200, 400")
        self.rf_n_estimators.setToolTip("随机森林树数量候选值，逗号分隔。")
        self.rf_max_depth = QLineEdit("6, 10, None")
        self.rf_max_depth.setToolTip("树深度候选值，可输入 None。")
        self.rf_min_samples_split = QLineEdit("2, 5")
        self.rf_min_samples_split.setToolTip("内部节点再划分所需最小样本数候选值。")
        self.rf_min_samples_leaf = QLineEdit("1, 3")
        self.rf_min_samples_leaf.setToolTip("叶节点最少样本数候选值。")
        self.rf_max_features = QLineEdit("sqrt, 0.8")
        self.rf_max_features.setToolTip("寻找最佳分割时考虑的特征数量。'sqrt', 'log2', 或 0.0-1.0 之间的小数")
        self.sobol_n = QSpinBox()
        self.sobol_n.setRange(256, 8192)
        self.sobol_n.setValue(1024)  # 默认更大的N以获得稳定估计
        self.sobol_n.setSingleStep(256)
        self.sobol_n.setToolTip("Saltelli 采样的基数 N，实际样本量与变量数量有关")
        self.chk_second_order = QCheckBox("计算二阶Sobol指数 (更慢)")
        self.chk_second_order.setChecked(False)
        self.chk_second_order.setToolTip("启用二阶指数将显著增加采样量和耗时")

        params_layout.addRow("RF n_estimators (逗号分隔):", self.rf_n_estimators)
        params_layout.addRow("RF max_depth (逗号分隔):", self.rf_max_depth)
        params_layout.addRow("RF min_samples_split (逗号分隔):", self.rf_min_samples_split)
        params_layout.addRow("RF min_samples_leaf (逗号分隔):", self.rf_min_samples_leaf)
        params_layout.addRow("RF max_features (逗号分隔):", self.rf_max_features)
        params_layout.addRow("Sobol 样本数 (N):", self.sobol_n)
        params_layout.addRow(" ", self.chk_second_order)
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # --- Data Augmentation Group ---
        augment_group = QGroupBox("4. 数据增强（可选）")
        augment_layout = QFormLayout()
        augment_layout.setSpacing(10)
        self.chk_augment = QCheckBox("启用数据增强（抖动 + 伪标签）")
        self.chk_augment.setChecked(False)
        self.aug_repeat = QSpinBox()
        self.aug_repeat.setRange(1, 50)
        self.aug_repeat.setValue(5)
        self.aug_repeat.setSingleStep(1)
        self.aug_noise_pct = QDoubleSpinBox()
        self.aug_noise_pct.setRange(0.1, 50.0)
        self.aug_noise_pct.setDecimals(1)
        self.aug_noise_pct.setSingleStep(0.5)
        self.aug_noise_pct.setSuffix(" %")
        self.aug_noise_pct.setValue(10.0)
        self.aug_weight = QDoubleSpinBox()
        self.aug_weight.setRange(0.05, 1.0)
        self.aug_weight.setSingleStep(0.05)
        self.aug_weight.setValue(0.3)
        self.chk_augment.setToolTip("对训练集进行抖动并使用伪标签扩增数据")
        self.aug_repeat.setToolTip("每条样本扩增 K 倍")
        self.aug_noise_pct.setToolTip("噪声强度按特征标准差的百分比")
        self.aug_weight.setToolTip("合成样本在训练中的权重")
        augment_layout.addRow(self.chk_augment)
        augment_layout.addRow("每条样本扩增倍数 (K):", self.aug_repeat)
        augment_layout.addRow("噪声强度（按特征标准差的%）:", self.aug_noise_pct)
        augment_layout.addRow("合成样本权重 (0.05-1.0):", self.aug_weight)
        augment_group.setLayout(augment_layout)
        main_layout.addWidget(augment_group)

        # --- Run Button ---
        self.run_button = QPushButton("开始分析")
        self.run_button.clicked.connect(self.run_analysis)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        run_layout = QHBoxLayout()
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.cancel_button)
        main_layout.addLayout(run_layout)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        main_layout.addWidget(self.progress_bar)

        # --- Output/Log Group ---
        log_group = QGroupBox("5. 分析日志")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("分析过程中的日志会显示在这里…")
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # --- Export Group ---
        export_group = QGroupBox("6. 导出结果")
        export_layout = QHBoxLayout()
        self.export_report_button = QPushButton("导出报告 (.txt)")
        self.export_plot_button = QPushButton("保存图表 (.pdf)")
        self.open_output_button = QPushButton("打开结果文件夹")
        self.export_report_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        self.export_report_button.clicked.connect(self.export_report)
        self.export_plot_button.clicked.connect(self.export_plot)
        self.open_output_button.clicked.connect(self.open_output_dir)
        export_layout.addWidget(self.export_report_button)
        export_layout.addWidget(self.export_plot_button)
        export_layout.addWidget(self.open_output_button)
        export_group.setLayout(export_layout)
        main_layout.addWidget(export_group)

        # Signals for dynamic updates
        self.features_list.itemSelectionChanged.connect(self.update_feature_selection_summary)
        self.target_combo.currentTextChanged.connect(self.update_feature_selection_summary)

        # Store input controls for unified enable/disable handling
        self._input_controls = [
            self.load_cols_button,
            self.only_numeric_checkbox,
            self.target_combo,
            self.features_list,
            self.btn_select_all,
            self.btn_clear_selection,
            self.rf_n_estimators,
            self.rf_max_depth,
            self.rf_min_samples_split,
            self.rf_min_samples_leaf,
            self.rf_max_features,
            self.sobol_n,
            self.chk_second_order,
            self.chk_augment,
            self.aug_repeat,
            self.aug_noise_pct,
            self.aug_weight
        ]

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV Files (*.csv)")
        if file_name:
            self.file_path_edit.setText(file_name)
            # 尝试自动加载列
            self.load_columns()

    def load_columns(self):
        path = self.file_path_edit.text()
        if not path:
            self.log_display.append("ℹ️ 请先选择CSV文件，再加载列名。")
            return
        try:
            try:
                df = pd.read_csv(path, encoding='gbk')
            except Exception:
                df = pd.read_csv(path, encoding='utf-8')
            cols_all = list(df.columns)
            if self.only_numeric_checkbox.isChecked():
                df_num = df.select_dtypes(include=[np.number])
                cols = [c for c in cols_all if c in df_num.columns]
            else:
                cols = cols_all
            # 填充目标下拉与特征列表
            self.target_combo.clear()
            self.target_combo.addItems(cols)
            self.features_list.clear()
            self.features_list.addItems(cols)
            # 默认选择：最后一列作为目标，其余为特征
            if cols:
                self.target_combo.setCurrentIndex(len(cols) - 1)
                for i in range(len(cols) - 1):
                    self.features_list.item(i).setSelected(True)
            has_cols = len(cols) > 0
            self.target_combo.setEnabled(has_cols)
            self.features_list.setEnabled(has_cols)
            self.btn_select_all.setEnabled(has_cols)
            self.btn_clear_selection.setEnabled(has_cols)
            self.update_feature_selection_summary()
            self.log_display.append(f"✅ 已加载列，共 {len(cols)} 列（仅数值={self.only_numeric_checkbox.isChecked()}）。")
        except Exception as e:
            self.log_display.append(f"❌ 加载列失败: {e}")
            self.target_combo.setEnabled(False)
            self.features_list.setEnabled(False)
            self.btn_select_all.setEnabled(False)
            self.btn_clear_selection.setEnabled(False)
            self.selection_summary_label.setText("加载列失败")

    def select_all_features(self):
        if not self.features_list.isEnabled():
            return
        for i in range(self.features_list.count()):
            self.features_list.item(i).setSelected(True)
        self.update_feature_selection_summary()

    def clear_feature_selection(self):
        if not self.features_list.isEnabled():
            return
        self.features_list.clearSelection()
        self.update_feature_selection_summary()

    def update_feature_selection_summary(self):
        if not self.features_list.isEnabled():
            self.selection_summary_label.setText("尚未加载列")
            return
        target = self.target_combo.currentText() if self.target_combo.currentText() else "未选择"
        selected = [item.text() for item in self.features_list.selectedItems()]
        summary = f"目标列：{target}  |  已选特征：{len(selected)} 列"
        if not selected:
            summary += "（请至少选择一列特征）"
        self.selection_summary_label.setText(summary)

    def run_analysis(self):
        file_path = self.file_path_edit.text()
        if not file_path:
            self.log_display.append("❌ 请先选择一个数据文件！")
            return

        selected_target = self.target_combo.currentText() if self.target_combo.isEnabled() else None
        selected_features = [item.text() for item in self.features_list.selectedItems()] if self.features_list.isEnabled() else []

        if not selected_target:
            self.log_display.append("❌ 请先加载列并选择一个目标列。")
            return
        if not selected_features:
            self.log_display.append("❌ 请至少选择一列特征。")
            return

        self.run_button.setEnabled(False)
        self.run_button.setText("分析中…")
        self.cancel_button.setEnabled(True)
        self.export_report_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        self.log_display.clear()
        self.progress_bar.setValue(0)

        for widget in self._input_controls:
            widget.setEnabled(False)

        params = {
            'n_estimators': self.rf_n_estimators.text(),
            'max_depth': self.rf_max_depth.text(),
            'min_samples_split': self.rf_min_samples_split.text(),
            'min_samples_leaf': self.rf_min_samples_leaf.text(),
            'max_features': self.rf_max_features.text(),
            'sobol_n': self.sobol_n.value(),
            'calc_second_order': self.chk_second_order.isChecked(),
            'augment_enabled': self.chk_augment.isChecked(),
            'augment_repeat': self.aug_repeat.value(),
            'augment_noise_pct': float(self.aug_noise_pct.value()),
            'augment_weight': float(self.aug_weight.value()),
            'selected_target': selected_target,
            'selected_features': selected_features,
        }

        self.thread = QThread()
        self.worker = AnalysisWorker(file_path, params)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_log)
        self.worker.progress_percent.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.analysis_finished)
        
        self.thread.start()

    def update_log(self, message):
        self.log_display.append(message)

    def analysis_finished(self, report_path, plot_path, si_dict):
        self.run_button.setEnabled(True)
        self.run_button.setText("开始分析")
        self.cancel_button.setEnabled(False)
        self.thread.quit()
        self.thread.wait()

        for widget in self._input_controls:
            widget.setEnabled(True)

        has_cols = self.target_combo.count() > 0
        self.target_combo.setEnabled(has_cols)
        self.features_list.setEnabled(has_cols)
        self.btn_select_all.setEnabled(has_cols)
        self.btn_clear_selection.setEnabled(has_cols)
        self.update_feature_selection_summary()
        
        if si_dict is not None:
            self.output_report_path = report_path
            self.output_plot_path = plot_path
            try:
                self.last_output_dir = os.path.dirname(report_path)
            except Exception:
                self.last_output_dir = ""
            self.export_report_button.setEnabled(True)
            self.export_plot_button.setEnabled(True)
            self.open_output_button.setEnabled(bool(self.last_output_dir))
            self.log_display.append("\n✅ 分析完成，可以导出结果。")
        else:
            self.log_display.append("\n❌ 分析失败，请检查日志中的错误信息。")
            self.open_output_button.setEnabled(False)

    def cancel_analysis(self):
        if self.worker is not None:
            self.worker.cancel()
            self.log_display.append("⏹️ 正在取消，请稍等...")
            self.cancel_button.setEnabled(False)
            self.run_button.setText("正在取消…")

    def export_report(self):
        default_name = os.path.join(self.last_output_dir or os.getcwd(), "sobol_analysis_report.txt")
        save_path, _ = QFileDialog.getSaveFileName(self, "保存报告", default_name, "Text Files (*.txt)")
        if save_path and self.output_report_path:
            try:
                import shutil
                shutil.copy(self.output_report_path, save_path)
                self.log_display.append(f"✅ 报告已成功保存到: {save_path}")
            except Exception as e:
                self.log_display.append(f"❌ 保存报告失败: {e}")

    def export_plot(self):
        default_name = os.path.join(self.last_output_dir or os.getcwd(), "sobol_sensitivity_analysis_gui.pdf")
        save_path, _ = QFileDialog.getSaveFileName(self, "保存图表", default_name, "PDF Files (*.pdf)")
        if save_path and self.output_plot_path:
            try:
                import shutil
                shutil.copy(self.output_plot_path, save_path)
                self.log_display.append(f"✅ 图表已成功保存到: {save_path}")
            except Exception as e:
                self.log_display.append(f"❌ 保存图表失败: {e}")

    def open_output_dir(self):
        if not self.last_output_dir or not os.path.isdir(self.last_output_dir):
            self.log_display.append("ℹ️ 尚未生成结果文件夹，或文件夹已被移动。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output_dir))
        self.log_display.append(f"📂 已打开结果文件夹: {self.last_output_dir}")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 3. 在主程序入口添加此行
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
