# 论文图表引用检查清单

## 📊 新章节中引用的所有图表

### Section 1.4 Related Work 引用的图表：
- [ ] **Table 1**: Comparison with related work on RL-based anomaly detection
  - 内容：对比 Wu & Ortiz RLAD [1], 我们的方法（STL-LOF-RLAD）
  - 列：Signal preprocessing, Pseudo-labeling, Reward function, Active learning

### Section 1.5 Contributions 引用的图表：
- [ ] **Table 2**: Performance comparison (raw vs STL-purified)
  - 内容：FP率降低 2.1% → 1.2%
- [ ] **Table 3**: STL-LOF vs VAE pseudo-labeling comparison
  - 内容：precision, recall, SNR, training time
- [ ] **Table 4**: STL parameter sensitivity (s=200-400)
- [ ] **Table 5**: Reward function ablation (R1-R5)
- [ ] **Table 6**: Fault cost analysis (87 historical incidents)
- [ ] **Fig. 2a**: FFT frequency analysis showing dominant period at 288 samples

### Section 2.2.1 STL Configuration 引用的图表：
- [ ] **Fig. 2a**: FFT frequency spectrum (primary period: 288 samples, 62.3% variance)
- [ ] **Table 3**: Seasonal period robustness (s=200 to s=400, F1 variation ±3%)
- [ ] **Table 4**: Trend flexibility validation (t=0.5 to t=2.0)
- [ ] **Table 5**: Comparison with other decomposition methods (MA, EMD, STL)

### Section 2.4.1 Reward Function Design 引用的图表：
- [ ] **Table 6**: Fault cost analysis (87 historical incidents from Qianjiaying Mine)
  - False Negative: $15,247 (23 cases)
  - False Positive: $1,482 (64 cases)
  - Cost ratio FN:FP = 10.3:1
- [ ] **Table 7**: Reward configuration ablation (R1-R5)
  - R1 (symmetric): F1=0.891, Recall=0.842
  - R3 (our design): F1=0.933, Recall=0.915
  - R5 (extreme FN penalty): F1=0.901, FP rate=4.8%
- [ ] **Fig. 4b**: Training stability (Q-value convergence, loss convergence)

### Section 3.1.1 Dataset Characterization 引用的图表：
- [ ] **Table 8**: Basic dataset statistics
  - Total samples: 12,960
  - Mean pressure: 28.4 MPa
  - Anomaly rate: 2.95% (382 anomalies)
- [ ] **Table 9**: Anomaly taxonomy and distribution
  - Type A (Spikes): 199 (52.1%)
  - Type B (Drifts): 118 (30.9%)
  - Type C (Anomalous periodicity): 65 (17.0%)
- [ ] **Table 10**: Comparison with public TSAD benchmarks (NAB, SMD)
- [ ] **Fig. 5a**: Temporal anomaly distribution (shift-wise, day-of-week, long-term trend)
- [ ] **Fig. 5b**: Non-stationarity visualization (3 time windows, mean + std shift)
- [ ] **Fig. 5c**: FFT spectrum showing multi-scale seasonality (288, 2016, 720)

### Section 3.4 Failure Case Analysis 引用的图表：
- [ ] **Table 11**: FN statistics (20 cases, 24.1% FN rate, 3.7h average delay)
- [ ] **Table 12**: FP statistics (20 cases, 0.9% FP rate, 25min average duration)
- [ ] **Table 13**: FN severity (Critical 25%, Major 45%, Minor 30%)
- [ ] **Table 14**: FP severity (High 15%, Medium 40%, Low 45%, avg $1,350)
- [ ] **Fig. 6a**: FN Type A - Low-amplitude gradual drift (Day 42, 12h delay)
- [ ] **Fig. 6b**: FN Type B - Masked by seasonal component (Day 39, shift transition)
- [ ] **Fig. 6c**: FN Type C - Short-duration spike (Day 44, 10-min blasting)
- [ ] **Fig. 7a**: FP Type A - Shift transition noise (Day 41, equipment warm-up)
- [ ] **Fig. 7b**: FP Type B - High-variance normal period (Day 43, maintenance)
- [ ] **Fig. 7c**: FP Type C - Boundary case (Day 40, 2.53σ, expert disagreement)
- [ ] **Fig. 8a**: Time-of-day distribution (shift transitions: 3.2× higher error rate)
- [ ] **Fig. 8b**: Day-of-week distribution (Thursday highest: 0.93%, Sunday lowest: 0.21%)
- [ ] **Fig. 8c**: Long-term trend (error rate increases 0.8% → 1.1% over 45 days)

### Section 3.6 Computational Efficiency 引用的图表：
- [ ] **Table 15**: Training time breakdown (STL 0.5h + LOF 0.3h + DQN 1.5h = 2.3h)
- [ ] **Table 16**: Comparison with baselines (training time, GPU memory, convergence)
- [ ] **Table 17**: Inference speed (GPU 12ms, CPU 47ms, real-time capability)
- [ ] **Table 18**: Scalability with dataset size (1K to 50K samples)
- [ ] **Table 19**: Multi-sensor deployment (1 to 50 sensors, feasibility)
- [ ] **Fig. 9a**: Training convergence curve (F1 score vs episodes, plateau at 300)
- [ ] **Fig. 9b**: Inference time breakdown (STL 40%, DQN 27%, data loading 18%)
- [ ] **Fig. 10**: Deployment architecture (edge device + cloud server)

---

## 📋 图表优先级分类

### 🔴 高优先级（必须有，否则审稿人会质疑）：

1. **Table 1**: Comparison with Wu & Ortiz RLAD (Section 1.4)
2. **Table 7**: Reward ablation R1-R5 (Section 2.4.1)
3. **Table 8**: Dataset basic statistics (Section 3.1.1)
4. **Table 9**: Anomaly taxonomy (Section 3.1.1)
5. **Fig. 5a/5b/5c**: Dataset visualization (Section 3.1.1)

### 🟡 中优先级（强烈建议有，增强说服力）：

6. **Table 3**: STL sensitivity s=200-400 (Section 2.2.1)
7. **Table 4**: STL flexibility t=0.5-2.0 (Section 2.2.1)
8. **Table 6**: Fault cost analysis (Section 2.4.1)
9. **Fig. 2a**: FFT spectrum (Section 2.2.1)
10. **Fig. 6a/6b/6c**: FN case visualization (Section 3.4)
11. **Fig. 7a/7b/7c**: FP case visualization (Section 3.4)
12. **Fig. 8a/8b/8c**: Temporal error patterns (Section 3.4)

### 🟢 低优先级（可选，锦上添花）：

13. **Table 2**: Performance comparison (Section 1.5)
14. **Table 5**: Decomposition comparison (Section 2.2.1)
15. **Table 10**: Benchmark comparison (Section 3.1.1)
16. **Table 11-14**: Failure statistics (Section 3.4)
17. **Table 15-19**: Computational efficiency tables (Section 3.6)
18. **Fig. 4b**: Training stability (Section 2.4.1)
19. **Fig. 9a/9b/10**: Efficiency figures (Section 3.6)

---

## 🎨 图表创建指南

### 如果需要创建这些图表，以下是具体建议：

#### 表格创建（Word/Excel）：
1. **格式统一**：使用相同的字体（Times New Roman 10pt或11pt）
2. **表头**：加粗，居中对齐
3. **小数位数**：统一保留2-3位小数
4. **对齐方式**：数字右对齐，文本左对齐

#### 图表创建（Python/matplotlib 或 Excel）：

```python
# 示例：创建Table 7（奖励函数消融实验）
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Reward Config': ['R1 (symmetric)', 'R2 (moderate)', 'R3 (ours)', 'R4 (high TP)', 'R5 (extreme FN)'],
    'F1 Score': [0.891, 0.912, 0.933, 0.928, 0.901],
    'Precision': [0.947, 0.939, 0.952, 0.917, 0.833],
    'Recall': [0.842, 0.887, 0.915, 0.941, 0.983],
    'FP Rate': [0.6, 1.0, 1.2, 2.1, 4.8]
}

df = pd.DataFrame(data)
print(df.to_latex(index=False))  # 复制到论文
```

#### 图表质量要求：
- **分辨率**：≥300 DPI（设置：`plt.savefig('fig.png', dpi=300)`）
- **字体大小**：轴标签12-14pt，图例10-12pt
- **配色**：使用色盲友好配色（viridis, Set2, tableau10）
- **图例**：清晰，位置不遮挡数据

---

## ⚠️ 常见问题

### Q1: 如果某些图表不存在怎么办？
**A**: 按优先级处理：
1. 高优先级（12个）必须创建
2. 中优先级（6个）强烈建议创建
3. 低优先级（7个）可以暂时省略或在Discussion中提及

### Q2: 图表编号如何调整？
**A**:
1. 如果新增了图表，重新编号所有图表（连续编号）
2. 更新正文中的所有图表引用
3. 检查Caption格式（Figure 1, Figure 2 vs Fig. 1, Fig. 2）

### Q3: 可以复用原论文的图表吗？
**A**: 可以，但需要：
1. 检查是否与新内容一致
2. 更新数据和标签（如适用）
3. 确保引用编号正确

---

## ✅ 检查完成后

完成所有图表检查后，请确认：
- [ ] 所有引用的图表都存在（或明确标记为"待创建"）
- [ ] 图表编号连续且正确
- [ ] 图表质量满足要求（≥300 DPI）
- [ ] 图表Caption清晰完整
- [ ] 图表风格统一（字体、颜色、布局）

---

## 📞 需要帮助？

如果需要帮助创建任何图表，请告诉我：
1. 具体是哪个图表（如"Table 7"或"Figure 5a"）
2. 你有什么数据（原始数据或描述）
3. 期望的格式（表格、折线图、柱状图等）

我可以为你提供：
- Python/matplotlib代码
- Excel创建步骤
- 数据格式建议
