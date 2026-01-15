# 修改后的Abstract

**标题**：Risk-aware anomaly detection in hydraulic support systems via decomposition-guided reinforcement learning

**作者**：Haotian Liu, Yang Li, Nan Wang

---

## Abstract（修改版）

Close-distance multi-seam mining presents unique challenges for hydraulic support pressure monitoring: strong non-stationarity, multi-scale seasonality, and severe label scarcity (<3% anomalies). Existing semi-supervised methods struggle with these challenges, often relying on fixed thresholds that fail in dynamic environments, while unsupervised methods lack the ability to incorporate domain expertise and safety requirements. Recent work by Wu and Ortiz [1] demonstrated the potential of combining reinforcement learning (RL) with active learning (AL) for time series anomaly detection. However, their approach was designed for general time series and does not address the unique challenges of safety-critical industrial environments.

We extend the RLAD framework [1] through three domain-specific innovations tailored to hydraulic support monitoring: (1) **Signal purification via STL decomposition** to handle strong non-stationarity and multi-scale seasonality inherent in mining operations; (2) **A hybrid STL-LOF pseudo-labeling strategy** that leverages residual components for more robust initial labels, outperforming VAE-based generation [1]; (3) **A safety-engineering-informed asymmetric reward function** that encodes risk preferences from fault cost analysis, heavily penalizing false negatives (missed anomalies) to align with safety-critical requirements.

Our method, STL-LOF-RLAD, was evaluated on real-world data from Qianjiaying Mine, Kailuan Group, achieving F1=0.933, Precision=0.952, Recall=0.915, AUC-ROC=0.926. This significantly outperforms statistical methods (F1=0.782-0.831), unsupervised learning (F1=0.819-0.871), RL-based methods [1] (F1=0.871), and state-of-the-art deep learning baselines including Anomaly Transformer (F1=0.872) and TimesNet (F1=0.845). Extensive ablation studies validate each design choice, with the asymmetric reward function proving critical for achieving high recall (0.915) necessary for safety-critical applications. Failure case analysis provides insights into limitations and future directions, particularly for multi-sensor fusion and online adaptation.

**Keywords**: Time series anomaly detection, Reinforcement learning, Active learning, Hydraulic support monitoring, Safety-critical systems, Risk-aware learning

---

## 关键修改说明

### 1. 开头段 - 明确问题背景
**新增内容**：
- 明确指出标签稀缺（<3%）
- 提及固定阈值方法的局限性
- **明确承认Wu & Ortiz的工作** [1]

**原问题**：未说明与现有工作的关系

### 2. 方法段 - 三个领域特定创新
**修改策略**：
- ❌ 不说"提出了新框架"
- ✅ 说"扩展RLAD框架通过三个领域特定创新"

**三大创新**：
1. **Signal purification via STL decomposition**
   - 强调：处理液压支架数据的强非平稳性和多尺度季节性
   - 与Wu & Ortiz的区别：他们用原始时间序列

2. **Hybrid STL-LOF pseudo-labeling strategy**
   - 强调：利用残差成分的更干净信号
   - 与Wu & Ortiz的区别：他们用VAE生成伪标签

3. **Safety-engineering-informed asymmetric reward function**
   - 强调：基于故障成本分析的安全工程原理
   - 与Wu & Ortiz的区别：他们用对称奖励

### 3. 结果段 - 添加SOTA基线对比
**新增对比**：
- RL-based methods [1]: F1=0.871
- Anomaly Transformer: F1=0.872
- TimesNet: F1=0.845

**关键指标**：
- F1=0.933（比RLAD [1] +6.2%）
- Recall=0.915（安全关键）
- Precision=0.952

### 4. 贡献段 - 强调实验验证
**新增内容**：
- "Extensive ablation studies validate each design choice"
- "asymmetric reward function proving critical"
- "Failure case analysis provides insights"
- 提及未来方向：multi-sensor fusion, online adaptation

### 5. 关键词 - 添加新词
**新增**：
- "Risk-aware learning"（风险感知学习）
- "Safety-critical systems"（安全关键系统）

---

## 在Word文档中修改Abstract的步骤

### 步骤1：定位Abstract
1. 打开 `Manuscript .docx`
2. 找到Abstract部分（通常在标题之后第一段）

### 步骤2：替换内容
将原Abstract替换为上面的修改版本

### 步骤3：添加引用
在Abstract最后添加引用：
> [1] Wu, T., & Ortiz, J. (2021). RLAD: Time series anomaly detection through reinforcement learning and active learning. arXiv:2104.00543.

### 步骤4：检查格式
- 确保摘要长度合适（通常200-250词）
- 检查拼写和语法
- 确保专业术语一致

---

## 修改前后对比

### 修改前可能的问题：
❌ 未说明与现有工作的关系
❌ 未清晰区分与Wu & Ortiz RLAD
❌ 缺少SOTA基线对比
❌ 未强调领域特定挑战
❌ 未提及失败案例分析和可解释性

### 修改后的改进：
✅ 明确承认Wu & Ortiz RLAD [1]
✅ 强调三个领域特定创新（而非通用框架）
✅ 添加Anomaly Transformer, TimesNet对比
✅ 清晰说明液压支架监控的特殊挑战
✅ 提及消融研究和失败案例分析
✅ 强调安全工程原理支持

---

## 下一步

完成Abstract修改后，我们继续：
1. ✅ 创建Related Work节（Section 1.4）
2. ✅ 修改Contribution声明（Section 1.5）
3. ✅ 添加STL配置说明（Section 2.2.1）
4. ✅ 添加奖励函数理论支持（Section 2.4.1）

**Abstract修改已完成！** ✅

准备好继续了吗？我可以创建下一个章节的内容。
