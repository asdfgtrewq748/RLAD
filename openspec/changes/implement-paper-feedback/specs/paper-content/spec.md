## ADDED Requirements

### Requirement: Enhanced RLAD Positioning

The manuscript SHALL position the RLAD framework as "Enhanced RLAD" - a specialized solution for hydraulic support monitoring in coal mining, distinguishing it from existing RLAD research through two core innovations.

#### Scenario: Paper title and abstract differentiation

- **WHEN** a reader encounters the paper title and abstract
- **THEN** the title SHALL emphasize "Risk-Aware" and the application domain (automating mine safety)
- **AND** the abstract SHALL clearly state the two core innovations (ensemble bootstrapping and asymmetric reward function)
- **AND** the abstract SHALL follow the structure: problem → limitations → solution → results → impact

#### Scenario: Introduction contribution clarity

- **WHEN** a reviewer reads Section 1.4 (Contributions)
- **THEN** the section SHALL explicitly list the two core innovations
- **AND** the section SHALL include a comparison table (Table I) showing feature-by-feature differences with existing RLAD
- **AND** the table SHALL highlight Enhanced RLAD innovations in bold
- **AND** the comparison SHALL cover: supervision source, core algorithm, learning objective, and industrial adaptation

#### Scenario: Two-pillar innovation narrative

- **WHEN** any section of the paper describes the framework
- **THEN** the description SHALL reference one or both core innovations
- **AND** the innovations SHALL be mentioned in: abstract, introduction, methodology, experimental results, discussion, and conclusion
- **AND** each innovation SHALL be connected to solving specific industrial challenges (risk, safety, label scarcity)

### Requirement: Technical Detail Enhancement

The manuscript SHALL provide detailed technical explanations for the two core innovations, enabling reviewers and readers to understand their necessity and effectiveness.

#### Scenario: Ensemble method explanation

- **WHEN** Section 3.2.2 describes multi-dimensional ensemble bootstrapping
- **THEN** each of the four methods SHALL have a 1-2 sentence explanation of the anomaly type it detects:
  - Isolation Forest: structural/morphological anomalies
  - Z-score: magnitude-based anomalies
  - Variance Analysis: fluctuation/spike anomalies
  - Gradient Analysis: change-rate anomalies
- **AND** the section SHALL explain why integration is necessary for robustness
- **AND** the section SHALL reference literature supporting multi-method diversity

#### Scenario: Asymmetric reward function rationale

- **WHEN** Section 3.3.1 describes the MDP reward function
- **THEN** the section SHALL include a dedicated paragraph explaining the reward value rationale
- **AND** each reward value SHALL be justified with domain knowledge:
  - TP=+5.0: high reward for detecting risk (incentivizes exploration)
  - FN=-3.0: heavy penalty for missing anomalies (safety-first principle)
  - FP=-0.5: minor penalty for false alarms (nuisance but not catastrophic)
  - TN=+1.0: small reward for correct normal classification
- **AND** the section SHALL connect to industrial safety cost literature
- **AND** the section SHALL emphasize the "risk-averse" optimization direction

### Requirement: Industrial Impact Quantification

The manuscript SHALL quantify the industrial and economic impact of the Enhanced RLAD framework to demonstrate practical value beyond academic contributions.

#### Scenario: Economic value presentation

- **WHEN** Section 5.2 discusses economic value
- **THEN** the section SHALL include quantitative metrics from industry reports:
  - Predictive maintenance ROI: 10x return on investment
  - Downtime reduction: 70% reduction in unplanned downtime
  - Cost reduction: 25-30% reduction in total maintenance costs
- **AND** the section SHALL cite authoritative industry sources for these statistics
- **AND** the section SHALL connect these benefits to the mining domain specifically

#### Scenario: Market growth context

- **WHEN** Section 5.2 discusses market relevance
- **THEN** the section SHALL present market size and growth data:
  - Global predictive maintenance market: $10.6B (2024) → $47.8B (2029)
  - Digital mining services market: $150B by 2025
- **AND** the section SHALL position Enhanced RLAD as a key technology in this growth
- **AND** the section SHALL emphasize the strategic value for mining digital transformation

#### Scenario: Future work roadmap

- **WHEN** Section 5.3 describes limitations and future work
- **THEN** the section SHALL include Explainable AI (XAI) as a future direction
- **AND** the section SHALL mention specific techniques (LIME, SHAP) for generating explanations
- **AND** the section SHALL explain how XAI builds trust for industrial adoption
- **AND** the section SHALL connect to "trustworthy AI" (可信AI) trends
- **AND** the section SHALL reference key XAI survey papers

### Requirement: Conclusion Reinforcement

The manuscript SHALL conclude with a clear, impactful summary that reinforces the core contributions and industrial significance.

#### Scenario: Conclusion structure

- **WHEN** a reader reaches Section 6 (Conclusion)
- **THEN** the section SHALL restate the close-distance coal mining challenge
- **AND** the section SHALL explicitly name the two core innovations
- **AND** the section SHALL summarize the two key advantages: robustness (from ensemble) and safety (from asymmetric reward)
- **AND** the section SHALL highlight the practical advantage: no manual threshold tuning required for high recall
- **AND** the section SHALL end with a strong statement about contribution to safer, smarter mines
