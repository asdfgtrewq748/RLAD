# STL-LOF-RLAD hydraulic-support loading warning

This repository contains the code and manuscript-supporting materials for the study on early identification of roof-weighting-related hazardous hydraulic-support loading in close-distance multi-seam mining.

## Recommended manuscript target

The current submission strategy is to revise the manuscript for **Mining, Metallurgy & Exploration**. The code package should therefore emphasize engineering reproducibility, field-data traceability, and a clear link between model outputs and mining-pressure events, rather than presenting the project as a general-purpose artificial-intelligence benchmark.

## Canonical scripts

Use the following files as the current canonical implementation and supporting experiment scripts:

- `RLADv3.2(TRUE).py` — main STL-LOF-RLAD implementation used for the manuscript results.
- `对比实验（STL + 3σ）.py` — STL + 3-sigma comparison.
- `对比实验（STL+LOF）.py` — STL-LOF standalone comparison.
- `对比实验（Isolation Forest）.py` — Isolation Forest baseline.
- `对比实验 （Autoencoder ）.py` — Autoencoder baseline.
- `对比实验（Original Ensemble-RLAD）.py` — original RLAD / ensemble RLAD baseline.
- `消融实验.py` or `消融实验_简化版.py` — component ablation experiments.
- `generate_visualizations.py` — manuscript figure generation.

Older files such as `RLADv2.*`, `RLADv3.0.py`, `RLADv3.1.py`, copy files, and exploratory GUI scripts should be treated as development history only unless a result in the manuscript explicitly depends on them.

## Suggested repository structure for submission

Before sharing the repository with reviewers, keep the root directory simple:

```text
RLAD/
├── README.md
├── requirements.txt
├── SUBMISSION_TODO_MME.md
├── CODE_AUDIT_FOR_MME.md
├── RLADv3.2(TRUE).py
├── generate_visualizations.py
├── 对比实验（STL + 3σ）.py
├── 对比实验（STL+LOF）.py
├── 对比实验（Isolation Forest）.py
├── 对比实验 （Autoencoder ）.py
├── 对比实验（Original Ensemble-RLAD）.py
├── 消融实验.py
├── data/
│   ├── README.md
│   └── sample_or_anonymized_data.csv
├── results/
│   └── mme_submission/
└── figures/
```

Large raw mine-monitoring data, trained model checkpoints, and private production logs should not be committed unless they are anonymized and approved for release.

## Environment

Create a clean Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for full model training, but CPU execution should still be sufficient for data preprocessing and figure regeneration.

## Reproducibility notes

For the revised manuscript, record the following information in `results/mme_submission/` after each final run:

1. Dataset identifier or anonymized file name.
2. Support number or support group used for each experiment.
3. Sampling interval and time span.
4. Train / validation / test chronological split.
5. Window length and sliding step.
6. STL period, LOF neighbor number, contamination or threshold rule.
7. Reward setting used in the DQN detector.
8. Random seed.
9. Final precision, recall, F1-score, AUC-ROC, and confusion matrix.
10. Warning lead time for each engineering event, if available.

## Submission caution

Do not claim that every result is fully reproducible from this repository unless the raw or anonymized data, exact parameters, and output logs are included. For journal submission, a safe data-availability statement is:

> The field monitoring data used in this study contain mine-production information and are available from the corresponding author upon reasonable request, subject to approval by the mine operator. The source code and parameter settings used for the experiments can be provided for academic review.
