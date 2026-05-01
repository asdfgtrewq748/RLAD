# STL-LOF-RLAD for Early Warning of Hazardous Hydraulic-Support Loading

## Overview

This repository reproduces the locked STL-LOF-RLAD framework for early warning of roof-weighting-induced hazardous hydraulic-support loading in close-distance multi-seam mining. The method combines STL decomposition, LOF-based pseudo-labeling, and risk-aware reinforcement learning to identify precursor windows associated with periodic roof weighting.

The purpose of this cleaned repository is auditability for manuscript submission to *Mining, Metallurgy & Exploration*: the final result is locked, archived scripts are preserved, and paper metrics and figures are traceable to `results_final/` and `figures_final/`.

## Repository Structure

```text
STL-LOF-RLAD/
  README.md
  requirements.txt
  config/final_config.json
  src/
    data_processing.py
    stl_lof_labeling.py
    model.py
    train.py
    evaluate.py
    plotting.py
    utils.py
  scripts/
    run_main_experiment.py
    run_baselines.py
    run_ablation.py
    generate_figures.py
  results_final/
    final_metrics.json
    baseline_results.csv
    ablation_results.csv
    confusion_matrix.json
    training_history.json
  figures_final/
  archive/old_scripts/
```

## Environment

Recommended: Python 3.10 or newer.

Install dependencies:

```bash
pip install -r requirements.txt
```

The core dependencies are `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `torch`, and `tqdm`. Figure generation uses the non-GUI Matplotlib `Agg` backend.

## Reproducibility

- Seed: `42`
- Window size: `288`
- Sliding step: `12`
- Split strategy: chronological `60% train / 20% validation / 20% test`
- Normalization: z-score parameters must be fitted only on the training set
- Final locked metrics: `results_final/final_metrics.json`
- Raw mine data may not be publicly available due to confidentiality restrictions

To avoid information leakage, chronological splitting was performed before normalization and threshold determination. Normalization parameters and LOF threshold were estimated only from the training set.

Audit note: the archived historical script is preserved under `archive/old_scripts/` and should not be treated as the clean reviewer entry point. It contains GUI/TkAgg dependencies and historical code paths where `StandardScaler.fit_transform` appears before splitting. The clean structure documents the leakage-safe protocol in `src/data_processing.py` and `src/stl_lof_labeling.py`; the main script locks the existing reproducible JSON rather than retuning or rerunning old exploratory code.

## How To Run

Main locked result:

```bash
python scripts/run_main_experiment.py --config config/final_config.json
```

Baseline summary:

```bash
python scripts/run_baselines.py --config config/final_config.json
```

Ablation summary:

```bash
python scripts/run_ablation.py --config config/final_config.json
```

Paper figures:

```bash
python scripts/generate_figures.py --results_dir results_final --output_dir figures_final
```

The GUI annotation components from older scripts are optional historical tools and are not required for reproducing the main result.

## Expected Final Performance

The locked final values are stored in `results_final/final_metrics.json` and should be treated as authoritative. Rounded manuscript values are:

| Metric | Value |
| --- | ---: |
| Precision | 0.939 |
| Recall | 0.958 |
| F1-score | 0.948 |
| AUC-ROC | 0.986 |
| Average precision | 0.983 |
| FN | 2 |

The derived confusion matrix is stored in `results_final/confusion_matrix.json`.

## Data Availability

Due to confidentiality restrictions associated with field monitoring data from an underground coal mine, the raw data are not publicly released. Processed data or additional information may be made available from the corresponding author upon reasonable request.

## Audit Trail

- `results_final/final_metrics.json` is generated from the existing locked source: `../output_rlad_v3_optimized_20250907_143612/final_metrics.json`.
- `archive/old_scripts/` preserves the historical main, baseline, ablation, and visualization scripts.
- `results_final/baseline_results.csv` records only available baseline metrics. Missing AP values remain marked as `not available`.
- `results_final/ablation_results.csv` does not invent unavailable final-protocol ablation results.
- `figures_final/` contains fixed-name `.tif` files at 600 dpi. Figures that lack required intermediate data are rendered as explicit unavailable-data audit panels and are also noted in `TODO.md`.
