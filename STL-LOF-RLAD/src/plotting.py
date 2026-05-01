"""Publication figure generation from locked result artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_curve


STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "axes.unicode_minus": False,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _unavailable(path: Path, title: str, reason: str):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.axis("off")
    ax.text(0.02, 0.72, title, fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.48, "Source data not available in results_final.", transform=ax.transAxes)
    ax.text(0.02, 0.30, reason, transform=ax.transAxes)
    _save(fig, path)


def generate_all(results_dir: str | Path, output_dir: str | Path):
    _apply_style()
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    metrics = pd.read_json(results_dir / "final_metrics.json", typ="series").to_dict()

    labels = np.asarray(metrics.get("labels", []), dtype=int)
    preds = np.asarray(metrics.get("predictions", metrics.get("all_predictions", [])), dtype=int)
    probs = np.asarray(metrics.get("probabilities", metrics.get("all_probabilities", [])), dtype=float)

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    cm = np.array([[metrics.get("TN", 0), metrics.get("FP", 0)], [metrics.get("FN", 0), metrics.get("TP", 0)]])
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    ax.set_xticks([0, 1], ["Normal", "Hazardous"])
    ax.set_yticks([0, 1], ["Normal", "Hazardous"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Reference label")
    ax.set_title("Pseudo-label and final decision consistency")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, output_dir / "fig7_pseudo_label_quality.tif")

    history_path = results_dir / "training_history.json"
    if history_path.exists() and history_path.stat().st_size > 2:
        history_payload = pd.read_json(history_path, typ="series").to_dict()
        history = pd.DataFrame(history_payload if isinstance(history_payload, list) else [])
        if history.empty:
            _unavailable(output_dir / "fig8_training_dynamics.tif", "Training dynamics", "Archived final run did not include episode-level history.")
        else:
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            for col in history.columns:
                if pd.api.types.is_numeric_dtype(history[col]):
                    ax.plot(history.index, history[col], label=col)
            ax.set_xlabel("Epoch / episode")
            ax.set_ylabel("Value")
            ax.set_title("Training dynamics")
            ax.legend(frameon=False)
            _save(fig, output_dir / "fig8_training_dynamics.tif")
    else:
        _unavailable(output_dir / "fig8_training_dynamics.tif", "Training dynamics", "Archived final run did not include episode-level history.")

    values = [metrics.get("precision"), metrics.get("recall"), metrics.get("f1"), metrics.get("auc_roc"), metrics.get("average_precision")]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(["Precision", "Recall", "F1", "AUC-ROC", "AP"], values, color=["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#76B7B2"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Final test performance")
    _save(fig, output_dir / "fig9_final_metrics.tif")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    if labels.size and probs.size:
        ax.hist(probs[labels == 0], bins=18, alpha=0.75, label="Normal", color="#4C78A8")
        ax.hist(probs[labels == 1], bins=18, alpha=0.75, label="Hazardous", color="#E15759")
        ax.axvline(metrics.get("threshold", 0.5), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted risk score")
    ax.set_ylabel("Window count")
    ax.set_title("Risk score distribution")
    ax.legend(frameon=False)
    _save(fig, output_dir / "fig10_score_distribution.tif")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fpr, tpr, _ = roc_curve(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)
    axes[0].plot(fpr, tpr, color="#4C78A8", label=f"AUC = {auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(frameon=False)
    axes[1].plot(rec, prec, color="#E15759", label=f"AP = {metrics.get('average_precision', np.nan):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(frameon=False)
    _save(fig, output_dir / "fig11_roc_pr.tif")

    features = np.asarray(metrics.get("features", []), dtype=float)
    if features.ndim == 2 and features.shape[0] >= 5:
        perplexity = max(2, min(30, features.shape[0] // 3))
        emb = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(features)
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1], s=16, label="Normal", alpha=0.8)
        ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1], s=16, label="Hazardous", alpha=0.8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Feature embedding")
        ax.legend(frameon=False)
        _save(fig, output_dir / "fig12_tsne.tif")
    else:
        _unavailable(output_dir / "fig12_tsne.tif", "Feature embedding", "Feature vectors were not found in final_metrics.json.")

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(np.arange(len(probs)), probs, color="#4C78A8", linewidth=1.5, label="Risk score")
    ax.scatter(np.where(labels == 1)[0], probs[labels == 1], color="#E15759", s=20, label="Hazardous windows")
    ax.axhline(metrics.get("threshold", 0.5), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Predicted risk score")
    ax.set_title("Case-study warning sequence")
    ax.legend(frameon=False)
    _save(fig, output_dir / "fig13_case_study.tif")

    sensitivity_path = results_dir / "sensitivity_results.csv"
    if sensitivity_path.exists():
        sens = pd.read_csv(sensitivity_path)
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.plot(sens.iloc[:, 0], sens.iloc[:, 1], marker="o")
        ax.set_xlabel(sens.columns[0])
        ax.set_ylabel(sens.columns[1])
        ax.set_title("Parameter sensitivity")
        _save(fig, output_dir / "fig14_sensitivity.tif")
    else:
        _unavailable(output_dir / "fig14_sensitivity.tif", "Parameter sensitivity", "No sensitivity_results.csv was available.")
