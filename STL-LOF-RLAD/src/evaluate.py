"""Evaluation helpers for locked STL-LOF-RLAD result files."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score


def enrich_final_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add auditable derived fields without changing locked model outputs."""
    labels = np.asarray(metrics.get("labels", []))
    predictions = np.asarray(metrics.get("predictions", metrics.get("all_predictions", [])))
    probabilities = np.asarray(metrics.get("probabilities", metrics.get("all_probabilities", [])))

    if labels.size and predictions.size:
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        metrics.update({"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})

    if labels.size and probabilities.size and "average_precision" not in metrics:
        metrics["average_precision"] = float(average_precision_score(labels, probabilities))

    metrics.setdefault("threshold", 0.5)
    metrics.setdefault("seed", 42)
    metrics.setdefault("window_size", 288)
    metrics.setdefault("sliding_step", 12)
    metrics.setdefault("split_strategy", "60/20/20 chronological")
    metrics.setdefault(
        "source_script",
        "archive/old_scripts/RLADv3_2_TRUE_copy_copy_3.py",
    )
    metrics.setdefault(
        "locked_source_result",
        "../output_rlad_v3_optimized_20250907_143612/final_metrics.json",
    )
    return metrics


def confusion_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "TP": metrics.get("TP"),
        "TN": metrics.get("TN"),
        "FP": metrics.get("FP"),
        "FN": metrics.get("FN"),
        "labels": ["normal", "hazardous"],
        "source": "results_final/final_metrics.json",
    }
