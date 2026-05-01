"""Collect available baseline results without inventing missing experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "final_config.json"))
    args = parser.parse_args()
    config = load_json(args.config)
    set_seed(int(config.get("seed", 42)))

    comparison = ROOT.parent / "output_rlad_v3_optimized_20250716_152736" / "comparison_results" / "comparison_metrics.json"
    rows = []
    if comparison.exists():
        data = load_json(comparison)
        name_map = {
            "STL+3σ": "STL + 3sigma",
            "Autoencoder": "Autoencoder",
            "Isolation Forest": "Isolation Forest",
            "STL-LOF": "STL-LOF standalone",
            "STL-LOF-RLAD": "STL-LOF-RLAD",
        }
        for source_name, model_name in name_map.items():
            item = data.get(source_name, {})
            rows.append({
                "model": model_name,
                "precision": item.get("Precision", "not available"),
                "recall": item.get("Recall", "not available"),
                "f1": item.get("F1-Score", "not available"),
                "auc_roc": item.get("AUC-ROC", "not available"),
                "average_precision": "not available",
                "notes": f"Collected from {comparison.name}; AP not reported in source.",
            })
    else:
        rows.extend([
            {"model": name, "precision": "not available", "recall": "not available", "f1": "not available", "auc_roc": "not available", "average_precision": "not available", "notes": "No reproducible source result found."}
            for name in ["STL + 3sigma", "Autoencoder", "Isolation Forest", "STL-LOF standalone", "STL-LOF-RLAD"]
        ])

    rows.append({
        "model": "Original Ensemble-RLAD",
        "precision": "not available",
        "recall": "not available",
        "f1": "not available",
        "auc_roc": "not available",
        "average_precision": "not available",
        "notes": "Archived script exists, but no audited JSON matching the final protocol was found.",
    })

    out = ROOT / "results_final" / "baseline_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "precision", "recall", "f1", "auc_roc", "average_precision", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Baseline summary written to {out}")


if __name__ == "__main__":
    main()
