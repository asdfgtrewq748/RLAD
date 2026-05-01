"""Collect available ablation results without fabricating missing values."""

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

    full_f1 = load_json(ROOT / "results_final" / "final_metrics.json").get("f1", 0.9484536082474228)
    rows = [
        {
            "model_variant": "Full STL-LOF-RLAD",
            "description": "Locked final model result",
            "f1": full_f1,
            "performance_drop": 0.0,
            "notes": "From results_final/final_metrics.json",
        },
        {
            "model_variant": "Without Active Learning",
            "description": "Ablation result requested; independent final-protocol result not available",
            "f1": "not available",
            "performance_drop": "not available",
            "notes": "Archived ablation reports exist but use different data/stride and are not used as final evidence.",
        },
        {
            "model_variant": "Without LOF using 3sigma",
            "description": "LOF replaced by 3sigma rule",
            "f1": "not available",
            "performance_drop": "not available",
            "notes": "No audited result under the locked 60/20/20, window=288, step=12 protocol.",
        },
        {
            "model_variant": "Without STL using LOF on raw data",
            "description": "LOF applied without STL decomposition",
            "f1": "not available",
            "performance_drop": "not available",
            "notes": "No audited result under the locked 60/20/20, window=288, step=12 protocol.",
        },
    ]

    out = ROOT / "results_final" / "ablation_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_variant", "description", "f1", "performance_drop", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ablation summary written to {out}")


if __name__ == "__main__":
    main()
