"""Lock the final STL-LOF-RLAD paper result into results_final/.

This script intentionally does not retune or redesign the model. It copies the
existing reproducible JSON result, enriches it with audit fields, and writes the
derived confusion matrix.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import confusion_payload, enrich_final_metrics
from src.utils import load_json, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "final_config.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(int(config.get("seed", 42)))

    source = (ROOT / config["locked_result_source"]).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Locked final result JSON not found: {source}")

    metrics = enrich_final_metrics(load_json(source))
    metrics["timestamp"] = metrics.get("timestamp") or datetime.now().isoformat(timespec="seconds")

    results_dir = ROOT / "results_final"
    save_json(metrics, results_dir / "final_metrics.json")
    save_json(confusion_payload(metrics), results_dir / "confusion_matrix.json")

    history = {
        "status": "not available",
        "reason": "The locked final run directory did not include episode-level training history.",
        "source_run": str(source),
    }
    save_json(history, results_dir / "training_history.json")

    print(f"Locked final metrics written to {results_dir / 'final_metrics.json'}")
    print(f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
