"""Generate fixed-name manuscript figures from results_final/ artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import generate_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(ROOT / "results_final"))
    parser.add_argument("--output_dir", default=str(ROOT / "figures_final"))
    args = parser.parse_args()
    generate_all(args.results_dir, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
