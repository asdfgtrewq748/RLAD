"""Shared utilities for the locked STL-LOF-RLAD reproduction package."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # Allows metadata/figure generation before dependencies are installed.
    torch = None


def set_seed(seed: int = 42) -> None:
    """Set all random seeds used by the reproduction scripts."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
