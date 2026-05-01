"""Leakage-aware data processing utilities.

The locked paper result is copied from an existing JSON file by the main
reproduction script. These helpers document and support the intended data
protocol for future reruns: chronological split first, then train-only fitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def chronological_split_indices(n_samples: int, train_ratio: float = 0.6, val_ratio: float = 0.2):
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    return np.arange(train_end), np.arange(train_end, val_end), np.arange(val_end, n_samples)


def fit_train_only_zscore(X_train, X_val=None, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    outputs = [X_train_scaled]
    if X_val is not None:
        outputs.append(scaler.transform(X_val))
    if X_test is not None:
        outputs.append(scaler.transform(X_test))
    outputs.append(scaler)
    return tuple(outputs)
