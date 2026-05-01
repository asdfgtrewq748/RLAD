"""STL-LOF pseudo-label utilities with train-only threshold estimation."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL


def stl_residual(series, period: int = 288, robust: bool = True):
    return STL(np.asarray(series, dtype=float), period=period, robust=robust).fit().resid


def train_lof_threshold(train_residual, n_neighbors: int = 20, sigma: float = 3.0):
    train_residual = np.asarray(train_residual, dtype=float).reshape(-1, 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit_predict(train_residual)
    scores = -lof.negative_outlier_factor_
    threshold = float(scores.mean() + sigma * scores.std())
    return threshold, scores
