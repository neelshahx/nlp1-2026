"""Scoring (core)."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import LABELS

METRICS = ("accuracy", "macro_f1")


def score(y_true: np.ndarray, probs: np.ndarray, metrics: tuple[str, ...] = METRICS) -> dict[str, float]:
    """Metric name -> value for one held-out fold (or the dev set)."""
    y_pred = probs.argmax(axis=1)
    out = {}
    for m in metrics:
        if m == "accuracy":
            out[m] = float(accuracy_score(y_true, y_pred))
        elif m == "macro_f1":
            out[m] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        else:
            raise ValueError(f"unknown metric {m!r}; choose from {METRICS}")
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """``(3, 3)`` confusion matrix in ``LABELS`` order (rows = gold, cols = predicted)."""
    return confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
