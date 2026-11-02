"""Log-linear (multinomial logistic regression) classifier. OWNER: Neel.

Starting point: ``nlp1-student/part1/loglin.py:14`` ``fit`` (sklearn ``LogisticRegression``
with ``solver="saga"``, pickled). KNOWN ISSUES there: no ``random_state`` (saga is
stochastic); it re-encodes texts on every call. Here ``features.payload`` is already a CSR
matrix. Pass ``random_state=self.seed``. Hyperparameters: ``C``, ``penalty``, ``max_iter``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..representations import Features
from .base import Model


class LogLinearModel(Model):
    kind = "loglinear"

    def __init__(self, seed: int = 0, C: float = 1.0, penalty: str = "l2", max_iter: int = 200):
        super().__init__(seed)
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self._clf = None  # sklearn.linear_model.LogisticRegression

    def fit(self, features: Features, labels: np.ndarray) -> "LogLinearModel":
        raise NotImplementedError("Neel: LogisticRegression(..., random_state=self.seed).fit(features.payload, labels)")

    def predict_proba(self, features: Features) -> np.ndarray:
        raise NotImplementedError("Neel: self._clf.predict_proba(features.payload)")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Neel: pickle self._clf to path + '.pickle'")

    @classmethod
    def load(cls, path: Path) -> "LogLinearModel":
        raise NotImplementedError("Neel")
