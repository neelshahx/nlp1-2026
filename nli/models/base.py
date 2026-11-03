"""Model interface: sklearn-shaped, as in cs224u's ``TorchModelBase``."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..representations import Features


class Model(ABC):
    """A trainable classifier over Features.

    Rules (ADR-0001):
    - ``fit`` receives only the fold's training Features and labels. A Model that wants
      early stopping carves its own validation slice from what it was given
      (cs224u's ``validation_fraction`` pattern). It never sees the held-out fold.
    - Every source of randomness (init, shuffling, dropout, sklearn ``random_state``)
      must be driven by ``self.seed``. KNOWN ISSUE in the tutorial: torch was never seeded
      and ``LogisticRegression`` had no ``random_state``.
    - ``fit`` may be a no-op (majority baseline, rule-based or logic-based approaches).
    """

    kind: str  # the value of config["model"]["kind"]

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def fit(self, features: Features, labels: np.ndarray) -> "Model":
        """Train on the fold's training portion. Return self."""

    @abstractmethod
    def predict_proba(self, features: Features) -> np.ndarray:
        """``(n_rows, 3)`` probabilities in ``LABELS`` order; rows sum to 1."""

    def predict(self, features: Features) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write to ``path`` (no extension; add your own)."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Model":
        """Inverse of ``save``."""
