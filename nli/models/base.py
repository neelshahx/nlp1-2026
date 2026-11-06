"""Model interface: Features and labels in, class probabilities out."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..representations import Features


class Model(ABC):
    """A classifier over Features whose every source of randomness is driven by ``self.seed``."""

    kind: str  # the value of config["model"]["kind"]

    def __init__(self, seed: int = 0):
        """``seed`` must control init, shuffling, dropout, ``random_state`` — anything random."""
        self.seed = seed

    @abstractmethod
    def fit(self, features: Features, labels: np.ndarray) -> "Model":
        """Train on the fold's training rows only; carve any validation slice from these. Return self."""

    @abstractmethod
    def predict_proba(self, features: Features) -> np.ndarray:
        """``(n_rows, 3)`` probabilities in ``LABELS`` order; rows sum to 1."""

    def predict(self, features: Features) -> np.ndarray:
        """Most probable label per row."""
        return self.predict_proba(features).argmax(axis=1)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write to ``path`` (no extension; add your own)."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Model":
        """Inverse of ``save``."""
