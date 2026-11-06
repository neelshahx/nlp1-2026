"""Baseline that ignores the input and predicts the training label distribution for every pair."""

import json
from pathlib import Path

import numpy as np

from ..data import LABELS
from ..representations import Features
from .base import Model


class MajorityModel(Model):
    kind = "majority"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._dist: np.ndarray | None = None

    def fit(self, features: Features, labels: np.ndarray) -> "MajorityModel":
        """Record how often each label occurs; ``features`` are unused."""
        counts = np.bincount(labels, minlength=len(LABELS)).astype(float)
        self._dist = counts / counts.sum()
        return self

    def predict_proba(self, features: Features) -> np.ndarray:
        """The same label distribution for every row."""
        return np.tile(self._dist, (len(features), 1))

    def save(self, path: Path) -> None:
        Path(str(path) + ".json").write_text(json.dumps({"seed": self.seed, "dist": self._dist.tolist()}))

    @classmethod
    def load(cls, path: Path) -> "MajorityModel":
        d = json.loads(Path(str(path) + ".json").read_text())
        m = cls(d["seed"])
        m._dist = np.array(d["dist"])
        return m
