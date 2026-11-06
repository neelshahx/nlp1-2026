"""Log-linear classifier: multinomial logistic regression over sparse Features. OWNER: Neel."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..data import LABELS
from ..representations import Features
from .base import Model


class LogLinearModel(Model):
    kind = "loglinear"

    def __init__(self, seed: int = 0, C: float = 1.0, l1_ratio: float = 0.0, max_iter: int = 200):
        """``C`` is inverse regularisation strength; ``l1_ratio`` 0 = l2, 1 = l1, between = elastic net."""
        super().__init__(seed)
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self._clf = None  # sklearn.linear_model.LogisticRegression

    def fit(self, features: Features, labels: np.ndarray) -> "LogLinearModel":
        """Fit sklearn ``LogisticRegression`` on the sparse payload; the solver is stochastic, so pass the seed."""
        self._clf = LogisticRegression(
            solver="saga", l1_ratio=self.l1_ratio, C=self.C, max_iter=self.max_iter, random_state=self.seed
        )
        self._clf.fit(features.payload, labels)
        return self

    def predict_proba(self, features: Features) -> np.ndarray:
        """Class probabilities from the fitted classifier."""
        probs = self._clf.predict_proba(features.payload)
        out = np.zeros((len(features), len(LABELS)))  # guards against dropped label
        out[:, self._clf.classes_] = probs
        return out

    def save(self, path: Path) -> None:
        Path(str(path) + ".pickle").write_bytes(pickle.dumps(self._clf))

    @classmethod
    def load(cls, path: Path) -> "LogLinearModel":
        clf = pickle.loads(Path(str(path) + ".pickle").read_bytes())
        model = cls(seed=clf.random_state, C=clf.C, l1_ratio=clf.l1_ratio, max_iter=clf.max_iter)
        model._clf = clf
        return model
