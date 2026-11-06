"""Cross-validation folds (core)."""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_folds(labels: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold over ``labels``. Returns ``[(train_idx, heldout_idx), ...]``.

    Deterministic in ``seed`` (the data seed), so every Experiment on a Dataset shares the same folds.
    """
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr, ho) for tr, ho in splitter.split(np.zeros(len(labels)), labels)]
