"""Ensembles: a set of fitted Models plus a Combiner, scored on a held-out split (dev, or test once)."""

from pathlib import Path

import numpy as np

from .data import LABELS
from .metrics import score
from .store import Result, read_predictions

COMBINERS = ("mean_proba", "majority")


def combine(probs_list: list[np.ndarray], combiner: str) -> np.ndarray:
    """Merge member probability matrices into one ``(n, 3)`` matrix.

    ``mean_proba`` averages probabilities; ``majority`` returns vote fractions, so ties resolve to the lowest
    label index under argmax.
    """
    stack = np.stack(probs_list)
    if combiner == "mean_proba":
        return stack.mean(axis=0)
    if combiner == "majority":
        votes = stack.argmax(axis=2)
        counts = np.stack([(votes == c).sum(axis=0) for c in range(len(LABELS))], axis=1)
        return counts / counts.sum(axis=1, keepdims=True)
    raise ValueError(f"unknown combiner {combiner!r}; choose from {COMBINERS}")


def ensemble_on_split(
    result: Result,
    split: str,
    combiner: str = "mean_proba",
    root: Path | str = ".",
    seeds: list[int] | None = None,
) -> dict[str, float]:
    """Score the Ensemble of a Result's Fold models (all seeds by default) on a split.

    Reads stored predictions, so no model is reloaded: ``run_experiment`` writes dev,
    ``experiment.predict_split`` writes test.
    """
    seeds = result.seeds if seeds is None else seeds
    k = result.config["dataset"]["k"]
    frames = [read_predictions(root, result.id, s, f, split) for s in seeds for f in range(k)]
    probs = combine([df[[f"p_{n}" for n in LABELS]].to_numpy() for df in frames], combiner)
    return score(frames[0]["label"].to_numpy(), probs, tuple(result.config["metrics"]))


def ensemble_on_dev(
    result: Result, combiner: str = "mean_proba", root: Path | str = ".", seeds: list[int] | None = None
) -> dict[str, float]:
    return ensemble_on_split(result, "dev", combiner, root, seeds)
