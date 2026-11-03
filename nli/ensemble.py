"""Ensembles: a set of fitted Models plus a Combiner, scored on the dev set."""

from pathlib import Path

import numpy as np

from .data import LABELS
from .metrics import score
from .store import Result, read_predictions

COMBINERS = ("mean_proba", "majority")


def combine(probs_list: list[np.ndarray], combiner: str) -> np.ndarray:
    """Merge member probability matrices into one ``(n, 3)`` matrix (core, pure).

    ``majority`` returns vote fractions; ties resolve to the lowest label index via argmax.
    KNOWN ISSUE in the tutorial: ``MajorityVoteClassifier`` broke ties randomly with its own seed.
    """
    stack = np.stack(probs_list)  # (members, n, 3)
    if combiner == "mean_proba":
        return stack.mean(axis=0)
    if combiner == "majority":
        votes = stack.argmax(axis=2)  # (members, n)
        counts = np.stack([(votes == c).sum(axis=0) for c in range(len(LABELS))], axis=1)
        return counts / counts.sum(axis=1, keepdims=True)
    raise ValueError(f"unknown combiner {combiner!r}; choose from {COMBINERS}")


def ensemble_on_dev(
    result: Result, combiner: str = "mean_proba", root: Path | str = ".", seeds: list[int] | None = None
) -> dict[str, float]:
    """Score the Ensemble of a Result's Fold models (all seeds by default) on the dev set.

    Reads the dev predictions ``run_experiment`` stored, so no model is reloaded.
    Heterogeneous ensembles (members from different Experiments) are not built yet; when
    needed, read several Results' dev predictions and call ``combine`` on them.
    """
    seeds = result.seeds if seeds is None else seeds
    k = result.config["dataset"]["k"]
    frames = [read_predictions(root, result.id, s, f, "dev") for s in seeds for f in range(k)]
    probs = combine([df[[f"p_{n}" for n in LABELS]].to_numpy() for df in frames], combiner)
    return score(frames[0]["label"].to_numpy(), probs, tuple(result.config["metrics"]))
