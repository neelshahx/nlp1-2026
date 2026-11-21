"""Ensembles: a set of fitted Models plus a Combiner, scored on a held-out split (dev, or test once)."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS
from .metrics import score
from .store import Result, read_final_predictions, read_predictions

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


# --- across configs -----------------------------------------------------------


def _probs(frames: list) -> list[np.ndarray]:
    ids = frames[0]["id"].to_numpy()
    for df in frames[1:]:
        if not np.array_equal(df["id"].to_numpy(), ids):
            raise ValueError("ensemble members predicted different rows (ids differ); folds must be paired")
    return [df[[f"p_{n}" for n in LABELS]].to_numpy() for df in frames]


def _check_paired(results: list[Result]) -> None:
    keys = {r.folds_key for r in results}
    if len(keys) != 1:
        raise ValueError(f"results use different folds {keys}; a CV ensemble needs the same folds")


def cv_ensemble(
    results: list[Result], combiner: str = "mean_proba", root: Path | str = ".", seeds: list[int] | None = None
) -> pd.DataFrame:
    """Cross-validated score of an Ensemble over configs (and seeds): one row per (fold, metric).

    For held-out fold K the members are fold-K models of every Result and seed, all trained without fold K, so
    the score is a fair CV estimate paired fold by fold with the single models'. Long format with ``seed`` 0
    (one combined model per fold), so ``Result``/``per_fold`` read it like a single-seed run.
    """
    _check_paired(results)
    rows = []
    for fold in range(results[0].k):
        frames = [
            read_predictions(root, r.id, s, fold, "heldout")
            for r in results
            for s in (r.seeds if seeds is None else seeds)
        ]
        probs = combine(_probs(frames), combiner)
        for m, v in score(frames[0]["label"].to_numpy(), probs, tuple(results[0].config["metrics"])).items():
            rows.append({"seed": 0, "fold": fold, "metric": m, "value": v})
    return pd.DataFrame(rows)


def split_ensemble(
    results: list[Result],
    split: str,
    combiner: str = "mean_proba",
    root: Path | str = ".",
    seeds: list[int] | None = None,
) -> dict[str, float]:
    """Score every Fold model of every Result (all seeds by default), combined, on a split (dev; test only for the final evaluation)."""
    frames = [
        read_predictions(root, r.id, s, f, split)
        for r in results
        for s in (r.seeds if seeds is None else seeds)
        for f in range(r.k)
    ]
    probs = combine(_probs(frames), combiner)
    return score(frames[0]["label"].to_numpy(), probs, tuple(results[0].config["metrics"]))


def final_ensemble(
    results: list[Result],
    split: str,
    combiner: str = "mean_proba",
    root: Path | str = ".",
    seeds: list[int] | None = None,
) -> dict[str, float]:
    """Score the full-train refits (``experiment.run_final``) of every Result and seed, combined, on a split."""
    frames = [
        read_final_predictions(root, r.id, s, split) for r in results for s in (r.seeds if seeds is None else seeds)
    ]
    probs = combine(_probs(frames), combiner)
    return score(frames[0]["label"].to_numpy(), probs, tuple(results[0].config["metrics"]))
