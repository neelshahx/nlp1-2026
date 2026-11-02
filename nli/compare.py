"""Bayesian comparison of two Results with a ROPE (core; wraps ``baycomp``).

KNOWN ISSUES in the tutorial's ``BenavoliCorrelatedT``: it never passed ``runs`` when
several seeds were used, and built baycomp with ``rope=0`` then recomputed probabilities
itself. Here ``runs=len(seeds)`` always, and baycomp does the arithmetic.
"""
from __future__ import annotations

import numpy as np

from .store import Result


def _check_paired(a: Result, b: Result, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if a.folds_key != b.folds_key:
        raise ValueError(f"unpaired comparison: folds differ ({a.folds_key} vs {b.folds_key}); see ADR-0001")
    if a.seeds != b.seeds:
        raise ValueError(f"unpaired comparison: model seeds differ ({a.seeds} vs {b.seeds})")
    x, y = a.per_fold(metric), b.per_fold(metric)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError(f"score vectors differ in length or are empty ({len(x)} vs {len(y)})")
    return x, y


def compare(a: Result, b: Result, metric: str = "accuracy", rope: float = 0.01,
            plot: bool = False) -> dict:
    """Correlated Bayesian t-test (``baycomp.two_on_single``) between two Results on one Dataset.

    Returns ``{"left", "right", "metric", "rope", "runs", "p_left", "p_rope", "p_right"}``
    plus ``"fig"`` when ``plot`` is true. ``rope`` is in metric units (0.01 = one accuracy point).
    """
    import baycomp

    x, y = _check_paired(a, b, metric)
    runs = len(a.seeds)
    names = (a.id, b.id)
    out = baycomp.two_on_single(x, y, rope=rope, runs=runs, names=names, plot=plot)
    probs, fig = (out, None) if not plot else out
    p_left, p_rope, p_right = (probs[0], 0.0, probs[1]) if rope == 0 else probs
    res = {"left": a.id, "right": b.id, "metric": metric, "rope": rope, "runs": runs,
           "p_left": float(p_left), "p_rope": float(p_rope), "p_right": float(p_right)}
    if fig is not None:
        res["fig"] = fig
    return res


def compare_across_datasets(pairs: list[tuple[Result, Result]], metric: str = "accuracy",
                            rope: float = 0.01, plot: bool = False) -> dict:
    """Hierarchical test (``baycomp.two_on_multiple``): the same two systems on several Datasets.

    ``pairs`` is one ``(a, b)`` per Dataset, each paired as in ``compare``. All must share
    the same number of scores. Returns the same keys as ``compare`` with ``left``/``right``
    as lists of ids.
    """
    import baycomp

    xs, ys = zip(*[_check_paired(a, b, metric) for a, b in pairs])
    x, y = np.stack(xs), np.stack(ys)
    runs = len(pairs[0][0].seeds)
    out = baycomp.two_on_multiple(x, y, rope=rope, runs=runs, plot=plot)
    probs, fig = (out, None) if not plot else out
    p_left, p_rope, p_right = (probs[0], 0.0, probs[1]) if rope == 0 else probs
    res = {"left": [a.id for a, _ in pairs], "right": [b.id for _, b in pairs], "metric": metric,
           "rope": rope, "runs": runs, "p_left": float(p_left), "p_rope": float(p_rope), "p_right": float(p_right)}
    if fig is not None:
        res["fig"] = fig
    return res
