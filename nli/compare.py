"""Bayesian comparison of two Results with a region of practical equivalence (ROPE); wraps ``baycomp``."""

import numpy as np
import pandas as pd

from .store import Result


def restrict_seeds(result: Result, seeds: list[int]) -> Result:
    """The same Result with only the scores of ``seeds``, so it can be paired with a Result run on fewer seeds."""
    missing = set(seeds) - set(result.seeds)
    if missing:
        raise ValueError(f"{result.id} has no seeds {sorted(missing)}")
    config = {**result.config, "model_seeds": list(seeds)}
    return Result(result.id, config, result.scores[result.scores["seed"].isin(seeds)].reset_index(drop=True))


def seed_fold_spread(result: Result, metric: str = "accuracy") -> dict:
    """Split a Result's run-to-run spread into a fold part and a seed part (needs >= 2 seeds).

    ``sd_seed``: typical sd between seeds on the same fold, sqrt(mean over folds of the between-seed variance),
    i.e. training randomness with the data held fixed. ``sd_fold``: sd over folds of the seed-averaged score,
    i.e. what changing the train/held-out split does. ``max_seed_gap``: the largest seed-to-seed difference on
    any one fold.
    """
    df = result.scores[result.scores["metric"] == metric]
    table = df.pivot(index="fold", columns="seed", values="value")  # (fold, seed)
    if table.shape[1] < 2:
        raise ValueError(f"{result.id} has one seed; the seed part of the spread needs at least two")
    return {
        "id": result.id,
        "seeds": table.shape[1],
        "mean": float(table.to_numpy().mean()),
        "sd_fold": float(table.mean(axis=1).std(ddof=1)),
        "sd_seed": float(np.sqrt(table.var(axis=1, ddof=1).mean())),
        "max_seed_gap": float((table.max(axis=1) - table.min(axis=1)).max()),
    }


def compare_table(pairs: list[tuple[Result, Result]], metric: str = "accuracy", rope: float = 0.01) -> pd.DataFrame:
    """``compare`` for each ``(a, b)`` pair, one row each, with the mean difference ``a - b`` added."""
    rows = []
    for a, b in pairs:
        out = compare(a, b, metric=metric, rope=rope)
        rows.append({**out, "mean_diff": a.mean(metric) - b.mean(metric)})
    return pd.DataFrame(rows)


def _check_paired(a: Result, b: Result, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if a.folds_key != b.folds_key:
        raise ValueError(f"unpaired comparison: folds differ ({a.folds_key} vs {b.folds_key})")
    if a.seeds != b.seeds:
        raise ValueError(f"unpaired comparison: model seeds differ ({a.seeds} vs {b.seeds})")
    x, y = a.per_fold(metric), b.per_fold(metric)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError(f"score vectors differ in length or are empty ({len(x)} vs {len(y)})")
    return x, y


def compare(a: Result, b: Result, metric: str = "accuracy", rope: float = 0.01, plot: bool = False) -> dict:
    """Correlated Bayesian t-test between two Results on the same folds; ``runs`` is the number of model seeds.

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
    res = {
        "left": a.id,
        "right": b.id,
        "metric": metric,
        "rope": rope,
        "runs": runs,
        "p_left": float(p_left),
        "p_rope": float(p_rope),
        "p_right": float(p_right),
    }
    if fig is not None:
        res["fig"] = fig
    return res


def compare_across_datasets(
    pairs: list[tuple[Result, Result]], metric: str = "accuracy", rope: float = 0.01, plot: bool = False
) -> dict:
    """Hierarchical test of the same two systems on several Datasets.

    ``pairs`` is one ``(a, b)`` per Dataset, each paired as in ``compare`` and with the same number of
    scores. Returns the same keys as ``compare`` with ``left``/``right`` as lists of ids.
    """
    import baycomp

    xs, ys = zip(*[_check_paired(a, b, metric) for a, b in pairs], strict=True)
    x, y = np.stack(xs), np.stack(ys)
    runs = len(pairs[0][0].seeds)
    out = baycomp.two_on_multiple(x, y, rope=rope, runs=runs, plot=plot)
    probs, fig = (out, None) if not plot else out
    p_left, p_rope, p_right = (probs[0], 0.0, probs[1]) if rope == 0 else probs
    res = {
        "left": [a.id for a, _ in pairs],
        "right": [b.id for _, b in pairs],
        "metric": metric,
        "rope": rope,
        "runs": runs,
        "p_left": float(p_left),
        "p_rope": float(p_rope),
        "p_right": float(p_right),
    }
    if fig is not None:
        res["fig"] = fig
    return res
