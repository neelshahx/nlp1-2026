"""Results and models on disk (shell).

Layout, relative to ``root`` (the repo by default)::

    results/index.csv                               one row per Experiment ever run
    results/<id>/config.json
    results/<id>/folds.csv                          long: seed, fold, metric, value
    results/<id>/predictions/seed-S-fold-K-<split>.csv   id, label, p_entailment, p_neutral, p_contradiction
    models/<id>/seed-S-fold-K.<ext>                 fitted Fold models
    models/tokenizers/<dataset>-<tokenizer id>.<ext>
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS

PROB_COLUMNS = [f"p_{name}" for name in LABELS]


class Result:
    """What running an Experiment produced. ``scores`` is long format: seed, fold, metric, value."""

    def __init__(self, id: str, config: dict, scores: pd.DataFrame):
        self.id = id
        self.config = config
        self.scores = scores

    @property
    def folds_key(self) -> tuple:
        d = self.config["dataset"]
        return (d["name"], d["k"], d["data_seed"], tuple(d.get("rounds") or ()))

    @property
    def seeds(self) -> list[int]:
        return list(self.config["model_seeds"])

    def per_fold(self, metric: str) -> np.ndarray:
        """Scores ordered by (seed, fold): the layout ``baycomp`` expects with ``runs=len(seeds)``."""
        df = self.scores[self.scores["metric"] == metric].sort_values(["seed", "fold"])
        return df["value"].to_numpy()

    def mean(self, metric: str) -> float:
        return float(self.per_fold(metric).mean())

    def summary(self) -> pd.DataFrame:
        return self.scores.groupby("metric")["value"].agg(["mean", "std", "count"])

    def __repr__(self) -> str:
        return f"Result({self.id!r}, {self.summary().to_dict()['mean']})"


# --- paths --------------------------------------------------------------------


def results_dir(root: Path, id: str) -> Path:
    return Path(root) / "results" / id


def models_dir(root: Path, id: str) -> Path:
    return Path(root) / "models" / id


def tokenizers_dir(root: Path) -> Path:
    return Path(root) / "models" / "tokenizers"


def model_path(root: Path, id: str, seed: int, fold: int) -> Path:
    return models_dir(root, id) / f"seed-{seed}-fold-{fold}"


# --- write ---------------------------------------------------------------------


def write_result(root: Path, result: Result) -> None:
    d = results_dir(root, result.id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(result.config, indent=2))
    result.scores.to_csv(d / "folds.csv", index=False)
    _append_index(root, result)


def write_predictions(
    root: Path, id: str, seed: int, fold: int, split: str, ids: pd.Series, labels: np.ndarray, probs: np.ndarray
) -> None:
    d = results_dir(root, id) / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(probs, columns=PROB_COLUMNS)
    df.insert(0, "label", labels)
    df.insert(0, "id", ids.to_numpy())
    df.to_csv(d / f"seed-{seed}-fold-{fold}-{split}.csv", index=False)


def _append_index(root: Path, result: Result) -> None:
    path = Path(root) / "results" / "index.csv"
    c = result.config
    row = pd.DataFrame(
        [
            {
                "id": result.id,
                "name": c["name"],
                "dataset": c["dataset"]["name"],
                "tokenizer": c["tokenizer"]["kind"],
                "representation": c["representation"]["kind"],
                "model": c["model"]["kind"],
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "config_json": json.dumps(c, sort_keys=True),
            }
        ]
    )
    if path.exists():
        index = pd.read_csv(path)
        index = pd.concat([index[index["id"] != result.id], row], ignore_index=True)
    else:
        index = row
    index.to_csv(path, index=False)


# --- read ----------------------------------------------------------------------


def has_result(root: Path, id: str) -> bool:
    return (results_dir(root, id) / "folds.csv").exists()


def load_result(root: Path, id: str) -> Result:
    d = results_dir(root, id)
    return Result(id, json.loads((d / "config.json").read_text()), pd.read_csv(d / "folds.csv"))


def load_index(root: Path) -> pd.DataFrame:
    path = Path(root) / "results" / "index.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_results(root: Path = ".") -> pd.DataFrame:
    """Every stored fold score joined with its index row: the table plotting and selection read."""
    index = load_index(root)
    parts = []
    for _, row in index.iterrows():
        scores = pd.read_csv(results_dir(root, row["id"]) / "folds.csv")
        for col in ("id", "name", "dataset", "tokenizer", "representation", "model"):
            scores[col] = row[col]
        parts.append(scores)
    if not parts:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "dataset",
                "tokenizer",
                "representation",
                "model",
                "seed",
                "fold",
                "metric",
                "value",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def read_predictions(root: Path, id: str, seed: int, fold: int, split: str) -> pd.DataFrame:
    return pd.read_csv(results_dir(root, id) / "predictions" / f"seed-{seed}-fold-{fold}-{split}.csv")
