"""Dataset loading (shell).

Every loader returns ``{"train": df, "dev": df, "test": df}``. Each DataFrame has the
columns in ``COLUMNS`` first, then dataset-specific extras (``genre`` and parse strings
for MNLI, ``round`` and ``reason`` for ANLI). Labels are ints indexing ``LABELS``;
pairs without a gold label are dropped.
"""
from __future__ import annotations

import itertools

import pandas as pd

LABELS = ["entailment", "neutral", "contradiction"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
SPLITS = ("train", "dev", "test")
COLUMNS = ["id", "premise", "hypothesis", "label"]

DATASETS = ("snli", "mnli", "anli", "toy")


def load_dataset(name: str, rounds: tuple[int, ...] | None = None) -> dict[str, pd.DataFrame]:
    """Load one Dataset by name. ``rounds`` only applies to ANLI (default: all three)."""
    if name == "toy":
        return toy()
    if name == "snli":
        return _snli()
    if name == "mnli":
        return _mnli()
    if name == "anli":
        return _anli(rounds or (1, 2, 3))
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")


def _frame(hf_split, id_prefix: str | None, extras: list[str]) -> pd.DataFrame:
    df = hf_split.to_pandas()
    df = df[df["label"] >= 0].reset_index(drop=True)
    if id_prefix is not None:
        df.insert(0, "id", [f"{id_prefix}-{i}" for i in range(len(df))])
    return df[COLUMNS + extras]


def _snli() -> dict[str, pd.DataFrame]:
    from datasets import load_dataset as hf_load

    ds = hf_load("stanfordnlp/snli")
    return {
        "train": _frame(ds["train"], "snli-train", []),
        "dev": _frame(ds["validation"], "snli-dev", []),
        "test": _frame(ds["test"], "snli-test", []),
    }


def _mnli() -> dict[str, pd.DataFrame]:
    """MNLI has no public test labels: dev = validation_matched, test = validation_mismatched."""
    from datasets import load_dataset as hf_load

    ds = hf_load("nyu-mll/multi_nli")
    extras = ["genre", "premise_parse", "hypothesis_parse"]

    def frame(split):
        df = split.to_pandas().rename(columns={"pairID": "id"})
        df = df[df["label"] >= 0].reset_index(drop=True)
        return df[COLUMNS + extras]

    return {
        "train": frame(ds["train"]),
        "dev": frame(ds["validation_matched"]),
        "test": frame(ds["validation_mismatched"]),
    }


def _anli(rounds: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    from datasets import load_dataset as hf_load

    ds = hf_load("facebook/anli")
    out = {}
    for split, hf_name in (("train", "train"), ("dev", "dev"), ("test", "test")):
        parts = []
        for r in rounds:
            df = ds[f"{hf_name}_r{r}"].to_pandas().rename(columns={"uid": "id"})
            df["round"] = r
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        df = df[df["label"] >= 0].reset_index(drop=True)
        out[split] = df[COLUMNS + ["round", "reason"]]
    return out


# --- toy dataset -------------------------------------------------------------

_SUBJECTS = ["A man", "A woman", "A child", "Two kids", "An old man", "A girl"]
_ACTIONS = ["running", "swimming", "reading", "singing", "cooking", "dancing"]
_PLACES = ["park", "pool", "library", "kitchen", "street", "garden"]
_CONTRADICTIONS = ["sleeping", "sitting still", "lying down"]
_NEUTRAL_TAILS = ["for a competition", "because it is sunny", "with a friend"]


def toy() -> dict[str, pd.DataFrame]:
    """A deterministic 54-pair dataset for smoke tests: 36 train, 9 dev, 9 test, balanced."""
    rows = []
    combos = list(itertools.product(_SUBJECTS, _ACTIONS))
    for i, (subject, action) in enumerate(combos[:18]):
        place = _PLACES[i % len(_PLACES)]
        premise = f"{subject} is {action} in the {place}."
        rows.append((premise, f"{subject} is {action}.", "entailment"))
        rows.append((premise, f"{subject} is {_CONTRADICTIONS[i % 3]}.", "contradiction"))
        rows.append((premise, f"{subject} is {action} {_NEUTRAL_TAILS[i % 3]}.", "neutral"))
    df = pd.DataFrame(rows, columns=["premise", "hypothesis", "label_name"])
    df["label"] = df["label_name"].map(LABEL_TO_ID)
    df.insert(0, "id", [f"toy-{i}" for i in range(len(df))])
    df = df[COLUMNS]
    return {"train": df.iloc[:36].reset_index(drop=True),
            "dev": df.iloc[36:45].reset_index(drop=True),
            "test": df.iloc[45:].reset_index(drop=True)}
