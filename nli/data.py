"""Dataset loading (shell)."""

import itertools
from typing import cast

import pandas as pd
from datasets import load_dataset as hf_load

LABELS = ["entailment", "neutral", "contradiction"]
COLUMNS = ["id", "premise", "hypothesis", "label"]


def load_anli(rounds: tuple[int, ...] = (1, 2, 3)) -> dict[str, pd.DataFrame]:
    ds = hf_load("facebook/anli")
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "dev", "test"):
        parts: list[pd.DataFrame] = []
        for r in rounds:
            df = cast(pd.DataFrame, ds[f"{split}_r{r}"].to_pandas())
            df.insert(0, "id", [f"anli-{split}-r{r}-{i}" for i in range(len(df))])
            df["round"] = r
            parts.append(df)
        out[split] = pd.concat(parts, ignore_index=True)[COLUMNS + ["round", "reason"]]
    return out


def load_mnli() -> dict[str, pd.DataFrame]:
    ds = hf_load("nyu-mll/multi_nli")

    def frame(hf_split: str, split: str) -> pd.DataFrame:
        df = cast(pd.DataFrame, ds[hf_split].to_pandas())
        df.insert(0, "id", [f"mnli-{split}-{i}" for i in range(len(df))])
        return df[COLUMNS + ["genre", "premise_parse", "hypothesis_parse"]]

    return {
        "train": frame("train", "train"),
        "dev": frame("validation_matched", "dev"),
        "test": frame("validation_mismatched", "test"),
    }


def load_snli() -> dict[str, pd.DataFrame]:
    ds = hf_load("stanfordnlp/snli")

    def frame(hf_split: str, split: str) -> pd.DataFrame:
        df = cast(pd.DataFrame, ds[hf_split].to_pandas())
        df = df[df["label"].ge(0)].reset_index(drop=True)
        df.insert(0, "id", [f"snli-{split}-{i}" for i in range(len(df))])
        return df[COLUMNS]

    return {
        "train": frame("train", "train"),
        "dev": frame("validation", "dev"),
        "test": frame("test", "test"),
    }


def load_toy() -> dict[str, pd.DataFrame]:
    subjects = ["A man", "A woman", "A child", "Two kids", "An old man", "A girl"]
    actions = ["running", "swimming", "reading", "singing", "cooking", "dancing"]
    places = ["park", "pool", "library", "kitchen", "street", "garden"]
    contradictions = ["sleeping", "sitting still", "lying down"]
    neutral_tails = ["for a competition", "because it is sunny", "with a friend"]
    rows = []
    for i, (subject, action) in enumerate(list(itertools.product(subjects, actions))[:18]):
        premise = f"{subject} is {action} in the {places[i % len(places)]}."
        rows.append((premise, f"{subject} is {action}.", "entailment"))
        rows.append((premise, f"{subject} is {contradictions[i % 3]}.", "contradiction"))
        rows.append((premise, f"{subject} is {action} {neutral_tails[i % 3]}.", "neutral"))
    df = pd.DataFrame(rows, columns=["premise", "hypothesis", "label_name"])
    df["label"] = df["label_name"].map(LABELS.index)
    df.insert(0, "id", [f"toy-{i}" for i in range(len(df))])
    df = df[COLUMNS]
    return {
        "train": df.iloc[:36].reset_index(drop=True),
        "dev": df.iloc[36:45].reset_index(drop=True),
        "test": df.iloc[45:].reset_index(drop=True),
    }
