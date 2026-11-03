"""Exploratory data analysis (core: DataFrame in, DataFrame out; plotting lives in ``plotting``)."""

import pandas as pd

from .data import LABELS
from .tokenizers import Tokenizer


def label_distribution(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts and proportions of each Label per split. Rows: split; columns: label names."""
    rows = {}
    for split, df in data.items():
        counts = df["label"].value_counts().reindex(range(len(LABELS)), fill_value=0)
        rows[split] = {LABELS[i]: int(c) for i, c in counts.items()}
    out = pd.DataFrame(rows).T
    out["total"] = out.sum(axis=1)
    return out


def length_stats(pairs: pd.DataFrame, tokenizer: Tokenizer) -> pd.DataFrame:
    """Per-side length stats (chars and tokens) by Label: mean, std, quantiles.

    Entailed hypotheses are typically the shortest (Gururangan et al. 2018).
    """
    raise NotImplementedError


def vocabulary_overlap(pairs: pd.DataFrame, tokenizer: Tokenizer) -> pd.DataFrame:
    """Per Pair: token-set sizes of each side, their overlap, and Jaccard; grouped by Label."""
    raise NotImplementedError


def hypothesis_cue_words(pairs: pd.DataFrame, tokenizer: Tokenizer, top_n: int = 20) -> pd.DataFrame:
    """Tokens most associated with each Label in hypotheses alone (PMI with a count floor).

    Reproduces the Gururangan et al. 2018 artifact table: negation for contradiction,
    generic nouns for entailment, modifiers and purpose clauses for neutral.
    """
    raise NotImplementedError


def breakdown(pairs: pd.DataFrame, column: str) -> pd.DataFrame:
    """Label distribution by a categorical column (``genre`` for MNLI, ``round`` for ANLI)."""
    raise NotImplementedError


def hard_subset(dev_predictions: pd.DataFrame) -> pd.Series:
    """Boolean mask of Pairs a hypothesis-only Experiment got wrong (the hard subset).

    ``dev_predictions`` is what ``store.read_predictions`` returns for that Experiment.
    """
    raise NotImplementedError
