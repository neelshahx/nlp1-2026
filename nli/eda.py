"""Exploratory data analysis (core: DataFrame in, DataFrame out; plotting lives in ``plotting``)."""

import numpy as np
import pandas as pd

from .data import LABELS, label_names
from .metrics import METRICS, score
from .store import PROB_COLUMNS, predicted_labels
from .tokenizers import UNK, Tokenizer

SIDES = ("premise", "hypothesis")


def label_distribution(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts and proportions of each Label per split. Rows: split; columns: label names."""
    rows = {}
    for split, df in data.items():
        counts = df["label"].value_counts().reindex(range(len(LABELS)), fill_value=0)
        rows[split] = {LABELS[i]: int(c) for i, c in counts.items()}
    out = pd.DataFrame(rows).T
    out["total"] = out.sum(axis=1)
    return out


def lengths(pairs: pd.DataFrame, tokenizer: Tokenizer) -> pd.DataFrame:
    """One row per pair and side: id, label, side, chars, tokens."""
    parts = []
    for side in SIDES:
        parts.append(
            pd.DataFrame(
                {
                    "id": pairs["id"].to_numpy(),
                    "label": label_names(pairs).to_numpy(),
                    "side": side,
                    "chars": pairs[side].str.len().to_numpy(),
                    "tokens": [len(t) for t in tokenizer.tokenize_many(pairs[side])],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def length_stats(pairs: pd.DataFrame, tokenizer: Tokenizer) -> pd.DataFrame:
    """Mean, std and quartiles of chars and tokens per (label, side); entailed hypotheses are typically shortest."""
    long = lengths(pairs, tokenizer)
    stats = long.groupby(["label", "side"])[["chars", "tokens"]].describe(percentiles=[0.25, 0.5, 0.75])
    stats.columns = [f"{unit}_{stat}" for unit, stat in stats.columns]
    return stats.drop(columns=[c for c in stats.columns if c.endswith(("_count", "_min", "_max"))])


def vocabulary_overlap(pairs: pd.DataFrame, tokenizer: Tokenizer) -> pd.DataFrame:
    """Per pair: id, label, premise_size, hypothesis_size (distinct tokens), overlap, jaccard."""
    p_sets = [set(t) for t in tokenizer.tokenize_many(pairs["premise"])]
    h_sets = [set(t) for t in tokenizer.tokenize_many(pairs["hypothesis"])]
    overlap = np.array([len(p & h) for p, h in zip(p_sets, h_sets, strict=True)])
    union = np.array([len(p | h) for p, h in zip(p_sets, h_sets, strict=True)])
    return pd.DataFrame(
        {
            "id": pairs["id"].to_numpy(),
            "label": label_names(pairs).to_numpy(),
            "premise_size": [len(s) for s in p_sets],
            "hypothesis_size": [len(s) for s in h_sets],
            "overlap": overlap,
            "jaccard": np.divide(overlap, union, out=np.zeros(len(union)), where=union > 0),
        }
    )


def hypothesis_cue_words(
    pairs: pd.DataFrame, tokenizer: Tokenizer, top_n: int = 20, min_count: int = 20
) -> pd.DataFrame:
    """Top ``top_n`` hypothesis tokens per Label by PMI, counting a token once per hypothesis.

    ``pmi = log p(token | label) / p(token)``; tokens seen in fewer than ``min_count`` hypotheses of a
    label are dropped. Columns: label, token, count, pmi.
    """
    rows = [
        {"label": label, "token": token}
        for label, text in zip(label_names(pairs), pairs["hypothesis"], strict=True)
        for token in set(tokenizer.tokenize(text))
    ]
    presence = pd.DataFrame(rows, columns=["label", "token"])
    counts = presence.groupby(["label", "token"]).size().rename("count").reset_index()
    counts = counts[counts["count"] >= min_count].copy()
    if counts.empty:
        return pd.DataFrame(columns=["label", "token", "count", "pmi"])
    n_label = label_names(pairs).value_counts()  # hypotheses per label
    n_token = presence["token"].value_counts()  # hypotheses containing the token
    p_token_given_label = counts["count"] / counts["label"].map(n_label)
    p_token = counts["token"].map(n_token) / len(pairs)
    counts["pmi"] = np.log(p_token_given_label / p_token)
    counts["label"] = pd.Categorical(counts["label"], categories=LABELS)
    return (
        counts.sort_values(["label", "pmi", "token"], ascending=[True, False, True])
        .groupby("label", observed=True)
        .head(top_n)
        .reset_index(drop=True)
    )


def breakdown(pairs: pd.DataFrame, column: str) -> pd.DataFrame:
    """Label counts by a categorical column (``genre`` for MNLI, ``round`` for ANLI), plus a total column."""
    table = pd.crosstab(pairs[column], pairs["label"]).reindex(columns=range(len(LABELS)), fill_value=0)
    table.columns = LABELS
    table["total"] = table.sum(axis=1)
    return table


def oov_rate(pairs: pd.DataFrame, tokenizer: Tokenizer) -> float:
    """Share of tokens across both sides that the tokenizer maps to its unknown token."""
    tokens = [t for side in SIDES for toks in tokenizer.tokenize_many(pairs[side]) for t in toks]
    return sum(t == UNK for t in tokens) / len(tokens) if tokens else 0.0


def hard_subset(dev_predictions: pd.DataFrame) -> pd.Series:
    """Boolean mask, indexed by pair id, of the pairs a hypothesis-only Experiment got wrong.

    ``dev_predictions`` is what ``store.read_predictions`` returns for that Experiment.
    """
    wrong = predicted_labels(dev_predictions) != dev_predictions["label"].to_numpy()
    return pd.Series(wrong, index=dev_predictions["id"].to_numpy(), name="hard")


def hard_subset_scores(
    predictions: list[pd.DataFrame], hypothesis_only_predictions: list[pd.DataFrame]
) -> pd.DataFrame:
    """Per fold: share of pairs in the hard subset and the full Experiment's scores on just those pairs.

    Both lists come from ``store.read_fold_predictions`` on the same split; fold ``i`` of each is paired.
    """
    rows = []
    for fold, (full, hyp) in enumerate(zip(predictions, hypothesis_only_predictions, strict=True)):
        hard = hard_subset(hyp)
        subset = full.set_index("id").loc[hard[hard].index]
        row = {"fold": fold, "hard_share": float(hard.mean()), **dict.fromkeys(METRICS, np.nan)}
        if len(subset):
            row.update(score(subset["label"].to_numpy(), subset[PROB_COLUMNS].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)
