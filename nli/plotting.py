"""Plots (core: every function takes DataFrames or Results and returns a matplotlib Figure)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import LABELS
from .metrics import confusion
from .store import predicted_labels


def plot_scores(results: pd.DataFrame, metric: str = "accuracy", by: str = "name") -> plt.Figure:
    """Box plot plus per-fold points of ``metric`` for each Experiment.

    ``results`` is the long table from ``store.load_results`` (optionally filtered).
    Lifted from ``nlp1-student/part1/analysis.py:30`` ``plot_box_and_points`` minus ``plt.show()``.
    """
    df = results[results["metric"] == metric]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * df[by].nunique()), 4))
    sns.boxplot(data=df, x=by, y="value", ax=ax, color="white", showfliers=False)
    sns.stripplot(data=df, x=by, y="value", hue="seed", ax=ax, dodge=False, alpha=0.7)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_posterior(comparison: dict) -> plt.Figure:
    """The baycomp posterior for a ``compare(..., plot=True)`` result."""
    if "fig" not in comparison:
        raise ValueError("call compare(..., plot=True) to get a figure")
    return comparison["fig"]


def plot_label_distribution(table: pd.DataFrame) -> plt.Figure:
    """Stacked proportion bars per split from ``eda.label_distribution``."""
    shares = table[LABELS].div(table["total"], axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    shares.plot.bar(stacked=True, ax=ax, rot=0)
    ax.set_ylabel("share of pairs")
    ax.set_ylim(0, 1)
    ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_lengths(lengths: pd.DataFrame, unit: str = "tokens") -> plt.Figure:
    """One histogram per side of ``unit`` (tokens or chars) by Label, from ``eda.lengths``."""
    sides = list(dict.fromkeys(lengths["side"]))
    fig, axes = plt.subplots(1, len(sides), figsize=(5 * len(sides), 4), sharey=True)
    for ax, side in zip(np.atleast_1d(axes), sides, strict=True):
        sub = lengths[lengths["side"] == side]
        sns.histplot(
            data=sub, x=unit, hue="label", hue_order=LABELS, element="step", stat="density", common_norm=False, ax=ax
        )
        ax.set_title(side)
    fig.tight_layout()
    return fig


def plot_vocab_coverage(coverage: pd.DataFrame) -> plt.Figure:
    """Out-of-vocabulary (OOV) rate vs. vocabulary size, one line per tokenizer kind; columns kind, vocab_size, oov_rate."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=coverage, x="vocab_size", y="oov_rate", hue="kind", marker="o", ax=ax)
    ax.set_xscale("log")
    ax.set_ylabel("OOV rate on dev")
    fig.tight_layout()
    return fig


def plot_confusion(predictions: pd.DataFrame, title: str = "") -> plt.Figure:
    """Row-normalised confusion matrix of a predictions frame (``store.read_fold_predictions``, concatenated)."""
    matrix = confusion(predictions["label"].to_numpy(), predicted_labels(predictions))
    shares = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        shares, annot=matrix, fmt="d", cmap="Blues", vmin=0, vmax=1, xticklabels=LABELS, yticklabels=LABELS, ax=ax
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_accuracy_by_bin(binned: dict[str, pd.DataFrame]) -> plt.Figure:
    """One panel per feature: accuracy per bin and system, bars spanning min to max over fold/seed models.

    ``binned`` maps a feature name to ``analysis.accuracy_by_bin(table, features, feature)``.
    """
    fig, axes = plt.subplots(1, len(binned), figsize=(4 * len(binned), 3.5), sharey=True, squeeze=False)
    for ax, (feature, df) in zip(axes[0], binned.items(), strict=True):
        bins = list(dict.fromkeys(df["bin"].astype(str)))
        systems = list(dict.fromkeys(df["system"]))
        width = 0.8 / len(systems)
        for i, system in enumerate(systems):
            sub = df[df["system"] == system]
            sub = sub.set_index(sub["bin"].astype(str)).reindex(bins)
            x = np.arange(len(bins)) + (i - (len(systems) - 1) / 2) * width
            err = np.vstack([sub["mean"] - sub["min"], sub["max"] - sub["mean"]]).clip(min=0)  # float round-off
            ax.errorbar(x, sub["mean"], yerr=err, fmt="o", capsize=3, label=system)
        ax.set_xticks(range(len(bins)), bins, rotation=30, ha="right")
        ax.set_title(feature)
        ax.set_xlabel("bin")
    axes[0][0].set_ylabel("accuracy")
    axes[0][-1].legend(loc="best")
    fig.tight_layout()
    return fig


def plot_learning_curve(history: list[dict]) -> plt.Figure:
    """Train/val loss and accuracy per epoch from ``MLPModel.history``."""
    df = pd.DataFrame(history)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, unit in ((ax_loss, "loss"), (ax_acc, "accuracy")):
        ax.plot(df["epoch"], df[f"train_{unit}"], marker="o", label="train")
        if f"val_{unit}" in df:
            ax.plot(df["epoch"], df[f"val_{unit}"], marker="o", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel(unit)
        ax.legend()
    fig.tight_layout()
    return fig
