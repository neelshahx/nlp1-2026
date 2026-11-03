"""Plots (core: every function takes DataFrames or Results and returns a matplotlib Figure)."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .store import Result


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
    """Stacked bars per split from ``eda.label_distribution``."""
    raise NotImplementedError


def plot_lengths(stats: pd.DataFrame) -> plt.Figure:
    """Histograms of premise and hypothesis token lengths by Label (from ``eda.length_stats``)."""
    raise NotImplementedError


def plot_vocab_coverage(tokenizer_vocab_sizes: pd.DataFrame) -> plt.Figure:
    """OOV rate on dev vs. tokenizer vocabulary size, one line per tokenizer kind."""
    raise NotImplementedError


def plot_confusion(result: Result, split: str = "heldout", seed: int = 0) -> plt.Figure:
    """Confusion matrix pooled over folds from stored predictions (``metrics.confusion``)."""
    raise NotImplementedError


def plot_learning_curve(history: list[dict]) -> plt.Figure:
    """Train/val loss and val accuracy per epoch from ``MLPModel.history``."""
    raise NotImplementedError
