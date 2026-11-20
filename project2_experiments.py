# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project 2, point 5: hyperparameter search and experimental variance
#
# *How did you structure your hyperparameter search? Are you keeping an eye on experimental variance?*
#
# Everything here reads the committed per-fold scores (`results/<id>/folds.csv`), so it needs no predictions.
# Every Experiment uses the same 5 stratified folds of SNLI train (`data_seed=0`); `model_seeds` repeats training
# on those folds with a different initialisation / data order. Accuracy is on each fold's held-out part.

# %%
import warnings
from pathlib import Path

import pandas as pd

from nli import compare, plotting
from nli.store import load_result, load_results

warnings.filterwarnings("ignore", category=FutureWarning)

FIGURES = Path("results/figures/point5")
FIGURES.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. How the search was structured
#
# Not a full grid: one design dimension at a time, each stage carrying forward what the previous one settled
# (as tutorial 2 suggests: "if you are varying encoder type you don't necessarily need to vary it at the same
# time that you vary hidden size"). The saved effort goes into 5 folds everywhere and 2 seeds for the neural
# models.
#
# | stage | varies | held fixed |
# |---|---|---|
# | A. log-linear | C ∈ {0.1, 1, 10} × counts/binary (full 3×2 grid) | BPE 8k, BoW, concat pairing |
# | B. log-linear inputs | BPE vocab 4k/8k/16k, word-level, n-grams n=1/2, joint pairing, hypothesis-only | log-linear |
# | C. BoW MLP | hidden 128/512 × residual blocks 0/2 | BPE 8k BoW input, 10 epochs, early stopping |
# | D. Bowman et al. (2015) | encoder sum/LSTM × GloVe frozen/trainable | the paper's sizes (100d, 3×200 tanh MLP) |
#
# A to B to C to D is also the order of the course (part 1, then embed-and-pool, then recurrent).

# %%
EXPERIMENTS = [  # (stage, short label for plots, id); an id may appear in more than one stage
    ("A. log-linear", "C=0.1 counts", "snli-bow-loglin-C=0.1-binary=False-84c1fe55"),
    ("A. log-linear", "C=0.1 binary", "snli-bow-loglin-C=0.1-binary=True-2187c5b0"),
    ("A. log-linear", "C=1 counts", "snli-bow-loglin-C=1.0-binary=False-cb2a35a9"),
    ("A. log-linear", "C=1 binary", "snli-bow-loglin-C=1.0-binary=True-6ecd012a"),
    ("A. log-linear", "C=10 counts", "snli-bow-loglin-C=10.0-binary=False-01f89ec6"),
    ("A. log-linear", "C=10 binary", "snli-bow-loglin-C=10.0-binary=True-c9de66d8"),
    ("B. log-linear inputs", "BPE 4k", "snli-bow-loglin-vocab_size=4000-cb12fc2e"),
    # the vocab grid's base config (C=1, counts, BPE 8k) is stage A's "C=1 counts" run
    ("B. log-linear inputs", "BPE 8k", "snli-bow-loglin-C=1.0-binary=False-cb2a35a9"),
    ("B. log-linear inputs", "BPE 16k", "snli-bow-loglin-vocab_size=16000-47cda660"),
    ("B. log-linear inputs", "word-level", "snli-wordlevel-bow-loglin-5e8b71c9"),
    ("B. log-linear inputs", "n-gram n=1", "snli-ngram-loglin-n=1-99271563"),
    ("B. log-linear inputs", "n-gram n=2", "snli-ngram-loglin-n=2-c85e6df8"),
    ("B. log-linear inputs", "joint pairing", "snli-bow-loglin-pairing=joint-4b3beda3"),
    ("B. log-linear inputs", "hypothesis only", "snli-hyp-only-bow-loglin-8059a0af"),
    ("C. BoW MLP", "128, 0 blocks (seed 0 only)", "snli-bow-mlp-9d26a5aa"),
    ("C. BoW MLP", "128, 0 blocks", "snli-bow-mlp-hidden_dim=128-n_blocks=0-f8c85fe5"),
    ("C. BoW MLP", "128, 2 blocks", "snli-bow-mlp-hidden_dim=128-n_blocks=2-fc266eb2"),
    ("C. BoW MLP", "512, 0 blocks", "snli-bow-mlp-hidden_dim=512-n_blocks=0-38b6d7fe"),
    ("D. Bowman", "sum, trainable", "snli-glove-bowman-sum-freeze_embeddings=False-efb8a605"),
    ("D. Bowman", "sum, frozen", "snli-glove-bowman-sum-freeze_embeddings=True-c193df9d"),
    ("D. Bowman", "LSTM, trainable", "snli-glove-bowman-lstm-freeze_embeddings=False-2e249164"),
    ("D. Bowman", "LSTM, frozen", "snli-glove-bowman-lstm-freeze_embeddings=True-a6fcfc57"),
]
R = {label: load_result(".", eid) for _, label, eid in EXPERIMENTS}  # label -> Result

all_scores = load_results(".")
missing = set(all_scores["id"]) - {eid for *_, eid in EXPERIMENTS}
assert not missing, f"Experiments without a stage: {missing}"

scores = pd.concat(
    [all_scores[all_scores["id"] == eid].assign(stage=stage, label=label) for stage, label, eid in EXPERIMENTS],
    ignore_index=True,
)
acc = scores[scores["metric"] == "accuracy"]

summary = (
    acc.groupby(["stage", "label"], sort=False)["value"]
    .agg(mean="mean", sd="std", min="min", max="max", n="count")
    .mul({"mean": 100, "sd": 100, "min": 100, "max": 100, "n": 1})
    .round(2)
)
summary.to_csv(FIGURES / "scores_summary.csv")
summary

# %% [markdown]
# ## 2. Look at every fold and seed, not just averages
#
# One panel per stage; each dot is one fold model, coloured by seed. Note the y-axis ranges differ.

# %%
for stage, df in acc.groupby("stage"):
    fig = plotting.plot_scores(df, by="label")
    fig.axes[0].set_title(stage)
    fig.axes[0].set_xlabel("")
    fig.savefig(FIGURES / f"scores_{stage.split('.')[0]}.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 3. Where does the variance come from?
#
# With 2 seeds on the same 5 folds, the spread splits into a **fold** part (which train/held-out split) and a
# **seed** part (training randomness with the data fixed); see `compare.seed_fold_spread`.
#
# - Log-linear (lbfgs) is deterministic: the hypothesis-only run's two seeds are bit-identical, so for the
#   log-linear models one seed is all there is (the extra seed just costs compute).
# - The default MLP was run twice under different ids (`9d26a5aa`, seed 0; `f8c85fe5`, seeds 0 and 1); its seed 0
#   should reproduce exactly, which checks that reruns are deterministic on the same machine.

# %%
spread = pd.DataFrame([{"label": label, **compare.seed_fold_spread(r)} for label, r in R.items() if len(r.seeds) >= 2])
spread = (
    spread.drop(columns="id")
    .set_index("label")
    .mul({"seeds": 1, "mean": 100, "sd_fold": 100, "sd_seed": 100, "max_seed_gap": 100})
    .round(2)
)
spread.to_csv(FIGURES / "variance_seed_vs_fold.csv")
spread

# %%
rerun = R["128, 0 blocks (seed 0 only)"].per_fold("accuracy")
seed0 = compare.restrict_seeds(R["128, 0 blocks"], [0])
print("default MLP, seed 0, max |difference| between the two runs:", abs(rerun - seed0.per_fold("accuracy")).max())

# %% [markdown]
# ## 4. Which differences are real? Bayesian correlated t-test with a ROPE
#
# For each pair (a, b), on the same folds: P(a better), P(practically equivalent), P(b better), where
# "practically equivalent" means |difference| < ROPE = 0.5 accuracy points (as in tutorial 1). The test models the per-fold differences
# with a Student t whose variance is inflated by the Nadeau-Bengio factor (1/n + 1/(k−1)), because fold models
# share most of their training data and so their scores are correlated.
#
# Caveat: baycomp's `runs` assumes repeated cross-validation with *new* folds; our seeds reuse the same folds, so
# with 2 seeds the test is somewhat overconfident. Decisions with P > 0.95 are safe either way.
#
# Log-linear Results only have seed 0 (and seed 1 would be identical), so neural models are compared to them on
# seed 0 only (`restrict_seeds`).


# %%
def on_seed0(r):
    return compare.restrict_seeds(r, [0])


PAIRS = {  # question -> (a, b)
    "A: C=0.1 vs C=1 (binary)": (R["C=0.1 binary"], R["C=1 binary"]),
    "A: binary vs counts (C=0.1)": (R["C=0.1 binary"], R["C=0.1 counts"]),
    "B: BPE 8k vs 16k": (R["BPE 8k"], R["BPE 16k"]),
    "B: premise + hypothesis vs hypothesis only": (R["C=1 counts"], on_seed0(R["hypothesis only"])),
    "B: concat vs joint pairing": (R["C=0.1 binary"], R["joint pairing"]),
    "C: MLP 512 vs 128": (R["512, 0 blocks"], R["128, 0 blocks"]),
    "C: 2 residual blocks vs 0": (R["128, 2 blocks"], R["128, 0 blocks"]),
    "A to C: MLP 512 vs log-linear": (on_seed0(R["512, 0 blocks"]), R["C=0.1 binary"]),
    "C to D: Bowman sum (trainable) vs MLP 512": (R["sum, trainable"], R["512, 0 blocks"]),
    "D: LSTM vs sum (trainable)": (R["LSTM, trainable"], R["sum, trainable"]),
    "D: sum, trainable vs frozen GloVe": (R["sum, trainable"], R["sum, frozen"]),
    "D: LSTM, trainable vs frozen GloVe": (R["LSTM, trainable"], R["LSTM, frozen"]),
}
ROPE = 0.005  # half an accuracy point, as in tutorial 1
table = compare.compare_table(list(PAIRS.values()), rope=ROPE)
table.insert(0, "question", list(PAIRS))
table["mean_diff"] *= 100
table = table[["question", "left", "right", "runs", "mean_diff", "p_left", "p_rope", "p_right"]]
table.to_csv(FIGURES / "rope_0.5pt.csv", index=False)
table.drop(columns=["left", "right"]).round(3)

# %% [markdown]
# Reading the table: `mean_diff` is a − b in accuracy points; `p_left` = P(a better by more than 0.5 points),
# `p_rope` = P(within 0.5 points), `p_right` = P(b better by more than 0.5 points).
#
# The ROPE is a choice. Rerun with a wider ROPE of 1 point to see which conclusions depend on it:

# %%
wide = compare.compare_table(list(PAIRS.values()), rope=0.01)
wide.insert(0, "question", list(PAIRS))
wide[["question", "p_left", "p_rope", "p_right"]].to_csv(FIGURES / "rope_1pt.csv", index=False)
wide[["question", "p_left", "p_rope", "p_right"]].round(3)

# %%
fig = compare.compare(R["LSTM, trainable"], R["sum, trainable"], rope=ROPE, plot=True)["fig"]
fig.savefig(FIGURES / "posterior_lstm_vs_sum.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# baycomp plots the posterior of b − a (here sum − LSTM), so the mass at about −3 points means the LSTM is better.
#
# ## 5. Findings (ROPE = 0.5 points unless stated)
#
# - **Most tuning was within noise.** Stage A (C, counts vs binary), BPE vocabulary size and MLP width/depth: every
#   difference is under 0.5 points and P(practically equivalent) ≥ 0.94. The search could stop early on these.
# - **Three design choices clearly matter:** pairing (concat beats joint by 5.5 points), going from log-linear to
#   an MLP (+8.4) and from sum-pooling to an LSTM encoder (+3.2), each with P(better) ≥ 0.99.
# - **The BoW log-linear model barely uses the premise:** +0.46 points over hypothesis-only, P(tie) = 0.66,
#   P(better) = 0.34; with a 1-point ROPE it is a clear tie (0.997).
# - **GloVe sum-pooling does not beat the BoW MLP** (−0.5 points; P(MLP better) = 0.52, P(tie) = 0.39): pretrained
#   embeddings with sum-pooling did not add accuracy; the recurrent encoder did.
# - **Trainable vs frozen GloVe** matters for sum (+2.3, P(better) = 0.98) but is undecided for the LSTM (+0.3;
#   P = 0.40 / 0.45 / 0.15): more seeds would be needed to settle it.
# - **Variance grows with the model:** log-linear is deterministic (seed sd = 0); for the MLP, seed and fold sd are
#   both ≤ 0.22 points; for the Bowman models the seed sd (0.7–1.2) is as large as or larger than the fold sd, and
#   one fold moved 3.1 points between seeds. One seed per config would have been enough for the linear models and
#   too few for the Bowman ones.
