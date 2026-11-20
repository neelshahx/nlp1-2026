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
# # Project 2, point 6: error analysis with the BDA model (V1.2: + GloVe max pool, BERT)
#
# Which properties of an SNLI pair predict whether a system classifies it correctly, and do those properties
# affect different designs differently?
#
# - **Systems:** one per major design choice: hypothesis-only BoW log-linear (reference), full BoW log-linear,
#   BoW MLP, Bowman et al. (2015) GloVe sum-pooling (embed-and-pool) and GloVe LSTM (recurrent).
# - **Items:** a fixed, label-stratified sample of SNLI dev (300 per label); every fold model predicts the same items.
# - **Observations:** one per item × system × seed × fold model (correct or not).
# - **Features:** hypothesis/premise length, lexical overlap, negation, and two label-relative annotation-artifact
#   features: `cue_gold` (the hypothesis has a cue word for its gold label) and `cue_misleading` (it has a cue for
#   another label). Cue words: top-15 PMI per label on train hypotheses, add-100 smoothing, document counts
#   (Gururangan et al. 2018); a purpose clause counts as a neutral cue.
# - **Model:** the tutorial's Bayesian logistic regression; V1.1 conditions on the gold label (`label:neutral`,
#   `label:contradiction`; entailment = reference) instead of the predicted label.
#
# Predictions are gitignored and live under different roots; `link_results` gathers them as symlinks in
# `local-runs/analysis/`.

# %%
import json
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

from nli import analysis, data, plotting
from nli.store import load_result
from nli.tokenizers import build_tokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

SYSTEMS = {  # short name -> (experiment id, root holding results/<id>/predictions); the order fixes system ids
    "hyp-only": ("snli-hyp-only-bow-loglin-8059a0af", "local-runs/regen/snli-hyp-only-bow-loglin-8059a0af"),
    "bow-loglin": ("snli-bow-loglin-C=0.1-binary=True-2187c5b0", "local-runs"),
    "mlp": (
        "snli-bow-mlp-hidden_dim=128-n_blocks=0-f8c85fe5",
        "local-runs/regen/snli-bow-mlp-hidden_dim=128-n_blocks=0-f8c85fe5",
    ),
    "sum": ("snli-glove-bowman-sum-freeze_embeddings=False-efb8a605", "."),
    "lstm": ("snli-glove-bowman-lstm-freeze_embeddings=False-2e249164", "."),
    # V1.2: best embed-and-pool and best BERT (frozen, mean-pooled, matching features)
    "max": ("snli-glove-bowman-max-freeze_embeddings=False-b6d749c2", "."),
    "bert": ("snli-bert-bowman-combiner=matching-0c24c9b3", "."),
}
FEATURES = ["hypothesis_len", "premise_len", "overlap", "negation", "cue_gold", "cue_misleading"]
DIFFERENCES = [
    ("bow-loglin", "hyp-only"), ("mlp", "bow-loglin"), ("lstm", "sum"), ("lstm", "bow-loglin"),
    ("lstm", "max"), ("bert", "lstm"), ("bert", "bow-loglin"),  # V1.2
]
FIGURES = Path("results/figures/bda-v1.2")
FIGURES.mkdir(parents=True, exist_ok=True)

ROOT = analysis.link_results({eid: root for eid, root in SYSTEMS.values()}, "local-runs/analysis")

# %% [markdown]
# ## 0. Do the stored predictions belong to the committed scores?
#
# Regenerated predictions are only trustworthy if their run reproduces the committed per-fold scores.

# %%
for name, (eid, _) in SYSTEMS.items():
    committed = load_result(".", eid).scores.set_index(["seed", "fold", "metric"])["value"]
    stored = load_result(ROOT, eid).scores.set_index(["seed", "fold", "metric"])["value"]
    print(f"{name:11s} max |score diff| vs committed: {(committed - stored).abs().max():.2g}")

# %% [markdown]
# ## 1. Items, features, cue words

# %%
snli = data.load_snli()
ids = analysis.analysis_ids(snli["dev"], n_per_label=300, seed=0)
results = {name: load_result(ROOT, eid) for name, (eid, _) in SYSTEMS.items()}
table = analysis.item_table(results, root=ROOT, split="dev", ids=ids)
print(f"{len(table)} observations: {table.groupby('system').size().to_dict()}")

tokenizer = build_tokenizer({"kind": "wordlevel", "min_frequency": 1}).train(data.texts(snli["train"]))
cues = analysis.cue_words(snli["train"], tokenizer, top_n=15, smoothing=100.0)
cues.to_csv(FIGURES / "cue_words.csv", index=False)
all_features = analysis.item_features(snli["dev"], tokenizer).join(analysis.cue_features(snli["dev"], tokenizer, cues))
features = all_features.loc[ids]
cues.groupby("label", sort=False)["token"].apply(", ".join)

# %%
# share of items per gold label where each feature fires (full dev set), cf. Gururangan et al. 2018 Table 3
rates = analysis.feature_rates_by_label(snli["dev"], all_features)
rates.to_csv(FIGURES / "feature_rates_by_label.csv")
rates.round(3)

# %%
features[FEATURES].corr().round(2)

# %% [markdown]
# ## 2. Look first: accuracy per gold label and per feature bin

# %%
by_label = analysis.accuracy_by_label(table)
by_label.to_csv(FIGURES / "accuracy_by_label.csv")
by_label.round(3)

# %%
binned = {f: analysis.accuracy_by_bin(table, features, f) for f in FEATURES}
pd.concat([b.assign(feature=f) for f, b in binned.items()]).astype({"bin": str}).to_csv(
    FIGURES / "accuracy_by_bin.csv", index=False
)  # V1.2: exact numbers for the slides
fig = plotting.plot_accuracy_by_bin(binned)
fig.savefig(FIGURES / "accuracy_by_bin.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 3. BDA model
#
# c_n ~ Bernoulli(σ(φ + δ_s + w_sᵀ f_n)), features standardised; f_n = [label:neutral, label:contradiction,
# FEATURES].

# %%
mcmc, posterior = analysis.fit_bda(table, features, FEATURES, num_chains=2)
diag = analysis.diagnostics(mcmc)
(FIGURES / "diagnostics.json").write_text(json.dumps(diag, indent=2))
diag  # want max_r_hat ≈ 1.00 and 0 divergences

# %%
acc = posterior["posterior"]["accuracy"].stack(sample=("chain", "draw"))
acc_summary = pd.DataFrame(
    {
        "posterior_mean": acc.mean("sample").to_numpy(),
        "low_95": acc.quantile(0.025, "sample").to_numpy(),
        "high_95": acc.quantile(0.975, "sample").to_numpy(),
        "observed": table.groupby("system")["correct"].mean().reindex(list(SYSTEMS)).to_numpy(),
    },
    index=pd.Index(list(acc["system"].to_numpy()), name="system"),
)
acc_summary.to_csv(FIGURES / "accuracy_posterior.csv")
acc_summary.round(4)

# %%
az_stats = {"point_estimate": "median", "ci_kind": "hdi", "ci_probs": (0.5, 0.95)}
pc = az.plot_forest(posterior, var_names=["accuracy"], combined=True, **az_stats)
pc.savefig(FIGURES / "accuracy.png")

# %%
pc = az.plot_forest(
    posterior,
    var_names=["w"],
    combined=True,
    shade_label="system",
    figure_kwargs={"figsize": (14, 10), "width_ratios": [2, 3], "layout": "none"},
    **az_stats,
)
pc.savefig(FIGURES / "weights.png")

# %% [markdown]
# ## 4. Where do the designs differ?
#
# Posterior of w_a − w_b per feature. A 95% interval that excludes 0 means the feature affects the two systems'
# correctness differently.

# %%
diffs = []
for a, b in DIFFERENCES:
    d = analysis.weight_differences(posterior, a, b)
    d.to_csv(FIGURES / f"weight_differences_{a}_minus_{b}.csv", index=False)
    diffs.append(d.assign(comparison=f"{a} - {b}"))
diffs = pd.concat(diffs, ignore_index=True)
diffs["excludes_0"] = (diffs["low_95"] > 0) | (diffs["high_95"] < 0)
diffs.to_csv(FIGURES / "weight_differences_all.csv", index=False)
diffs[diffs["excludes_0"]].round(3)

# %%
plt.show()
