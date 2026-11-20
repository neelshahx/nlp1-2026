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
# # Project 2, point 6: error analysis with the BDA model (V1)
#
# Which properties of an SNLI pair predict whether a system classifies it correctly, and do those properties
# affect different designs differently?
#
# - **Systems (V1):** hypothesis-only vs full (premise + hypothesis) BoW log-linear, both from part 1. The
#   question: what does seeing the premise buy, and on which items? Later versions add systems to `SYSTEMS`
#   (V2: BoW MLP; V3: part 2's embed-and-pool and LSTM, once their dev predictions are stored).
# - **Items:** a fixed, label-stratified sample of SNLI dev (300 per label). Every fold model predicts the
#   same items, so all systems are compared on identical inputs.
# - **Observations:** one per item × system × seed × fold model (correct or not).
# - **Features:** hypothesis length, premise length, lexical overlap and negation in the hypothesis. `hard`
#   (the hypothesis-only model gets the item wrong) joins in V2, once hypothesis-only is no longer one of the
#   systems: for that system it would just restate correctness.
# - **Model:** the tutorial's Bayesian logistic regression (Laplace priors on the weights), via
#   `nli/analysis.py`.
#
# Predictions are gitignored. `ROOT` points at wherever they live: `local-runs/` after
# `scripts/regen_predictions.py`, or `.` with a teammate's copy of `results/*/predictions/`.

# %%
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt

from nli import analysis, data, plotting
from nli.store import load_result
from nli.tokenizers import build_tokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = "local-runs"
SYSTEMS = {  # short name -> experiment id; the order fixes the system ids
    "hyp-only": "snli-hyp-only-bow-loglin-7108aed7",
    "bow-loglin": "snli-bow-loglin-C=0.1-binary=True-2187c5b0",
}
HYPOTHESIS_ONLY = "snli-hyp-only-bow-loglin-7108aed7"
FEATURES = ["hypothesis_len", "premise_len", "overlap", "negation"]  # V2: + "hard"
FIGURES = Path("results/figures/bda-v1")
FIGURES.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Item table and features

# %%
snli = data.load_snli()
ids = analysis.analysis_ids(snli["dev"], n_per_label=300, seed=0)
results = {name: load_result(ROOT, eid) for name, eid in SYSTEMS.items()}
table = analysis.item_table(results, root=ROOT, split="dev", ids=ids)

tokenizer = build_tokenizer({"kind": "wordlevel", "min_frequency": 1}).train(data.texts(snli["train"]))
features = analysis.item_features(snli["dev"], tokenizer).join(
    analysis.hard_flags(load_result(ROOT, HYPOTHESIS_ONLY), root=ROOT)
)
features = features.loc[ids]
print(f"{len(table)} observations: {table.groupby('system').size().to_dict()}")
features.describe().T

# %%
# features should not be near-duplicates of each other, or their weights become unidentifiable
features[FEATURES].corr().round(2)

# %% [markdown]
# ## 2. Look first: accuracy per feature bin
#
# Bars span the min to max over the fold/seed models, so the run-to-run variation is visible.

# %%
binned = {f: analysis.accuracy_by_bin(table, features, f) for f in FEATURES}
fig = plotting.plot_accuracy_by_bin(binned)
fig.savefig(FIGURES / "accuracy_by_bin.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 3. BDA model
#
# $c_n \sim \text{Bernoulli}(\sigma(\phi + \delta_{s_n} + \mathbf w_{s_n}^\top \mathbf f_n))$, with the features
# standardised and preceded by a one-hot of the predicted label (as in the tutorial).

# %%
mcmc, posterior = analysis.fit_bda(table, features, FEATURES, num_chains=2)
analysis.diagnostics(mcmc)  # want max_r_hat ≈ 1.00 and 0 divergences

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
    figure_kwargs={"figsize": (12, 6), "width_ratios": [2, 3], "layout": "none"},
    **az_stats,
)
pc.savefig(FIGURES / "weights.png")

# %% [markdown]
# ## 4. Where do the two designs differ?
#
# Posterior of $w_{\text{full}} - w_{\text{hyp-only}}$ per feature. An interval away from 0 means the feature
# affects the two systems' correctness differently.

# %%
diff = analysis.weight_differences(posterior, "bow-loglin", "hyp-only")
diff.to_csv(FIGURES / "weight_differences.csv", index=False)
diff.round(3)

# %%
plt.show()
