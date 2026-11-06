# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project 1

# %% [markdown]
# Welcome to the first NLP1 project. The goal of the first tutorial was to get you familiar with text segmentation and simple text classification for topic detection.
# Given the skills you have developed in the first week, you will now be applying them within a different dataset and task.
#
# We are recommending a set of tasks and associated datasets; using those, you will be experimenting on problems introduced during the tutorials. You will be applying hands-on the experiments you have already explored in the last tutorial, and taking a deeper dive: extending your experiments, trying different set ups, possibly considering different evaluation techniques and analysing errors.
#
# The week after the deadline, your TA will perform a short oral assessment of the skills and knowledge you acquired (make sure to check the official schedule for this), probing you to answer questions about what you learnt, about what you did and about why you did the things you did (as opposed to some reasonable alternatives) as well as pros and cons of your method, and so on.
#
# As the tutorials will be building upon your knowledge throughout the weeks, the task and dataset you choose now will be the ones you use in projects 2 and 3; and the ones you will be finally writing about in your final blog post.

# %% [markdown]
# ## Get to work!

# %% [markdown]
# The next section has a checklist to help you not lose track of what needs to be done. Here we have a more informal guide, with prompts and questions for you.
#
#
# 1. [Choose a dataset](http://nlp-illc.github.io/nlp1/). Spend some time understanding how the dataset looks and some statistics of it (e.g. how many instances of each classification label we have?)
# 2. Remember what we learnt during the tutorial: to develop any classifier of the kind we saw, we needed a way to segment text. Do you want to use the tokeniser we have developed? Do you need to train a new one? If you do, what hyperparameters shall you use? What data shall you use to train it? Consider that your dataset may be too small to train a tokeniser! Think whether you will need a larger resource of similar nature to train a tokeniser.
# 3. Start thinking how to represent your documents into features. We have experimented with token occurences and frequencies. Can you think of any other ways, possibly?
# 4. We had experimented with Logistic Regression and a shallow neural network. You can experiment with and assess different architectures now and their hyper parameters (e.g. add layers, possibly change the activation functions, drop out rates).
# 5. Evaluate your models! Was the evaluation method adequate for your new experimental setup? Do you want to assess other aspects of performance?
# 6. Analyse the various models you assessed and try to decide on which is the best.
#
# Be mindful that you will be using your experiments to write up your final blog, so make sure they are clean and neat!
#
# You can write your own code and you can reuse code from the tutorial and its packages or from other sources. You may use coding assistants if you like, but use it mindfully as you take full responsibility for this kind of (semi-)automated coding. Also, keep a record of the kinds of things you delegate to any form of assistant, as you will be asked to disclose this in your blog post (not because we will pass a value judgement on it, but because it is a widespread and good practice to document such things).
#
# You may use jupyter notebooks or python files, it's up to you as we will not directly assess your code. But, once again, be organised, as we will ask you to share a reproducible version of your experiments along with your blog post (again, because this is how real research is publicised).
#
# If you are unsure about anything, ask your TA! They are always happy to help you! Happy coding! :)

# %% [markdown]
# ## Checklist

# %% [markdown]
# What **we expect you to do**:
#
# - [ ] Compare different systems of text classification for your choice of data/task.
# - Your comparisons should cover:
#     - [ ] the two representation spaces in the tutorial (token indicators and bag-of-words) and an additional one of your choice.
#     - [ ] the two kinds of architectures in the tutorial (log-linear and single-hidden-layer MLP) and an additional one of your choice (such as a modification of the MLP).
# - [ ] Your comparisons must be based on cross-validation and a relevant metric (accuracy or else).
# - [ ] You should make use of Bayesian estimation to help you analyse results in relation to a region of practical equivalence (ROPE), and you can motivate the ROPE size yourself.
#
# What you need to **bring to the oral assessment**:
# * You will be asked questions about the points above, you can bring a slide deck with up to five slides (one for each of the following topics _data and metrics_, _features_, _architectures_, _results_, _extra_); you _will not_ give a presentation, rather, these slides are to be thought of as "backup slides" that you can use to guide your answers to the questions the TA will ask you.
# * You may also be asked questions about the tutorial, for those, you don't need to prepare any deliverable (a good way to prepare is to work through the tutorial and reflect on the questions there).
#
# What you should **keep in mind** given that you will eventually write a blog post about your work throughout the course:
# * Organise your code;
# * Summarise your motivations and findings;
# * Document your sources and any form of AI-assistance you use.

# %% [markdown]
# ## Setup

# %%
import pandas as pd

from nli import data, eda, plotting
from nli.compare import compare, compare_across_datasets
from nli.ensemble import ensemble_on_dev
from nli.experiment import EXAMPLE_CONFIG, grid, run_experiment
from nli.store import load_results, read_fold_predictions
from nli.tokenizers import build_tokenizer

DATASETS = ["snli", "mnli", "anli"]

# %% [markdown]
# ## 1. Data and metrics

# %% [markdown]
# Label distribution per split, length statistics, hypothesis-only cue words (Gururangan et al. 2018).
# Metrics: accuracy (primary), macro-F1. Folds: stratified k=5, data seed 0, shared by every experiment on a dataset.

# %%
toy = data.load_toy()
eda.label_distribution(toy)

# %%
tokenizer = build_tokenizer({"kind": "wordlevel", "min_frequency": 2}).train(data.texts(toy["train"]))

# %%
plotting.plot_label_distribution(eda.label_distribution(toy))
eda.length_stats(toy["train"], tokenizer)

# %%
plotting.plot_lengths(eda.lengths(toy["train"], tokenizer))
eda.vocabulary_overlap(toy["train"], tokenizer).groupby("label")[["overlap", "jaccard"]].mean()

# %%
eda.hypothesis_cue_words(toy["train"], tokenizer, top_n=5, min_count=4)

# %%
# SNLI: the Gururangan et al. 2018 artifact table (negation -> contradiction, ...)
snli = data.load_snli()
eda.label_distribution(snli)
snli_tokenizer = build_tokenizer({"kind": "wordlevel", "min_frequency": 2}).train(data.texts(snli["train"]))
eda.length_stats(snli["train"], snli_tokenizer)
eda.hypothesis_cue_words(snli["train"], snli_tokenizer)

# %%
# OOV on dev vs. vocabulary size, per tokenizer kind
tokenizer_configs = [{"kind": "wordlevel", "min_frequency": n} for n in (1, 2, 5, 10)] + [
    {"kind": "bpe", "vocab_size": n} for n in (1000, 4000, 8000, 16000)
]
coverage = []
for cfg in tokenizer_configs:
    trained = build_tokenizer(cfg).train(data.texts(snli["train"]))
    coverage.append({"kind": cfg["kind"], "vocab_size": len(trained.vocab), "oov_rate": eda.oov_rate(snli["dev"], trained)})
plotting.plot_vocab_coverage(pd.DataFrame(coverage))

# %%
# MNLI by genre, ANLI by round
eda.breakdown(data.load_mnli()["train"], "genre")
eda.breakdown(data.load_anli()["train"], "round")

# %% [markdown]
# ## 2. Features

# %% [markdown]
# Tokenizers: `wordlevel` (baseline), `bpe` (Neel). Representations: `bow` (counts, or `binary=True`; Emil),
# `ngram` (Neel), `lengths` (length-artifact baseline). Pairing: `concat` (default), `joint`, `hypothesis_only`.

# %%
smoke = run_experiment(EXAMPLE_CONFIG)
smoke

# %%
# Hypothesis-only baselines (Gururangan et al. 2018): well above chance on SNLI/MNLI, near chance on ANLI.
# One concat + one hypothesis_only Result per dataset; compare each pair; score the concat model on the
# hard subset (dev pairs the hypothesis-only model got wrong).
hyp_only = {}
for name in DATASETS:
    base = {
        **EXAMPLE_CONFIG,
        "dataset": {"name": name, "k": 5},
        "tokenizer": {"kind": "bpe", "vocab_size": 8000},
        "representation": {"kind": "bow"},
        "model": {"kind": "loglinear"},
    }
    full = run_experiment({**base, "name": f"{name}-bow-loglin"})
    hyp = run_experiment(
        {**base, "name": f"{name}-hyp-only-bow-loglin", "representation": {"kind": "bow", "pairing": "hypothesis_only"}}
    )
    hard = eda.hard_subset_scores(read_fold_predictions(".", full, "dev"), read_fold_predictions(".", hyp, "dev"))
    posterior = compare(full, hyp)
    hyp_only[name] = {
        "concat": full.mean("accuracy"),
        "hypothesis_only": hyp.mean("accuracy"),
        "concat_on_hard": hard["accuracy"].mean(),
        "hard_share": hard["hard_share"].mean(),
        **{k: posterior[k] for k in ("p_left", "p_rope", "p_right")},
    }
pd.DataFrame(hyp_only).T

# %% [markdown]
# ## 3. Architectures

# %% [markdown]
# Models: `majority` (floor), `loglinear` (Neel), `mlp` (Emil), plus a third of our choice.
# Hyperparameter search is a grid of experiments, selected by mean CV score.

# %%
# base = {...}
# configs = grid(base, {"model.C": [0.1, 1.0, 10.0], "representation.n": [1, 2]})
# results = [run_experiment(c) for c in configs]

# %% [markdown]
# ## 4. Results

# %% [markdown]
# Per-fold score distributions, ROPE comparisons between the top candidates (within and across datasets),
# ensembles of fold models on the dev set.

# %%
table = load_results()
plotting.plot_scores(table, metric="accuracy")

# %%
# confusion matrix of the smoke run's fold models, pooled over folds; learning curve of one MLP fold model
plotting.plot_confusion(pd.concat(read_fold_predictions(".", smoke, "heldout")), title=smoke.id)
from nli.models import MLPModel
from nli.store import model_path

mlp_smoke = run_experiment({**EXAMPLE_CONFIG, "name": "toy-lengths-mlp", "model": {"kind": "mlp", "num_epochs": 10}})
plotting.plot_learning_curve(MLPModel.load(model_path(".", mlp_smoke.id, seed=0, fold=0)).history)

# %%
# a, b = results[0], results[1]
# compare(a, b, metric="accuracy", rope=0.01, plot=True)
# ensemble_on_dev(a, "mean_proba")

# %% [markdown]
# ## 5. Extra

# %% [markdown]
# Matched vs mismatched gap on MNLI, hard-subset scores, error analysis by genre/round, the logic-based question.

# %%
