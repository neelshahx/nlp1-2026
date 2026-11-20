"""Item-level error analysis: which input features predict whether a system gets an item right.

The BDA model (``one_hot``, ``numpyro_model``, ``fit_bda_model``, ``posterior_to_arviz``) is copied from
``nlp1-student/part2/analysis.py`` (NLP1 tutorial 2) with its logic unchanged (reformatted, docstrings
condensed); everything else here is ours.
"""

import re
from pathlib import Path

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.infer import MCMC, NUTS

from .data import LABELS
from .eda import vocabulary_overlap
from .ensemble import combine
from .store import PROB_COLUMNS, Result, read_predictions
from .tokenizers import Tokenizer

# --- items and features (ours) --------------------------------------------------------------------------

NEGATION = re.compile(r"\b(?:no|not|never|nobody|nothing|none|nowhere|neither|nor|cannot)\b|n't")
# Purpose clauses ("to X", because, so that): a neutral cue in SNLI (Gururangan et al. 2018, section 4)
PURPOSE = re.compile(r"\b(?:because|in order to|so that)\b|\bto [a-z]+\b")


def link_results(sources: dict[str, Path | str], dest: Path | str) -> Path:
    """One results root for systems whose stored Results live under different roots (shell).

    ``sources`` maps an experiment id to the root that holds ``results/<id>/``; each becomes a symlink
    ``dest/results/<id>``, so ``item_table`` and ``load_result`` can read every system from ``dest``.
    """
    dest = Path(dest)
    (dest / "results").mkdir(parents=True, exist_ok=True)
    for eid, root in sources.items():
        target = (Path(root) / "results" / eid).resolve()
        if not (target / "predictions").is_dir():
            raise FileNotFoundError(f"no predictions for {eid} under {root}")
        link = dest / "results" / eid
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)
    return dest


def cue_words(pairs: pd.DataFrame, tokenizer: Tokenizer, top_n: int = 15, smoothing: float = 100.0) -> pd.DataFrame:
    """Top ``top_n`` hypothesis words per label by smoothed PMI (Gururangan et al. 2018, section 3.1).

    Counts are document-level (a word counts once per hypothesis). Every (word, label) count gets
    ``smoothing`` added before ``PMI(w, c) = log p(w, c) / (p(w) p(c))``, which keeps rare words from topping
    the list. Use the TRAIN pairs. Columns: label, token, count, pmi.
    """
    rows = [
        (label, token)
        for label, text in zip(pairs["label"], pairs["hypothesis"], strict=True)
        for token in set(tokenizer.tokenize(text))
    ]
    counts = pd.crosstab(
        pd.Series([t for _, t in rows], name="token"), pd.Series([LABELS[c] for c, _ in rows], name="label")
    )
    counts = counts.reindex(columns=LABELS, fill_value=0)
    smoothed = counts + smoothing
    total = smoothed.to_numpy().sum()
    p_joint = smoothed / total
    p_word = smoothed.sum(axis=1) / total
    p_label = smoothed.sum(axis=0) / total
    pmi = np.log(p_joint.div(p_word, axis=0).div(p_label, axis=1))
    out = []
    for label in LABELS:
        top = pmi[label].sort_values(ascending=False).head(top_n)
        out.append(
            pd.DataFrame(
                {
                    "label": label,
                    "token": top.index,
                    "count": counts.loc[top.index, label].to_numpy(),
                    "pmi": top.to_numpy(),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def cue_features(pairs: pd.DataFrame, tokenizer: Tokenizer, cues: pd.DataFrame) -> pd.DataFrame:
    """Per item id: which labels' cues its hypothesis contains, and whether they match its gold label.

    ``cue:<label>`` is 1 if the hypothesis has one of that label's ``cues`` words (a purpose clause counts as
    a neutral cue). ``cue_gold``: it has a cue for its gold label (the shortcut points the right way);
    ``cue_misleading``: it has a cue for another label (the shortcut points the wrong way). An item can have both.
    """
    words = {label: set(cues.loc[cues["label"] == label, "token"]) for label in LABELS}
    tokens = [set(t) for t in tokenizer.tokenize_many(pairs["hypothesis"])]
    purpose = pairs["hypothesis"].str.lower().str.contains(PURPOSE).to_numpy()
    has = {label: np.array([bool(t & words[label]) for t in tokens]) for label in LABELS}
    has["neutral"] = has["neutral"] | purpose
    gold = pairs["label"].to_numpy()
    matrix = np.stack([has[label] for label in LABELS], axis=1)  # (items, labels)
    rows = np.arange(len(gold))
    cue_gold = matrix[rows, gold]
    others = matrix.copy()
    others[rows, gold] = False
    out = {f"cue:{label}": has[label].astype(int) for label in LABELS}
    out |= {
        "purpose": purpose.astype(int),
        "cue_gold": cue_gold.astype(int),
        "cue_misleading": others.any(axis=1).astype(int),
    }
    return pd.DataFrame(out, index=pd.Index(pairs["id"].to_numpy(), name="id"))


def feature_rates_by_label(pairs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Per gold label: median hypothesis length and the share of items where each binary feature fires."""
    df = features.join(
        pd.Series(pairs["label"].map(lambda i: LABELS[i]).to_numpy(), index=pairs["id"].to_numpy(), name="gold")
    )
    binary = [c for c in features.columns if set(features[c].unique()) <= {0, 1}]
    rates = df.groupby("gold")[binary].mean().reindex(LABELS)
    rates.insert(0, "median_hypothesis_len", df.groupby("gold")["hypothesis_len"].median().reindex(LABELS))
    rates.insert(1, "n_items", df.groupby("gold").size().reindex(LABELS))
    return rates.T


def accuracy_by_label(table: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per system and gold label, averaged over all fold/seed models (rows: system, columns: label)."""
    out = table.groupby(["system", "label"])["correct"].mean().unstack()
    out.columns = [LABELS[c] for c in out.columns]
    out["all"] = table.groupby("system")["correct"].mean()
    return out.reindex(table.drop_duplicates("system_id").sort_values("system_id")["system"])


def analysis_ids(pairs: pd.DataFrame, n_per_label: int = 300, seed: int = 0) -> np.ndarray:
    """A fixed, label-stratified sample of item ids: keeps MCMC fast and the analysis set identical across runs."""
    return pairs.groupby("label", group_keys=False).sample(n=n_per_label, random_state=seed)["id"].to_numpy()


def item_table(
    results: dict[str, Result], root: Path | str = ".", split: str = "dev", ids: np.ndarray | None = None
) -> pd.DataFrame:
    """Long table of every stored prediction: system, seed, fold, id, label, pred, confidence, correct.

    ``results`` maps a short system name (used in plots) to its Result. Every fold model of every seed
    contributes one row per item, so the BDA model sees run-to-run variation rather than an average.
    """
    parts = []
    for system, result in results.items():
        for seed in result.seeds:
            for fold in range(result.k):
                df = read_predictions(Path(root), result.id, seed, fold, split)
                if ids is not None:
                    df = df[df["id"].isin(ids)]
                probs = df[PROB_COLUMNS].to_numpy()
                pred = probs.argmax(axis=1)
                parts.append(
                    pd.DataFrame(
                        {
                            "system": system,
                            "seed": seed,
                            "fold": fold,
                            "id": df["id"].to_numpy(),
                            "label": df["label"].to_numpy(),
                            "pred": pred,
                            "confidence": probs.max(axis=1),
                            "correct": (pred == df["label"].to_numpy()).astype(int),
                        }
                    )
                )
    table = pd.concat(parts, ignore_index=True)
    table["system_id"] = pd.Categorical(table["system"], categories=list(results)).codes
    return table


def hard_flags(hypothesis_only: Result, root: Path | str = ".", split: str = "dev") -> pd.Series:
    """Per item id: 1 if the hypothesis-only Experiment's fold ensemble gets it wrong (Gururangan et al. 2018).

    Uses the mean-probability ensemble over all fold models, so the flag is a property of the item rather than
    of one fold. Keep the hypothesis-only system itself out of the BDA systems: for it this feature is circular.
    """
    frames = [
        read_predictions(Path(root), hypothesis_only.id, s, f, split)
        for s in hypothesis_only.seeds
        for f in range(hypothesis_only.k)
    ]
    probs = combine([df[PROB_COLUMNS].to_numpy() for df in frames], "mean_proba")
    wrong = probs.argmax(axis=1) != frames[0]["label"].to_numpy()
    return pd.Series(wrong.astype(int), index=frames[0]["id"].to_numpy(), name="hard")


def item_features(pairs: pd.DataFrame, tokenizer: Tokenizer) -> pd.DataFrame:
    """Per item id: premise_len, hypothesis_len (tokens), overlap (share of distinct hypothesis tokens that occur
    in the premise; the lexical-overlap heuristic of McCoy et al. 2019) and negation (hypothesis contains a
    negation word; the strongest contradiction cue in SNLI, Gururangan et al. 2018)."""
    vo = vocabulary_overlap(pairs, tokenizer)
    return pd.DataFrame(
        {
            "premise_len": [len(t) for t in tokenizer.tokenize_many(pairs["premise"])],
            "hypothesis_len": [len(t) for t in tokenizer.tokenize_many(pairs["hypothesis"])],
            "overlap": (vo["overlap"] / vo["hypothesis_size"].clip(lower=1)).to_numpy(),
            "negation": pairs["hypothesis"].str.lower().str.contains(NEGATION).astype(int).to_numpy(),
        },
        index=pd.Index(pairs["id"].to_numpy(), name="id"),
    )


def accuracy_by_bin(table: pd.DataFrame, features: pd.DataFrame, feature: str, q: int = 4) -> pd.DataFrame:
    """Accuracy per (system, feature bin), with its spread across fold/seed models.

    Continuous features are cut into ``q`` quantile bins; binary ones keep their two values. Columns: system,
    bin, n_items, mean (accuracy averaged over models), std, min, max (across models).
    """
    values = features[feature]
    bins = values if values.nunique() <= 2 else pd.qcut(values, q=q, duplicates="drop")
    df = table.join(bins.rename("bin"), on="id")
    per_model = df.groupby(["system", "bin", "seed", "fold"], observed=True)["correct"].mean()
    out = per_model.groupby(["system", "bin"], observed=True).agg(["mean", "std", "min", "max"])
    out.insert(0, "n_items", df.groupby(["system", "bin"], observed=True)["id"].nunique())
    return out.reset_index()


def design_matrix(
    table: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Features for the BDA model, one row per table row: gold-label indicators, then ``feature_cols``.

    ``label:neutral`` and ``label:contradiction`` (entailment is the reference, absorbed by the intercept) let a
    system be better or worse on some gold labels. V1 used the tutorial's one-hot of the *predicted* label; that
    is a system output rather than a property of the input, and a full one-hot is collinear with the intercept.
    """
    joined = table.join(features[feature_cols], on="id")
    if joined[feature_cols].isna().any().any():
        raise ValueError("some table ids have no features")
    gold = one_hot(joined["label"].to_numpy(), len(LABELS))[:, 1:]
    X = np.hstack([gold, joined[feature_cols].to_numpy(dtype=float)])
    names = [f"label:{name}" for name in LABELS[1:]] + list(feature_cols)
    return X, names


def fit_bda(table: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str], **mcmc_kwargs):
    """Fit the tutorial's BDA model to an item table. Returns ``(mcmc, posterior)``, ``posterior`` for arviz."""
    X, names = design_matrix(table, features, feature_cols)
    systems = list(table.drop_duplicates("system_id").sort_values("system_id")["system"])
    mcmc = fit_bda_model(
        num_systems=len(systems),
        sys_id=table["system_id"].to_numpy(),
        features=X,
        correctness=table["correct"].to_numpy(),
        **mcmc_kwargs,
    )
    return mcmc, posterior_to_arviz(mcmc, system_names=systems, feature_names=names)


def weight_differences(posterior, a: str, b: str) -> pd.DataFrame:
    """Posterior of ``w[a] - w[b]`` per feature: mean, 95% central interval and P(difference > 0).

    ``posterior`` is what ``fit_bda`` returns. This is the quantity point 6 asks about: a feature with a
    clearly non-zero difference is one along which the two designs' correctness behaves differently.
    Weights are on standardised features (logit scale).
    """
    w = posterior["posterior"]["w"].stack(sample=("chain", "draw"))  # (system, feature, sample)
    diff = (w.sel(system=a) - w.sel(system=b)).transpose("sample", "feature").to_numpy()
    return pd.DataFrame(
        {
            "feature": list(w["feature"].to_numpy()),
            "mean": diff.mean(axis=0),
            "low_95": np.quantile(diff, 0.025, axis=0),
            "high_95": np.quantile(diff, 0.975, axis=0),
            "p_positive": (diff > 0).mean(axis=0),
        }
    )


def diagnostics(mcmc) -> dict[str, float]:
    """Max r_hat over all parameters (want ~1.00) and number of divergent transitions (want 0)."""
    from numpyro.diagnostics import summary

    stats = summary(mcmc.get_samples(group_by_chain=True))
    return {
        "max_r_hat": float(max(np.max(s["r_hat"]) for s in stats.values())),
        "divergences": int(mcmc.get_extra_fields()["diverging"].sum()),
    }


# --- BDA model: copied from nlp1-student/part2/analysis.py ----------------------------------------------


def one_hot(y, num_classes):
    """
    Return a matrix [N, K] where each row is a K-dimensional one-hot encoding of the classes in the N-dimensional vector of classes.

    y: [N] a vector of N classes (each from 0 to K-1)
    num_classes: (int) number K of classes
    """
    return np.eye(num_classes, dtype=int)[y]


def numpyro_model(num_systems, sys_id, features, correctness=None, s_std=1.0, w_std=1.0):
    """
    Defines a Bayesian regressor for correctness.

    Observables:
        * c[n] is the correctness label for the nth observation
        * s[n] is the system that produced the nth observation
        * f[n] is the vector of I features that describe the nth input
    Latent parameters:
        * global_mean: a scalar
        * sys_shift: a vector of scalars, one per system
        * w: a collection of I-dimensional weights, one I-dimensional vector per system

    Likelihood function:

        c[n] | s[n], f[n] ~ Bernoulli(sigmoid(logit[n]))
            where logit[n] = global_mean + sys_shift[s[n]] + dot(w[s[n]], f[n])

    Priors:
        * Normal(0, 2) on global_mean, Laplace(0, 1) on sys_shift and w
    """
    n, d = features.shape

    global_mean = numpyro.sample("global_mean", dist.Normal(0.0, 2.0))

    with numpyro.plate("systems", num_systems):
        sys_shift = numpyro.sample("sys_shift", dist.Laplace(0.0, s_std))
        w = numpyro.sample("w", dist.Laplace(0.0, w_std).expand([d]).to_event(1))

    mu = global_mean + sys_shift[sys_id] + jnp.sum(w[sys_id] * features, axis=-1)
    mu = numpyro.deterministic("logit", mu)

    numpyro.sample("correctness", dist.Bernoulli(logits=mu), obs=correctness)

    # Evaluate EVERY system on the SAME feature/item/class settings: (n_observations, num_systems).
    comparison_logits = global_mean + sys_shift[None, :] + features @ w.T
    mean_probability = numpyro.deterministic("accuracy", jnp.mean(jax.nn.sigmoid(comparison_logits), axis=0))
    numpyro.deterministic("accuracy_difference", mean_probability[:, None] - mean_probability[None, :])


def fit_bda_model(
    num_systems,
    sys_id,
    features,
    correctness,
    s_std=1.0,
    w_std=1.0,
    seed=0,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
):
    """
    Fit the BDA model above: draw posterior samples via MCMC (NUTS).

    Standardises the input features (approximately 0 mean, unit std) before fitting, so weights are comparable.
    """
    sids = np.asarray(sys_id)
    x = np.asarray(features, dtype=float)
    x_mean = x.mean(axis=0)
    x_scale = jnp.sqrt(x.var(axis=0) + 1e-5)
    x = (x - x_mean) / x_scale

    y = np.asarray(correctness)

    if not isinstance(num_systems, (int, np.integer)) or num_systems <= 0:
        raise ValueError("num_systems must be a positive integer.")
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("features must be a nonempty N-by-D array.")
    if sids.shape != (x.shape[0],):
        raise ValueError("sys_id must have length N.")
    if not np.issubdtype(sids.dtype, np.integer):
        raise ValueError("System IDs must be integers.")
    if np.any(sids < 0) or np.any(sids >= num_systems):
        raise ValueError("System IDs must be in [0, num_systems).")
    if not np.all(np.isfinite(x)):
        raise ValueError("Features must be finite.")
    if y.shape != (x.shape[0],) or not np.all((y == 0) | (y == 1)):
        raise ValueError("correctness must contain N binary values.")

    mcmc = MCMC(
        NUTS(numpyro_model, target_accept_prob=0.9),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",  # works with a single device
    )
    mcmc.run(
        random.PRNGKey(seed),
        num_systems=num_systems,
        sys_id=jnp.asarray(sids, dtype=jnp.int32),
        features=jnp.asarray(x),
        correctness=jnp.asarray(y, dtype=jnp.int32),
        s_std=s_std,
        w_std=w_std,
    )
    return mcmc


def posterior_to_arviz(mcmc, system_names, feature_names):
    """Package the numpyro posterior for arviz plotting (dims: system, feature, system_a, system_b)."""
    import arviz as az

    return az.from_numpyro(
        mcmc,
        dims={
            "global_mean": [],
            "sys_shift": ["system"],
            "w": ["system", "feature"],
            "accuracy": ["system"],
            "accuracy_difference": ["system_a", "system_b"],
        },
        coords={
            "system": system_names,
            "feature": feature_names,
            "system_a": system_names,
            "system_b": system_names,
        },
    )
