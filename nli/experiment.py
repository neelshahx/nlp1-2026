"""Experiment configuration and the runner (shell).

A config is a plain nested dict, one block per stage (see ``EXAMPLE_CONFIG``). ``kind``
selects the concrete class; every other key in a block is passed to it as a keyword
argument, so a new hyperparameter needs no change here.

``dataset.train_fraction`` (optional, 0 < f <= 1; learning curves) trains each Fold model on a
stratified subsample of its fold's training rows; held-out folds and dev are unchanged. It is not
a default, so configs without it keep their ids.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_anli, load_mnli, load_snli, load_toy, texts
from .folds import make_folds
from .metrics import METRICS, score
from .models import MODELS, build_model
from .representations import REPRESENTATIONS, build_representation
from .store import (
    Result,
    final_dir,
    final_model_path,
    has_result,
    load_result,
    model_path,
    models_dir,
    read_predictions,
    results_dir,
    tokenizers_dir,
    write_final_predictions,
    write_history,
    write_predictions,
    write_result,
)
from .tokenizers import TOKENIZERS, Tokenizer, build_tokenizer

EXAMPLE_CONFIG = {
    "name": "toy-lengths-majority",
    "dataset": {"name": "toy", "k": 3, "data_seed": 0, "rounds": None},
    "tokenizer": {"kind": "wordlevel", "min_frequency": 1},
    "representation": {"kind": "lengths", "pairing": "concat"},
    "model": {"kind": "majority"},
    "model_seeds": [0],
    "metrics": list(METRICS),
}

DATASETS = {"anli": load_anli, "mnli": load_mnli, "snli": load_snli, "toy": load_toy}

_DATASET_DEFAULTS = {"k": 5, "data_seed": 0, "rounds": None}


def load_dataset(cfg: dict) -> dict[str, pd.DataFrame]:
    """The splits named by a normalised config's ``dataset`` block."""
    name, rounds = cfg["dataset"]["name"], cfg["dataset"]["rounds"]
    return load_anli(tuple(rounds)) if name == "anli" and rounds else DATASETS[name]()


def validate_config(config: dict) -> dict:
    """Fill defaults, check kinds exist, and return a normalised deep copy. Raises ValueError."""
    cfg = copy.deepcopy(config)
    for key in ("name", "dataset", "tokenizer", "representation", "model"):
        if key not in cfg:
            raise ValueError(f"config missing {key!r}")
    cfg["dataset"] = {**_DATASET_DEFAULTS, **cfg["dataset"]}
    if cfg["dataset"]["name"] not in DATASETS:
        raise ValueError(f"unknown dataset {cfg['dataset']['name']!r}")
    if cfg["dataset"]["rounds"] is not None:
        cfg["dataset"]["rounds"] = list(cfg["dataset"]["rounds"])
    if "train_fraction" in cfg["dataset"]:
        fraction = float(cfg["dataset"]["train_fraction"])
        if not 0 < fraction <= 1:
            raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
        cfg["dataset"]["train_fraction"] = fraction
    for block, registry in (("tokenizer", TOKENIZERS), ("representation", REPRESENTATIONS), ("model", MODELS)):
        kind = cfg[block].get("kind")
        if kind not in registry:
            raise ValueError(f"unknown {block} kind {kind!r}; choose from {sorted(registry)}")
    cfg.setdefault("model_seeds", [0])
    cfg["model_seeds"] = [int(s) for s in cfg["model_seeds"]]
    cfg.setdefault("metrics", list(METRICS))
    for m in cfg["metrics"]:
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}; choose from {METRICS}")
    return cfg


def config_hash(config: dict, n: int = 8) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:n]


def experiment_id(config: dict) -> str:
    """``<name>-<8-char hash of the normalised config>``."""
    cfg = validate_config(config)
    return f"{cfg['name']}-{config_hash(cfg)}"


def grid(base: dict, overrides: dict[str, list]) -> list[dict]:
    """Every combination of ``overrides``, keys dotted into ``base``.

    ``grid(cfg, {"model.C": [0.1, 1], "representation.n": [1, 2]})`` gives four configs
    named ``<base name>-C=0.1-n=1`` and so on.
    """
    import itertools

    keys = list(overrides)
    configs = []
    for values in itertools.product(*(overrides[k] for k in keys)):
        cfg = copy.deepcopy(base)
        suffix = []
        for key, value in zip(keys, values, strict=True):
            *path, leaf = key.split(".")
            node = cfg
            for p in path:
                node = node[p]
            node[leaf] = value
            suffix.append(f"{leaf}={value}")
        cfg["name"] = "-".join([base["name"], *suffix])
        configs.append(cfg)
    return configs


def subsample_train(train_idx: np.ndarray, labels: np.ndarray, fraction: float, seed: int, fold: int) -> np.ndarray:
    """A stratified ``fraction`` of a fold's training rows, sorted (pure).

    Deterministic in (data seed, fold), independent of the model seed and the encoder, so every model at one
    fraction sees the same rows; nested, so the 10% subset is inside the 25% one and so on.
    """
    if fraction >= 1:
        return train_idx
    rng = np.random.default_rng([seed, fold])
    keep = []
    for c in np.unique(labels[train_idx]):
        rows = train_idx[labels[train_idx] == c]
        keep.append(rng.permutation(rows)[: int(round(fraction * len(rows)))])
    return np.sort(np.concatenate(keep))


def fold_splits(cfg: dict, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The config's ``(train_idx, heldout_idx)`` per fold, train rows subsampled by ``train_fraction`` if set."""
    d = cfg["dataset"]
    folds = make_folds(labels, d["k"], d["data_seed"])
    fraction = d.get("train_fraction", 1.0)
    return [(subsample_train(tr, labels, fraction, d["data_seed"], i), ho) for i, (tr, ho) in enumerate(folds)]


def tokenizer_id(cfg: dict) -> str:
    d = cfg["dataset"]
    key = {"dataset": d["name"], "rounds": d["rounds"], "tokenizer": cfg["tokenizer"]}
    return f"{d['name']}-{cfg['tokenizer']['kind']}-{config_hash(key)}"


def get_or_train_tokenizer(cfg: dict, train, root: Path) -> Tokenizer:
    """Train once per (dataset, tokenizer config) on train premises + hypotheses; cache on disk."""
    tokenizer = build_tokenizer(cfg["tokenizer"])
    path = tokenizers_dir(root) / tokenizer_id(cfg)
    cached = list(path.parent.glob(path.name + ".*"))
    if cached:
        return type(tokenizer).load(path)
    tokenizer.train(texts(train))
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    return tokenizer


def run_experiment(config: dict, root: Path | str = ".", force: bool = False, verbose: bool = True) -> Result:
    """Run one Experiment end to end and persist it. Returns the stored Result if it already exists.

    Per (model seed, fold): fit the Representation on the fold's training Pairs, fit the
    Model, score the held-out fold, and save held-out and dev-set predictions plus the
    Fold model. Dev predictions are what ``ensemble.ensemble_on_dev`` consumes.
    """
    root = Path(root)
    cfg = validate_config(config)
    eid = experiment_id(cfg)
    if has_result(root, eid) and not force:
        if verbose:
            print(f"[{eid}] already run; loading")
        return load_result(root, eid)

    data = load_dataset(cfg)
    train, dev = data["train"], data["dev"]
    y = train["label"].to_numpy()
    tokenizer = get_or_train_tokenizer(cfg, train, root)
    folds = fold_splits(cfg, y)
    models_dir(root, eid).mkdir(parents=True, exist_ok=True)

    rows, history = [], []
    for seed in cfg["model_seeds"]:
        for fold_i, (tr, ho) in enumerate(folds):
            rep = build_representation(cfg["representation"], tokenizer).fit(train.iloc[tr])
            model = build_model(cfg["model"], seed=seed).fit(rep.transform(train.iloc[tr]), y[tr])
            # models trained by epochs (mlp, bowman) keep one dict per epoch; others have none
            history.extend({"seed": seed, "fold": fold_i, **e} for e in getattr(model, "history", []))

            probs = model.predict_proba(rep.transform(train.iloc[ho]))
            fold_scores = score(y[ho], probs, tuple(cfg["metrics"]))
            rows.extend({"seed": seed, "fold": fold_i, "metric": m, "value": v} for m, v in fold_scores.items())
            write_predictions(root, eid, seed, fold_i, "heldout", train["id"].iloc[ho], y[ho], probs)

            dev_probs = model.predict_proba(rep.transform(dev))
            write_predictions(root, eid, seed, fold_i, "dev", dev["id"], dev["label"].to_numpy(), dev_probs)
            model.save(model_path(root, eid, seed, fold_i))
            if verbose:
                print(
                    f"[{eid}] seed {seed} fold {fold_i}: " + ", ".join(f"{m}={v:.4f}" for m, v in fold_scores.items())
                )

    result = Result(eid, cfg, pd.DataFrame(rows))
    if history:
        write_history(root, eid, pd.DataFrame(history))
    write_result(root, result)
    return result


def predict_split(result: Result, split: str, root: Path | str = ".", force: bool = False) -> list[pd.DataFrame]:
    """Score every stored Fold model of ``result`` on a split and save the predictions.

    Reloads each model and refits its Representation on the fold's training Pairs, exactly as
    ``run_experiment`` did. Returns one predictions frame per (seed, fold), in that order. Meant for the
    test set, once, after model selection on dev; dev predictions are already stored. Refuses to overwrite
    stored predictions for the split unless ``force``.
    """
    root = Path(root)
    cfg = result.config
    first = results_dir(root, result.id) / "predictions" / f"seed-{cfg['model_seeds'][0]}-fold-0-{split}.csv"
    if first.exists() and not force:
        raise FileExistsError(f"{split} predictions already stored for {result.id}; pass force=True to redo them")
    data = load_dataset(cfg)
    train, target = data["train"], data[split]
    tokenizer = get_or_train_tokenizer(cfg, train, root)
    folds = fold_splits(cfg, train["label"].to_numpy())
    model_cls = MODELS[cfg["model"]["kind"]]

    frames = []
    for seed in cfg["model_seeds"]:
        for fold_i, (tr, _) in enumerate(folds):
            rep = build_representation(cfg["representation"], tokenizer).fit(train.iloc[tr])
            model = model_cls.load(model_path(root, result.id, seed, fold_i))
            probs = model.predict_proba(rep.transform(target))
            labels = target["label"].to_numpy()
            write_predictions(root, result.id, seed, fold_i, split, target["id"], labels, probs)
            frames.append(read_predictions(root, result.id, seed, fold_i, split))
    return frames


def run_final(
    config: dict, root: Path | str = ".", splits: tuple[str, ...] = ("dev", "test"), force: bool = False
) -> pd.DataFrame:
    """Refit a config on the full training set, once per model seed, and score it on ``splits``.

    The config is the one selected by cross-validation (same id, so the refit sits next to its CV run):
    scores go to ``results/<id>/final/scores.csv`` (seed, split, metric, value), predictions to
    ``results/<id>/final/predictions/seed-S-<split>.csv``, the per-epoch log to ``final/history.csv`` and
    models to ``models/<id>/final-seed-S``. The test set is meant to be scored once, so this refuses to run
    again for a config that already has final scores unless ``force``. ``train_fraction`` is ignored.
    """
    root = Path(root)
    cfg = validate_config(config)
    eid = experiment_id(cfg)
    out = final_dir(root, eid)
    if (out / "scores.csv").exists() and not force:
        raise FileExistsError(
            f"final evaluation already run for {eid}; the test set is scored once (force=True to redo)"
        )

    data = load_dataset(cfg)
    train = data["train"]
    y = train["label"].to_numpy()
    tokenizer = get_or_train_tokenizer(cfg, train, root)
    rows, history = [], []
    for seed in cfg["model_seeds"]:
        rep = build_representation(cfg["representation"], tokenizer).fit(train)
        model = build_model(cfg["model"], seed=seed).fit(rep.transform(train), y)
        history.extend({"seed": seed, **e} for e in getattr(model, "history", []))
        for split in splits:
            target = data[split]
            labels = target["label"].to_numpy()
            probs = model.predict_proba(rep.transform(target))
            write_final_predictions(root, eid, seed, split, target["id"], labels, probs)
            rows.extend(
                {"seed": seed, "split": split, "metric": m, "value": v}
                for m, v in score(labels, probs, tuple(cfg["metrics"])).items()
            )
        final_model_path(root, eid, seed).parent.mkdir(parents=True, exist_ok=True)
        model.save(final_model_path(root, eid, seed))

    scores = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "final_splits": list(splits)}, indent=2))
    if history:
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    scores.to_csv(out / "scores.csv", index=False)  # written last: its presence marks the final evaluation as done
    return scores
