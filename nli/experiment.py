"""Experiment configuration and the runner (shell).

A config is a plain nested dict, one block per stage (see ``EXAMPLE_CONFIG``). ``kind``
selects the concrete class; every other key in a block is passed to it as a keyword
argument, so a new hyperparameter needs no change here.
"""

import copy
import hashlib
import json
from pathlib import Path

from .data import load_anli, load_mnli, load_snli, load_toy
from .folds import make_folds
from .metrics import METRICS, score
from .models import MODELS, build_model
from .representations import REPRESENTATIONS, build_representation
from .store import (
    Result,
    has_result,
    load_result,
    model_path,
    models_dir,
    tokenizers_dir,
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
    texts = list(train["premise"]) + list(train["hypothesis"])
    tokenizer.train(texts)
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

    name, rounds = cfg["dataset"]["name"], cfg["dataset"]["rounds"]
    data = load_anli(tuple(rounds)) if name == "anli" and rounds else DATASETS[name]()
    train, dev = data["train"], data["dev"]
    y = train["label"].to_numpy()
    tokenizer = get_or_train_tokenizer(cfg, train, root)
    folds = make_folds(y, cfg["dataset"]["k"], cfg["dataset"]["data_seed"])
    models_dir(root, eid).mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in cfg["model_seeds"]:
        for fold_i, (tr, ho) in enumerate(folds):
            rep = build_representation(cfg["representation"], tokenizer).fit(train.iloc[tr])
            model = build_model(cfg["model"], seed=seed).fit(rep.transform(train.iloc[tr]), y[tr])

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

    import pandas as pd

    result = Result(eid, cfg, pd.DataFrame(rows))
    write_result(root, result)
    return result
