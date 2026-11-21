"""Ensembles of saved predictions, no retraining (shell). Runs where the predictions are (Snellius login node).

Members are the finalists' Fold models. Per ensemble it writes ``results/ens-<name>-<hash>/``:
``config.json`` (members, combiner), ``folds.csv`` (CV: per held-out fold, the fold-K models of every member
config and seed averaged; same long format as a single run, seed 0) and ``dev.csv`` (every Fold model of every
member predicting all of dev). Then it prints each ensemble vs the best single model on CV (correlated Bayesian
t-test, seeds averaged per fold, runs=1, ROPE 0.5 and 1 pt) and on dev.

    .venv/bin/python scripts/run_ensembles.py               # CV + dev
    .venv/bin/python scripts/run_ensembles.py --final       # ensembles of the full-train refits, dev + test

``--final`` reads ``results/<id>/final/predictions`` written by scripts/run_final.py; run it once, after the CV
comparison has decided which ensemble (if any) is evaluated on test.
"""

import argparse
import json
import sys
from pathlib import Path

import baycomp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nli.ensemble import cv_ensemble, final_ensemble, split_ensemble  # noqa: E402
from nli.experiment import config_hash  # noqa: E402
from nli.metrics import score  # noqa: E402
from nli.store import (  # noqa: E402
    PROB_COLUMNS,
    Result,
    final_dir,
    load_result,
    read_predictions,
    results_dir,
)

FINALISTS = {
    "gru": "snli-glove-bowman-gru-freeze_embeddings=False-dc7ea50d",
    "max": "snli-glove-bowman-max-freeze_embeddings=False-b6d749c2",
    "bert_match": "snli-bert-bowman-combiner=matching-0c24c9b3",
}
ENSEMBLES = {  # name -> member configs; every seed of each member is included
    "gru+max+bert_match": ["gru", "max", "bert_match"],  # (a) across configs: order-aware, order-free, contextual
    "gru+max": ["gru", "max"],  # GloVe only: does BERT add anything?
    "gru-seeds": ["gru"],  # (b) across the 2 seeds (and 5 folds) of one config
    "max-seeds": ["max"],
    "bert_match-seeds": ["bert_match"],
}


def seed_avg(per_fold: pd.DataFrame, metric: str = "accuracy") -> np.ndarray:
    """Per-fold score averaged over seeds, in fold order, in points."""
    df = per_fold[per_fold["metric"] == metric]
    return df.groupby("fold")["value"].mean().sort_index().to_numpy() * 100


def single_dev(root: Path, r: Result) -> float:
    """Mean dev accuracy of a config's single Fold models (not an ensemble)."""
    accs = []
    for s in r.seeds:
        for f in range(r.k):
            df = read_predictions(root, r.id, s, f, "dev")
            accs.append(score(df["label"].to_numpy(), df[PROB_COLUMNS].to_numpy(), ("accuracy",))["accuracy"])
    return float(np.mean(accs))


def ensemble_dir(root: Path, name: str, members: list[str], combiner: str) -> Path:
    spec = {"members": members, "combiner": combiner}
    return results_dir(root, f"ens-{name}-{config_hash(spec)}")


def rope(a: np.ndarray, b: np.ndarray) -> str:
    out = []
    for width in (0.5, 1.0):
        p = baycomp.two_on_single(a, b, rope=width, runs=1)
        out.append(f"rope {width}: " + " / ".join(f"{x:.2f}" for x in p))
    return " | ".join(out)


def run_cv(root: Path, singles: dict[str, Result], combiner: str) -> None:
    best = max(singles, key=lambda k: singles[k].mean("accuracy"))
    best_folds = seed_avg(singles[best].scores)
    print("single models (CV mean, seed0 / seed1; dev = mean over the 10 Fold models)")
    for k, r in singles.items():
        by_seed = r.scores[r.scores.metric == "accuracy"].groupby("seed")["value"].mean()
        seeds = " / ".join(f"{v:.3f}" for v in by_seed)
        print(f"  {k:12s} CV {r.mean('accuracy'):.4f} ({seeds})  dev {single_dev(root, r):.4f}")
    print(
        f"\nensembles ({combiner}); ROPE vs best single '{best}' (seed-avg per fold, runs=1): ens > best / rope / best > ens"
    )
    for name, keys in ENSEMBLES.items():
        members = [singles[k] for k in keys]
        folds = cv_ensemble(members, combiner, root)
        dev = split_ensemble(members, "dev", combiner, root)
        out = ensemble_dir(root, name, [r.id for r in members], combiner)
        out.mkdir(parents=True, exist_ok=True)
        cfg = {
            "name": f"ens-{name}",
            "members": [r.id for r in members],
            "member_seeds": {r.id: r.seeds for r in members},
            "combiner": combiner,
            "dataset": members[0].config["dataset"],
            "model_seeds": [0],
            "metrics": members[0].config["metrics"],
            "note": "folds.csv: per held-out fold, every member config x seed combined (seed 0 = the combined model)",
        }
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        folds.to_csv(out / "folds.csv", index=False)
        pd.DataFrame([{"split": "dev", "metric": m, "value": v} for m, v in dev.items()]).to_csv(
            out / "dev.csv", index=False
        )
        ens = seed_avg(folds)
        print(
            f"  {name:20s} CV {ens.mean() / 100:.4f} ({ens.mean() - best_folds.mean():+.2f} pt)  "
            f"dev {dev['accuracy']:.4f}  {rope(ens, best_folds)}  -> {out.name}"
        )


def run_final(root: Path, singles: dict[str, Result], combiner: str) -> None:
    print("Single full-train refits (full train; seed0 / seed1)")
    for k, r in singles.items():
        s = pd.read_csv(final_dir(root, r.id) / "scores.csv")
        s = s[s.metric == "accuracy"].pivot(index="seed", columns="split", values="value")
        print(
            f"  {k:12s} " + "  ".join(f"{sp} {s[sp].mean():.4f} ({' / '.join(f'{v:.3f}' for v in s[sp])})" for sp in s)
        )
    print(f"\nEnsembles of the full-train refits ({combiner})")
    for name, keys in ENSEMBLES.items():
        members = [singles[k] for k in keys]
        rows = []
        for split in ("dev", "test"):
            for m, v in final_ensemble(members, split, combiner, root).items():
                rows.append({"split": split, "metric": m, "value": v})
        out = ensemble_dir(root, name, [r.id for r in members], combiner)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "final.csv", index=False)
        acc = {r["split"]: r["value"] for r in rows if r["metric"] == "accuracy"}
        print(f"  {name:20s} dev {acc['dev']:.4f}  test {acc['test']:.4f}  -> {out.name}/final.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--combiner", default="mean_proba", choices=["mean_proba", "majority"])
    parser.add_argument("--final", action="store_true", help="ensembles of the full-train refits on dev and test")
    args = parser.parse_args()
    singles = {k: load_result(args.root, eid) for k, eid in FINALISTS.items()}
    (run_final if args.final else run_cv)(args.root, singles, args.combiner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
