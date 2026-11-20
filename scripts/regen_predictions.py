"""Re-run stored Experiments into a separate root to regenerate their (gitignored) predictions (shell).

``results/*/predictions/`` is not committed, so item-level analysis needs either a teammate's copy or a rerun.
This reads each Experiment's exact config from ``results/index.csv`` and runs it under ``--root`` (default
``local-runs/``, gitignored), so the committed ``folds.csv`` files are never touched and ids stay identical.

    uv run python scripts/regen_predictions.py snli-hyp-only-bow-loglin-7108aed7 snli-bow-mlp-9d26a5aa
    uv run python scripts/regen_predictions.py --list
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nli.experiment import run_experiment  # noqa: E402
from nli.store import load_index  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", help="experiment ids from results/index.csv")
    parser.add_argument("--root", default=str(REPO / "local-runs"))
    parser.add_argument("--list", action="store_true", help="print the stored ids and exit")
    args = parser.parse_args()

    index = load_index(REPO).set_index("id")
    if args.list or not args.ids:
        print("\n".join(index.index))
        return
    for eid in args.ids:
        config = json.loads(index.loc[eid, "config_json"])
        start = time.time()
        result = run_experiment(config, root=args.root)
        assert result.id == eid, f"config hash drifted: {result.id} != {eid}"
        print(f"[{eid}] done in {time.time() - start:.0f}s: {result.summary()['mean'].to_dict()}", flush=True)


if __name__ == "__main__":
    main()
