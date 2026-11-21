"""Refit the chosen configs on the full training set and score them once on dev and test (shell).

Takes a grid file like scripts/run_grid.py (grids/final.json holds the finalists with the same configs, and so the
same ids, as their CV runs). Writes results/<id>/final/ and models/<id>/final-seed-S; a config that already has
final scores is skipped, never rerun (the test set is scored once). Ensembles of the refits: scripts/run_ensembles.py --final.

    .venv/bin/python scripts/run_final.py grids/final.json --dry-run
    sbatch --time=03:00:00 scripts/snellius/run_final.job grids/final.json
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nli.experiment import experiment_id, run_final  # noqa: E402
from nli.store import final_dir  # noqa: E402
from scripts.run_grid import expand, finish, grid_parser, load_grids, log  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = grid_parser(__doc__).parse_args(argv)
    start = time.monotonic()
    report = {"done": [], "ran": [], "failed": []}
    configs = expand(load_grids(args.grid_file), args.only)
    for i, (name, cfg) in enumerate(configs, 1):
        eid = experiment_id(cfg)
        tag = f"[{i}/{len(configs)}] {name}: {eid}"
        if (final_dir(args.root, eid) / "scores.csv").exists():
            report["done"].append(eid)
            log(f"{tag} final done; not rerun")
            continue
        if args.dry_run:
            log(f"{tag} final pending")
            continue
        log(f"{tag} final running")
        t = time.monotonic()
        try:
            scores = run_final(cfg, root=args.root)
        except Exception:
            report["failed"].append(eid)
            log(f"{tag} FAILED\n{traceback.format_exc()}")
            continue
        report["ran"].append(eid)
        acc = scores[scores.metric == "accuracy"].pivot(index="seed", columns="split", values="value")
        summary = ", ".join(f"{sp}={acc[sp].mean():.4f} ({' / '.join(f'{v:.4f}' for v in acc[sp])})" for sp in acc)
        log(f"{tag} finished in {time.monotonic() - t:.0f}s ({summary})")
    return finish(report, start)


if __name__ == "__main__":
    sys.exit(main())
