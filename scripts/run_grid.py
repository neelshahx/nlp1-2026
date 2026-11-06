"""Run every Experiment in one or more grids, skipping those already stored (shell).

A grid file is JSON: ``{"<grid name>": {"base": <config>, "overrides": {"model.C": [0.1, 1.0], ...}}, ...}``.
A ``note`` key per grid is ignored by the runner. Each grid expands with ``nli.experiment.grid``. Stored
Experiments are skipped, so the script can be stopped and restarted; one failing config is logged and the
rest continue.

    uv run python scripts/run_grid.py grids/snli.json                  # run all grids
    uv run python scripts/run_grid.py grids/snli.json --only mlp       # one grid
    uv run python scripts/run_grid.py grids/snli.json --dry-run        # list configs and status

Long runs: ``nohup uv run python scripts/run_grid.py grids/snli.json > results/grid-log.txt 2>&1 &``.
"""

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nli.experiment import experiment_id, grid, run_experiment  # noqa: E402
from nli.store import has_result  # noqa: E402


def load_grids(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def expand(grids: dict, only: list[str] | None = None) -> list[tuple[str, dict]]:
    """``(grid name, config)`` for every config in the selected grids, in file order."""
    names = only or list(grids)
    unknown = sorted(set(names) - set(grids))
    if unknown:
        raise ValueError(f"unknown grids {unknown}; choose from {sorted(grids)}")
    return [(name, cfg) for name in names for cfg in grid(grids[name]["base"], grids[name].get("overrides", {}))]


def run_grids(grids: dict, root: Path = Path("."), only: list[str] | None = None, dry_run: bool = False) -> dict:
    """Run the selected grids under ``root``; returns ``{"done": [...], "ran": [...], "failed": [...]}`` of ids."""
    report = {"done": [], "ran": [], "failed": []}
    configs = expand(grids, only)
    for i, (name, cfg) in enumerate(configs, 1):
        eid = experiment_id(cfg)
        if has_result(root, eid):
            report["done"].append(eid)
            log(f"[{i}/{len(configs)}] {name}: {eid} done")
            continue
        if dry_run:
            log(f"[{i}/{len(configs)}] {name}: {eid} pending")
            continue
        log(f"[{i}/{len(configs)}] {name}: {eid} running")
        start = time.monotonic()
        try:
            result = run_experiment(cfg, root=root, verbose=False)
        except Exception:
            report["failed"].append(eid)
            log(f"[{i}/{len(configs)}] {name}: {eid} FAILED\n{traceback.format_exc()}")
            continue
        report["ran"].append(eid)
        means = ", ".join(f"{m}={v:.4f}" for m, v in result.summary()["mean"].items())
        log(f"[{i}/{len(configs)}] {name}: {eid} finished in {time.monotonic() - start:.0f}s ({means})")
    return report


def log(message: str) -> None:
    print(f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}", flush=True)


def grid_parser(description: str) -> argparse.ArgumentParser:
    """The arguments every grid script takes: grid file, ``--only``, ``--root``, ``--dry-run``."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid_file", type=Path)
    parser.add_argument("--only", nargs="+", metavar="GRID", help="grid names to run (default: all)")
    parser.add_argument("--root", type=Path, default=Path("."), help="where results/ and models/ live")
    parser.add_argument("--dry-run", action="store_true", help="list configs and whether each is stored")
    return parser


def finish(report: dict, start: float) -> int:
    """Log the totals and the elapsed time; exit status is 1 if anything failed."""
    minutes = (time.monotonic() - start) / 60
    log(f"done={len(report['done'])} ran={len(report['ran'])} failed={len(report['failed'])} in {minutes:.0f} min")
    return 1 if report["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    args = grid_parser(__doc__).parse_args(argv)
    start = time.monotonic()
    report = run_grids(load_grids(args.grid_file), root=args.root, only=args.only, dry_run=args.dry_run)
    return finish(report, start)


if __name__ == "__main__":
    sys.exit(main())
