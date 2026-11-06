"""Delete stored log-linear Experiments whose solver did not converge, then run every missing grid config (shell).

Convergence is read from the saved fold models, so this also catches runs made outside a grid. Deleted ids
keep their hash and are rerun with the current defaults; the saved classifier records the solver and
iteration cap actually used.

    uv run python scripts/backfill.py grids/snli.json --dry-run     # list what would be deleted and run
    uv run python scripts/backfill.py grids/snli.json
    nohup uv run python scripts/backfill.py grids/snli.json > results/backfill-log.txt 2>&1 &
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nli.models import LogLinearModel  # noqa: E402
from nli.store import delete_result, load_index, load_result, model_path  # noqa: E402
from scripts.run_grid import finish, grid_parser, load_grids, log, run_grids  # noqa: E402


def unconverged(root: Path) -> list[str]:
    """Ids of stored log-linear Experiments with at least one fold model that stopped at its iteration cap."""
    index = load_index(root)
    if index.empty:
        return []
    out = []
    for eid in index.loc[index["model"] == "loglinear", "id"]:
        result = load_result(root, eid)
        models = (
            LogLinearModel.load(model_path(root, eid, seed, fold)) for seed in result.seeds for fold in range(result.k)
        )
        if not all(m.converged for m in models):
            out.append(eid)
    return out


def main(argv: list[str] | None = None) -> int:
    args = grid_parser(__doc__).parse_args(argv)
    start = time.monotonic()
    for eid in unconverged(args.root):
        log(f"{eid} unconverged: {'would delete' if args.dry_run else 'deleting'}")
        if not args.dry_run:
            delete_result(args.root, eid)
    report = run_grids(load_grids(args.grid_file), root=args.root, only=args.only, dry_run=args.dry_run)
    return finish(report, start)


if __name__ == "__main__":
    sys.exit(main())
