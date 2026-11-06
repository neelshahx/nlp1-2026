import pytest

from scripts.run_grid import expand, run_grids

GRIDS = {
    "majority": {
        "base": {
            "name": "toy-lengths-majority",
            "dataset": {"name": "toy", "k": 2},
            "tokenizer": {"kind": "wordlevel"},
            "representation": {"kind": "lengths"},
            "model": {"kind": "majority"},
        },
        "overrides": {"representation.pairing": ["concat", "hypothesis_only"]},
    },
    "broken": {
        "base": {
            "name": "toy-broken",
            "dataset": {"name": "toy", "k": 2},
            "tokenizer": {"kind": "wordlevel"},
            "representation": {"kind": "bow", "binary": True, "normalize": True},
            "model": {"kind": "loglinear"},
        },
    },
}


def test_expand_selects_grids_in_order():
    assert [cfg["name"] for _, cfg in expand(GRIDS, ["majority"])] == [
        "toy-lengths-majority-pairing=concat",
        "toy-lengths-majority-pairing=hypothesis_only",
    ]
    with pytest.raises(ValueError, match="unknown grids"):
        expand(GRIDS, ["missing"])


def test_run_grids_skips_stored_and_continues_past_failures(tmp_path):
    first = run_grids(GRIDS, root=tmp_path)
    assert len(first["ran"]) == 2
    assert len(first["failed"]) == 1
    assert first["done"] == []

    second = run_grids(GRIDS, root=tmp_path, dry_run=True)
    assert second["done"] == first["ran"]
    assert second["ran"] == []
