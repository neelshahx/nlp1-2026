from nli.experiment import run_experiment
from nli.store import delete_result, has_result, load_index, models_dir
from scripts.backfill import unconverged

BASE = {
    "name": "toy-bow-loglin",
    "dataset": {"name": "toy", "k": 2},
    "tokenizer": {"kind": "wordlevel"},
    "representation": {"kind": "bow"},
}


def test_unconverged_lists_only_runs_that_hit_max_iter(tmp_path):
    stuck = run_experiment({**BASE, "model": {"kind": "loglinear", "max_iter": 1}}, root=tmp_path, verbose=False)
    fine = run_experiment({**BASE, "model": {"kind": "loglinear", "max_iter": 1000}}, root=tmp_path, verbose=False)
    assert unconverged(tmp_path) == [stuck.id]
    assert has_result(tmp_path, fine.id)


def test_delete_result_removes_files_and_index_row(tmp_path):
    stuck = run_experiment({**BASE, "model": {"kind": "loglinear", "max_iter": 1}}, root=tmp_path, verbose=False)
    delete_result(tmp_path, stuck.id)
    assert not has_result(tmp_path, stuck.id)
    assert not models_dir(tmp_path, stuck.id).exists()
    assert stuck.id not in set(load_index(tmp_path)["id"])
    assert unconverged(tmp_path) == []
