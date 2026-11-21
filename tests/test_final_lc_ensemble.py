"""train_fraction (learning curve), full-train refit, and cross-config ensembles on the toy dataset."""

import numpy as np
import pytest

from nli.ensemble import cv_ensemble, final_ensemble, split_ensemble
from nli.experiment import EXAMPLE_CONFIG, experiment_id, run_experiment, run_final, subsample_train
from nli.store import final_dir, read_final_predictions


def test_subsample_is_stratified_nested_and_model_seed_free():
    labels = np.repeat([0, 1, 2], [600, 300, 100])
    tr = np.arange(1000)
    small, big = subsample_train(tr, labels, 0.1, 0, 2), subsample_train(tr, labels, 0.5, 0, 2)
    assert np.bincount(labels[small]).tolist() == [60, 30, 10]
    assert set(small) <= set(big)
    assert np.array_equal(small, subsample_train(tr, labels, 0.1, 0, 2))
    assert not np.array_equal(small, subsample_train(tr, labels, 0.1, 0, 3))  # differs per fold
    assert subsample_train(tr, labels, 1.0, 0, 2) is tr


def test_train_fraction_changes_id_only_when_set():
    assert experiment_id(EXAMPLE_CONFIG) == experiment_id({**EXAMPLE_CONFIG})
    lc = {**EXAMPLE_CONFIG, "dataset": {**EXAMPLE_CONFIG["dataset"], "train_fraction": 0.5}}
    assert experiment_id(lc) != experiment_id(EXAMPLE_CONFIG)
    with pytest.raises(ValueError):
        experiment_id({**EXAMPLE_CONFIG, "dataset": {**EXAMPLE_CONFIG["dataset"], "train_fraction": 0}})


def test_train_fraction_runs_end_to_end(tmp_path):
    cfg = {**EXAMPLE_CONFIG, "dataset": {**EXAMPLE_CONFIG["dataset"], "train_fraction": 0.5}}
    result = run_experiment(cfg, root=tmp_path, verbose=False)
    assert len(result.scores) == 3 * 2


def test_run_final_scores_dev_and_test_once(tmp_path):
    cfg = {**EXAMPLE_CONFIG, "model_seeds": [0, 1]}
    scores = run_final(cfg, root=tmp_path)
    assert set(scores["split"]) == {"dev", "test"} and set(scores["seed"]) == {0, 1}
    eid = experiment_id(cfg)
    assert len(read_final_predictions(tmp_path, eid, 1, "test")) == 9
    assert (final_dir(tmp_path, eid) / "config.json").exists()
    with pytest.raises(FileExistsError):
        run_final(cfg, root=tmp_path)


def test_cross_config_ensembles(tmp_path):
    a = run_experiment({**EXAMPLE_CONFIG, "model_seeds": [0, 1]}, root=tmp_path, verbose=False)
    b = run_experiment(
        {**EXAMPLE_CONFIG, "name": "toy-b", "model": {"kind": "loglinear"}}, root=tmp_path, verbose=False
    )
    folds = cv_ensemble([a, b], root=tmp_path)
    assert folds["fold"].tolist().count(0) == 2 and set(folds["seed"]) == {0}
    # an ensemble of one model with itself is that model
    same = cv_ensemble([a], root=tmp_path, seeds=[0])
    assert np.allclose(
        same[same.metric == "accuracy"]["value"], a.scores.query("seed == 0 and metric == 'accuracy'")["value"]
    )
    assert 0 <= split_ensemble([a, b], "dev", root=tmp_path)["accuracy"] <= 1
    run_final(a.config, root=tmp_path)
    run_final(b.config, root=tmp_path)
    assert set(final_ensemble([a, b], "test", root=tmp_path)) == {"accuracy", "macro_f1"}
