import numpy as np
import pandas as pd
import pytest

from nli.compare import compare
from nli.ensemble import combine, ensemble_on_dev, ensemble_on_split
from nli.experiment import EXAMPLE_CONFIG, experiment_id, grid, predict_split, run_experiment, validate_config
from nli.store import Result, load_results, read_fold_predictions


def test_experiment_runs_end_to_end(tmp_path):
    result = run_experiment(EXAMPLE_CONFIG, root=tmp_path, verbose=False)
    assert set(result.scores["metric"]) == {"accuracy", "macro_f1"}
    assert len(result.scores) == 3 * 2  # k=3 folds x 2 metrics
    assert (tmp_path / "results" / result.id / "config.json").exists()
    assert (tmp_path / "results" / "index.csv").exists()
    assert (tmp_path / "models" / result.id / "seed-0-fold-0.json").exists()
    assert len(list((tmp_path / "models" / "tokenizers").iterdir())) == 1

    table = load_results(tmp_path)
    assert table["id"].unique().tolist() == [result.id]

    dev_scores = ensemble_on_dev(result, "mean_proba", root=tmp_path)
    assert 0 <= dev_scores["accuracy"] <= 1

    again = run_experiment(EXAMPLE_CONFIG, root=tmp_path, verbose=False)
    assert again.id == result.id and again.scores.equals(result.scores)


def test_predict_split_stores_test_predictions_for_every_fold_model(tmp_path):
    result = run_experiment(EXAMPLE_CONFIG, root=tmp_path, verbose=False)
    frames = predict_split(result, "test", root=tmp_path)
    assert len(frames) == 3
    assert all(len(df) == 9 for df in frames)
    assert (tmp_path / "results" / result.id / "predictions" / "seed-0-fold-2-test.csv").exists()
    stored = read_fold_predictions(tmp_path, result, "test")
    assert stored[1].equals(frames[1])
    with pytest.raises(FileExistsError):
        predict_split(result, "test", root=tmp_path)
    assert len(predict_split(result, "test", root=tmp_path, force=True)) == 3

    test_scores = ensemble_on_split(result, "test", "majority", root=tmp_path)
    assert set(test_scores) == {"accuracy", "macro_f1"}
    assert 0 <= test_scores["accuracy"] <= 1
    assert ensemble_on_split(result, "dev", root=tmp_path) == ensemble_on_dev(result, root=tmp_path)


def test_experiment_id_is_stable_and_config_sensitive():
    base = experiment_id(EXAMPLE_CONFIG)
    assert base.startswith("toy-lengths-majority-") and len(base.split("-")[-1]) == 8
    changed = {**EXAMPLE_CONFIG, "model_seeds": [1]}
    assert experiment_id(changed) != base


def test_validate_config_rejects_unknown_kind():
    bad = {**EXAMPLE_CONFIG, "model": {"kind": "transformer"}}
    with pytest.raises(ValueError):
        validate_config(bad)


def test_grid_dots_into_blocks():
    configs = grid(EXAMPLE_CONFIG, {"dataset.k": [2, 3], "tokenizer.min_frequency": [1, 2]})
    assert len(configs) == 4
    assert configs[0]["name"] == "toy-lengths-majority-k=2-min_frequency=1"
    assert configs[-1]["dataset"]["k"] == 3 and configs[-1]["tokenizer"]["min_frequency"] == 2


def test_combiners():
    a = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    b = np.array([[0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    assert np.allclose(combine([a, b], "mean_proba"), (a + b) / 2)
    votes = combine([a, b, a], "majority")
    assert votes.argmax(axis=1).tolist() == [0, 2]


def _fake_result(id, scores, seeds=(0,)):
    cfg = validate_config({**EXAMPLE_CONFIG, "name": id, "model_seeds": list(seeds)})
    rows = [{"seed": s, "fold": f, "metric": "accuracy", "value": v} for s in seeds for f, v in enumerate(scores)]
    return Result(id, cfg, pd.DataFrame(rows))


def test_compare_is_paired_and_reports_probabilities():
    a = _fake_result("a", [0.70, 0.72, 0.71, 0.69, 0.73])
    b = _fake_result("b", [0.60, 0.63, 0.61, 0.58, 0.62])
    out = compare(a, b, rope=0.01)
    assert out["p_left"] > 0.9
    assert abs(out["p_left"] + out["p_rope"] + out["p_right"] - 1) < 1e-6
    with pytest.raises(ValueError):
        compare(a, _fake_result("c", [0.6, 0.6, 0.6, 0.6, 0.6], seeds=(0, 1)))
