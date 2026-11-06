import numpy as np
import pytest

from nli import data, representations, tokenizers
from nli.experiment import EXAMPLE_CONFIG, run_experiment
from nli.models import MLPModel

SMALL = {"hidden_dim": 16, "num_epochs": 5, "batch_size": 8}


@pytest.fixture(scope="module")
def features():
    toy = data.load_toy()
    train, dev = toy["train"], toy["dev"]
    tok = tokenizers.WordLevelTokenizer().train(train["premise"].tolist() + train["hypothesis"].tolist())
    rep = representations.BagOfWordsRepresentation(tok).fit(train)
    return rep.transform(train), train["label"].to_numpy(), rep.transform(dev)


def test_predict_proba_shape_and_rows_sum_to_one(features):
    X_train, y_train, X_dev = features
    probs = MLPModel(**SMALL).fit(X_train, y_train).predict_proba(X_dev)
    assert probs.shape == (len(X_dev), 3)
    assert np.allclose(probs.sum(axis=1), 1)


def test_same_seed_is_deterministic(features):
    X_train, y_train, X_dev = features
    a = MLPModel(seed=1, **SMALL).fit(X_train, y_train)
    b = MLPModel(seed=1, **SMALL).fit(X_train, y_train)
    assert np.array_equal(a.predict_proba(X_dev), b.predict_proba(X_dev))
    assert a.history == b.history


def test_history_and_early_stopping(features):
    X_train, y_train, _ = features
    model = MLPModel(num_epochs=50, patience=2, hidden_dim=16, batch_size=8).fit(X_train, y_train)
    assert 1 <= len(model.history) <= 50
    assert set(model.history[0]) == {"epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"}
    assert [h["epoch"] for h in model.history] == list(range(1, len(model.history) + 1))


def test_no_validation_runs_all_epochs(features):
    X_train, y_train, _ = features
    model = MLPModel(validation_fraction=0.0, **SMALL).fit(X_train, y_train)
    assert len(model.history) == SMALL["num_epochs"]
    assert "val_loss" not in model.history[0]


@pytest.mark.parametrize("extra", [{}, {"n_blocks": 2, "residual": True}, {"n_blocks": 2, "residual": False}])
def test_save_load_round_trip(features, tmp_path, extra):
    X_train, y_train, X_dev = features
    model = MLPModel(seed=3, dropout=0.2, **SMALL, **extra).fit(X_train, y_train)
    model.save(tmp_path / "mlp")
    loaded = MLPModel.load(tmp_path / "mlp")
    assert (loaded.seed, loaded.dropout) == (3, 0.2)
    assert all(getattr(loaded, k) == v for k, v in extra.items())
    assert loaded.history == model.history
    assert np.allclose(loaded.predict_proba(X_dev), model.predict_proba(X_dev))


def test_lengths_mlp_smoke(tmp_path):
    config = {**EXAMPLE_CONFIG, "name": "toy-lengths-mlp", "model": {"kind": "mlp", **SMALL}}
    first = run_experiment(config, root=tmp_path / "a", verbose=False)
    second = run_experiment(config, root=tmp_path / "b", verbose=False)
    assert len(first.scores) == 3 * 2
    assert first.scores.equals(second.scores)
