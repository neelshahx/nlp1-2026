import numpy as np
import pytest

from nli import data, representations, tokenizers
from nli.experiment import EXAMPLE_CONFIG, run_experiment
from nli.models import LogLinearModel, MajorityModel


@pytest.fixture(scope="module")
def toy():
    return data.load_toy()


@pytest.fixture(scope="module")
def features(toy):
    train, dev = toy["train"], toy["dev"]
    tok = tokenizers.WordLevelTokenizer().train(train["premise"].tolist() + train["hypothesis"].tolist())
    rep = representations.BagOfWordsRepresentation(tok).fit(train)
    return rep.transform(train), train["label"].to_numpy(), rep.transform(dev), dev["label"].to_numpy()


def test_predict_proba_shape_and_rows_sum_to_one(features):
    X_train, y_train, X_dev, _ = features
    probs = LogLinearModel().fit(X_train, y_train).predict_proba(X_dev)
    assert probs.shape == (len(X_dev), 3)
    assert np.allclose(probs.sum(axis=1), 1)


def test_beats_majority_on_toy(features):
    X_train, y_train, X_dev, y_dev = features
    loglin = LogLinearModel().fit(X_train, y_train).predict(X_dev)
    majority = MajorityModel().fit(X_train, y_train).predict(X_dev)
    assert (loglin == y_dev).mean() > (majority == y_dev).mean()


def test_same_seed_is_deterministic(features):
    X_train, y_train, X_dev, _ = features
    a = LogLinearModel(seed=1).fit(X_train, y_train).predict_proba(X_dev)
    b = LogLinearModel(seed=1).fit(X_train, y_train).predict_proba(X_dev)
    assert np.array_equal(a, b)


def test_save_load_round_trip(features, tmp_path):
    X_train, y_train, X_dev, _ = features
    model = LogLinearModel(seed=3, C=0.5, l1_ratio=1.0, max_iter=50).fit(X_train, y_train)
    model.save(tmp_path / "loglin")
    loaded = LogLinearModel.load(tmp_path / "loglin")
    assert (loaded.seed, loaded.C, loaded.l1_ratio, loaded.max_iter) == (3, 0.5, 1.0, 50)
    assert np.array_equal(loaded.predict_proba(X_dev), model.predict_proba(X_dev))


def test_missing_training_label_still_gives_three_columns(features):
    X_train, y_train, X_dev, _ = features
    keep = y_train != 1
    subset = representations.Features(X_train.payload[keep], dim=X_train.dim)
    probs = LogLinearModel().fit(subset, y_train[keep]).predict_proba(X_dev)
    assert probs.shape == (len(X_dev), 3)
    assert np.all(probs[:, 1] == 0) and np.allclose(probs.sum(axis=1), 1)


def test_experiment_is_reproducible_across_roots(tmp_path):
    config = {**EXAMPLE_CONFIG, "name": "toy-lengths-loglinear", "model": {"kind": "loglinear"}}
    first = run_experiment(config, root=tmp_path / "a", verbose=False)
    second = run_experiment(config, root=tmp_path / "b", verbose=False)
    assert first.scores.equals(second.scores)


def test_bpe_ngram_loglinear_smoke(tmp_path):
    config = {
        **EXAMPLE_CONFIG,
        "name": "toy-ngram-loglinear",
        "tokenizer": {"kind": "bpe", "vocab_size": 1000},
        "representation": {"kind": "ngram", "n": 2},
        "model": {"kind": "loglinear", "max_iter": 100},
    }
    result = run_experiment(config, root=tmp_path, verbose=False)
    assert len(result.scores) == 3 * 2
