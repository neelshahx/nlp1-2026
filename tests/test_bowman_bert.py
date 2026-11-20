import copy

import numpy as np
import pandas as pd
import pytest
import torch

from nli import data
from nli.experiment import EXAMPLE_CONFIG, experiment_id, run_experiment
from nli.models.bowman_bert import COMBINERS, BowmanBertModel, BowmanBertNet
from nli.representations.bert import BertRepresentation, SentenceVectorPairs
from nli.tokenizers import build_tokenizer

SMALL = {"proj_dim": 8, "mlp_dim": 8, "num_epochs": 3, "batch_size": 8, "device": "cpu"}
DIM = 12


@pytest.fixture
def fake_bert(monkeypatch):
    """Deterministic per-sentence vectors instead of downloading BERT."""

    def encode(self, sentences):
        return np.stack(
            [np.random.default_rng(abs(hash(s)) % 2**32).normal(size=DIM).astype(np.float16) for s in sentences]
        )

    monkeypatch.setattr(BertRepresentation, "encode", encode)


def test_wordpiece_is_a_placeholder(tmp_path):
    tok = build_tokenizer({"kind": "wordpiece"}).train(["a man sleeps"])
    tok.save(tmp_path / "t")
    assert type(tok).load(tmp_path / "t").vocab == {}
    with pytest.raises(NotImplementedError):
        tok.tokenize("a man sleeps")


def test_sentences_pairing_keeps_premise_and_hypothesis_apart(fake_bert):
    toy = data.load_toy()["train"]
    X = BertRepresentation(build_tokenizer({"kind": "wordpiece"}), pairing="sentences").transform(toy).payload
    assert isinstance(X, SentenceVectorPairs) and X.premise.shape == X.hypothesis.shape == (len(toy), DIM)
    assert len(X.take([0, 2])) == 2


@pytest.mark.parametrize("combiner", COMBINERS)
def test_net_widths_and_hypothesis_only_ignores_premise(combiner):
    net = BowmanBertNet(DIM, combiner=combiner, proj_dim=4, mlp_dim=6, dropout=0.0).eval()
    u, v = torch.randn(5, DIM), torch.randn(5, DIM)
    out = net(u, v)
    assert out.shape == (5, 3)
    if combiner == "hypothesis_only":
        assert torch.allclose(out, net(torch.randn(5, DIM), v))


def test_fit_save_load(fake_bert, tmp_path):
    toy = data.load_toy()
    rep = BertRepresentation(build_tokenizer({"kind": "wordpiece"}), pairing="sentences")
    model = BowmanBertModel(seed=1, combiner="matching", **SMALL).fit(
        rep.transform(toy["train"]), toy["train"]["label"].to_numpy()
    )
    probs = model.predict_proba(rep.transform(toy["dev"]))
    assert probs.shape == (len(toy["dev"]), 3) and np.allclose(probs.sum(axis=1), 1)
    model.save(tmp_path / "m")
    assert np.allclose(BowmanBertModel.load(tmp_path / "m").predict_proba(rep.transform(toy["dev"])), probs)


def test_runs_through_the_fold_pipeline_with_wordpiece_in_config(fake_bert, tmp_path):
    cfg = copy.deepcopy(EXAMPLE_CONFIG)
    cfg["name"] = "toy-bert-bowman"
    cfg["tokenizer"] = {"kind": "wordpiece"}
    cfg["representation"] = {"kind": "bert", "pairing": "sentences"}
    cfg["model"] = {"kind": "bowman_bert", "combiner": "matching", **SMALL}
    # the tokenizer kind is part of the config hash, so a mislabelled tokenizer would give a different id
    assert experiment_id(cfg) != experiment_id({**cfg, "tokenizer": {"kind": "treebank"}})
    result = run_experiment(cfg, tmp_path, verbose=False)
    assert len(result.per_fold("accuracy")) == cfg["dataset"]["k"]
    history = pd.read_csv(tmp_path / "results" / result.id / "history.csv")
    assert set(history["fold"]) == set(range(cfg["dataset"]["k"]))
