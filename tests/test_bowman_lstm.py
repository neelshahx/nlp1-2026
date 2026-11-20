import copy

import numpy as np
import pytest
import torch

from nli import data
from nli.embeddings import glove_matrix, glove_words
from nli.experiment import EXAMPLE_CONFIG, run_experiment
from nli.models.bowman import ENCODERS, BowmanModel, BowmanNet, collate
from nli.representations import GloveRepresentation
from nli.tokenizers import TreebankTokenizer

SMALL = {"proj_dim": 8, "hidden_dim": 8, "mlp_dim": 8, "num_epochs": 3, "batch_size": 8, "device": "cpu"}


@pytest.fixture(scope="module")
def glove_file(tmp_path_factory):
    """Tiny 4d 'GloVe' covering most toy words, so OOV handling is exercised too."""
    toy = data.load_toy()
    tok = TreebankTokenizer()
    words = sorted({w for t in data.texts(toy["train"]) for w in tok.tokenize(t)})
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("glove") / "glove.txt"
    path.write_text("".join(f"{w} {' '.join(f'{x:.3f}' for x in rng.normal(size=4))}\n" for w in words[:-3]))
    return path


def test_glove_matrix_aligns_rows_and_zeroes_pad(tmp_path):
    path = tmp_path / "glove.txt"
    path.write_text("cat 1 2 3\ndog 4 5 6\nbird 7 8 9\n")
    vocab = ["<pad>", "<unk>", "dog", "zebra", "cat"]
    emb, coverage = glove_matrix(vocab, path, seed=0)
    assert emb.shape == (5, 3)
    assert torch.equal(emb[0], torch.zeros(3))
    assert emb[2].tolist() == [4, 5, 6] and emb[4].tolist() == [1, 2, 3]
    assert coverage == 2 / 3
    assert glove_words(path) == {"cat", "dog", "bird"}


def test_treebank_lowercases_and_splits_punctuation():
    assert TreebankTokenizer().tokenize("A man isn't running.") == ["a", "man", "is", "n't", "running", "."]


@pytest.mark.parametrize("encoder", ENCODERS)
def test_padding_does_not_change_output(encoder):
    torch.manual_seed(0)
    net = BowmanNet(torch.randn(10, 6), encoder=encoder, proj_dim=4, hidden_dim=5, mlp_dim=6, dropout=0.0).eval()
    p, pl, h, hl = collate([[2, 3], [4, 5, 6, 7]], [[8], [9, 2, 3]])
    logits = net(p, pl, h, hl)
    assert logits.shape == (2, 3)
    alone = net(p[:1, :2], pl[:1], h[:1, :1], hl[:1])  # the short pair without any padding
    assert torch.allclose(alone, logits[:1], atol=1e-6)


def test_glove_representation_vocab_from_fit_rows_only(glove_file):
    toy = data.load_toy()
    rep = GloveRepresentation(TreebankTokenizer(), path=str(glove_file)).fit(toy["train"].iloc[:5])
    X = rep.transform(toy["dev"]).payload
    assert len(X) == len(toy["dev"]) and X.embeddings.shape == (len(rep.vocab), 4)
    assert rep.unk_rate(toy["dev"]) > 0  # words outside the 5 fitted rows map to <unk>
    assert all(i < len(rep.vocab) for s in X.premise + X.hypothesis for i in s)


@pytest.mark.parametrize("freeze", [False, True])
def test_fit_predict_save_load(glove_file, tmp_path, freeze):
    toy = data.load_toy()
    rep = GloveRepresentation(TreebankTokenizer(), path=str(glove_file)).fit(toy["train"])
    X, y = rep.transform(toy["train"]), toy["train"]["label"].to_numpy()
    model = BowmanModel(seed=1, freeze_embeddings=freeze, **SMALL).fit(X, y)
    assert model._net is not None
    assert torch.equal(model._net.embed.weight.cpu(), X.payload.embeddings) == freeze
    probs = model.predict_proba(rep.transform(toy["dev"]))
    assert probs.shape == (len(toy["dev"]), 3) and np.allclose(probs.sum(axis=1), 1)
    assert 1 <= len(model.history) <= SMALL["num_epochs"]

    model.save(tmp_path / "m")
    loaded = BowmanModel.load(tmp_path / "m")
    assert np.allclose(loaded.predict_proba(rep.transform(toy["dev"])), probs)


@pytest.mark.parametrize("encoder", ["sum", "lstm", "gru"])
def test_runs_through_the_fold_pipeline(glove_file, tmp_path, encoder):
    cfg = copy.deepcopy(EXAMPLE_CONFIG)
    cfg["name"] = f"toy-glove-bowman-{encoder}"
    cfg["tokenizer"] = {"kind": "treebank"}
    cfg["representation"] = {"kind": "glove", "path": str(glove_file)}
    cfg["model"] = {"kind": "bowman", "encoder": encoder, **SMALL}
    result = run_experiment(cfg, tmp_path, verbose=False)
    assert len(result.per_fold("accuracy")) == cfg["dataset"]["k"]
