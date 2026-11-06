import numpy as np
import pytest

from nli import data, representations, tokenizers
from nli.experiment import EXAMPLE_CONFIG, run_experiment


@pytest.fixture(scope="module")
def toy():
    return data.load_toy()


@pytest.fixture(scope="module")
def tok(toy):
    train = toy["train"]
    return tokenizers.WordLevelTokenizer(min_frequency=1).train(
        train["premise"].tolist() + train["hypothesis"].tolist()
    )


TEXT = "A man and a man."


def test_counts_repeats(tok):
    rep = representations.BagOfWordsRepresentation(tok)
    X = rep.transform_texts([TEXT])
    assert X.shape == (1, rep.side_dim) and rep.side_dim == len(tok.vocab)
    assert set(X.indices.tolist()) == {tok.vocab[t] for t in tok.tokenize(TEXT)}
    assert X.max() == 2 and X.sum() == len(tok.tokenize(TEXT))


def test_binary_marks_presence(tok):
    X = representations.BagOfWordsRepresentation(tok, binary=True).transform_texts([TEXT])
    assert X.max() == 1 and X.nnz == len(set(tok.tokenize(TEXT)))


def test_normalize_rows_sum_to_one(tok):
    counts = representations.BagOfWordsRepresentation(tok).transform_texts([TEXT])
    freqs = representations.BagOfWordsRepresentation(tok, normalize=True).transform_texts([TEXT])
    assert np.isclose(freqs.sum(), 1)
    assert set(freqs.indices.tolist()) == set(counts.indices.tolist())
    assert representations.BagOfWordsRepresentation(tok, normalize=True).transform_texts([""]).nnz == 0


def test_pairing_dims(toy, tok):
    train, dev = toy["train"], toy["dev"]
    concat = representations.build_representation({"kind": "bow", "pairing": "concat"}, tok).fit(train)
    hyp = representations.build_representation({"kind": "bow", "pairing": "hypothesis_only"}, tok).fit(train)
    assert concat.dim == 2 * len(tok.vocab) and hyp.dim == len(tok.vocab)
    assert concat.transform(dev).payload.shape == (len(dev), concat.dim)


@pytest.mark.parametrize("extra", [{}, {"binary": True}])
def test_majority_smoke(tmp_path, extra):
    config = {**EXAMPLE_CONFIG, "name": "toy-bow-majority", "representation": {"kind": "bow", **extra}}
    result = run_experiment(config, root=tmp_path, verbose=False)
    assert len(result.scores) == 3 * 2
