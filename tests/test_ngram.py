import pytest

from nli import data, representations, tokenizers
from nli.experiment import EXAMPLE_CONFIG, run_experiment


@pytest.fixture(scope="module")
def toy():
    return data.load_toy()


@pytest.fixture(scope="module")
def tok(toy):
    train = toy["train"]
    return tokenizers.BPETokenizer(vocab_size=1000, min_frequency=2).train(
        train["premise"].tolist() + train["hypothesis"].tolist()
    )


def test_n_gram_finder_lists_every_run_up_to_n(tok):
    rep = representations.NgramRepresentation(tok, n=2)
    tokens = tok.tokenize("a man sleeps")
    grams = rep.n_gram_finder("a man sleeps")
    assert grams[: len(tokens)] == tokens
    assert grams[len(tokens) :] == [" ".join(pair) for pair in zip(tokens, tokens[1:], strict=False)]


def test_unseen_ngrams_are_ignored(toy, tok):
    train = toy["train"]
    rep = representations.NgramRepresentation(tok, n=2)
    rep.fit_texts(train["premise"].tolist() + train["hypothesis"].tolist())
    X = rep.transform_texts([train["premise"][0], "Zebras juggle quantum xylophones."])
    assert X.shape == (2, rep.side_dim)
    assert X[0].nnz > 0


def test_pairing_sets_dim(toy, tok):
    train, dev = toy["train"], toy["dev"]
    concat = representations.build_representation({"kind": "ngram", "n": 2, "pairing": "concat"}, tok).fit(train)
    hyp = representations.build_representation({"kind": "ngram", "n": 2, "pairing": "hypothesis_only"}, tok).fit(train)
    assert concat.dim == 2 * concat.side_dim
    assert hyp.dim == hyp.side_dim
    assert concat.transform(dev).payload.shape == (len(dev), concat.dim)
    assert hyp.transform(dev).payload.shape == (len(dev), hyp.dim)


def test_binary_gives_presence_not_counts(toy, tok):
    train = toy["train"]
    counts = representations.NgramRepresentation(tok, n=1).fit(train)
    present = representations.NgramRepresentation(tok, n=1, binary=True).fit(train)
    repeated = "A man and a man."
    assert counts.transform_texts([repeated]).max() > 1
    assert present.transform_texts([repeated]).max() == 1


def test_ngram_majority_smoke(tmp_path):
    config = {
        **EXAMPLE_CONFIG,
        "name": "toy-ngram-majority",
        "tokenizer": {"kind": "bpe", "vocab_size": 1000},
        "representation": {"kind": "ngram", "n": 2, "pairing": "concat"},
    }
    result = run_experiment(config, root=tmp_path, verbose=False)
    assert len(result.scores) == 3 * 2
