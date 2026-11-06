from nli import data, tokenizers
from nli.experiment import EXAMPLE_CONFIG, run_experiment


def _texts():
    train = data.load_toy()["train"]
    return (train["premise"] + " " + train["hypothesis"]).tolist()


def test_bpe_save_load_round_trip(tmp_path):
    tok = tokenizers.BPETokenizer(vocab_size=50, min_frequency=2).train(_texts())
    sentence = _texts()[0]
    before = tok.tokenize(sentence)
    tok.save(tmp_path / "bpe")

    loaded = tokenizers.BPETokenizer.load(tmp_path / "bpe")
    assert loaded.tokenize(sentence) == before
    assert loaded.vocab == tok.vocab
    assert (loaded.vocab_size, loaded.min_frequency) == (50, 2)


def test_build_tokenizer_from_config():
    tok = tokenizers.build_tokenizer({"kind": "bpe", "min_frequency": 2, "vocab_size": 20})
    tok.train(_texts())
    tokens = tok.tokenize_many(_texts()[:3])
    assert len(tokens) == 3 and all(isinstance(t, str) for row in tokens for t in row)
    assert (tok.vocab_size, tok.min_frequency) == (20, 2)


def test_bpe_experiment_caches_tokenizer(tmp_path):
    config = {**EXAMPLE_CONFIG, "name": "toy-bpe", "tokenizer": {"kind": "bpe", "vocab_size": 1000}}
    first = run_experiment(config, root=tmp_path, verbose=False)
    cache = tmp_path / "models" / "tokenizers"
    files = sorted(p.name for p in cache.iterdir())
    assert files, "first run should write the trained tokenizer"

    again = run_experiment(config, root=tmp_path, verbose=False)
    assert again.id == first.id
    assert sorted(p.name for p in cache.iterdir()) == files
