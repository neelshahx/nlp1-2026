import pytest

from nli import data, eda, tokenizers


@pytest.fixture(scope="module")
def toy():
    return data.load_toy()


@pytest.fixture(scope="module")
def tokenizer(toy):
    return tokenizers.WordLevelTokenizer().train(data.texts(toy["train"]))


def test_cue_words_recover_toy_templates(toy, tokenizer):
    table = eda.hypothesis_cue_words(toy["train"], tokenizer, top_n=3, min_count=4)
    by_label = {label: set(group["token"]) for label, group in table.groupby("label", observed=True)}
    assert by_label["contradiction"] <= {"sleeping", "sitting", "still", "lying", "down"}
    assert by_label["neutral"] <= {"for", "a", "competition", "because", "it", "is", "sunny", "with", "friend"}
    assert (table.loc[table["label"] != "entailment", "pmi"] > 0).all()
    assert (table.loc[table["label"] == "entailment", "pmi"] <= 0).all()
    assert table.groupby("label", observed=True).size().max() <= 3


def test_cue_words_respect_count_floor(toy, tokenizer):
    assert eda.hypothesis_cue_words(toy["train"], tokenizer, min_count=100).empty
