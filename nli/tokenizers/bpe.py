"""BPE tokenizer via HuggingFace ``tokenizers``"""

import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer as hf_Tokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

from .base import Tokenizer


class BPETokenizer(Tokenizer):
    kind = "bpe"

    def __init__(self, vocab_size: int = 8000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self._tok = None  # tokenizers.Tokenizer once trained

    def train(self, texts: Iterable[str]) -> "BPETokenizer":
        tok = hf_Tokenizer(models.BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
        tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.decoder = decoders.BPEDecoder(suffix="</w>")
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["[PAD]", "[UNK]"],
            end_of_word_suffix="</w>",
        )
        tok.train_from_iterator(texts, trainer=trainer)
        self._tok = tok
        return self

    def tokenize(self, text: str) -> list[str]:
        assert self._tok is not None, "train the tokenizer first"
        return self._tok.encode(text).tokens

    @property
    def vocab(self) -> dict[str, int]:
        assert self._tok is not None, "train the tokenizer first"
        return self._tok.get_vocab()

    def save(self, path: Path) -> None:
        self._tok.save(str(Path(str(path) + "_hf_bpe.json")))
        Path(str(path) + ".json").write_text(
            json.dumps(
                {
                    "vocab_size": self.vocab_size,
                    "min_frequency": self.min_frequency,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        params = json.loads(Path(str(path) + ".json").read_text())
        tok = cls(params["vocab_size"], params["min_frequency"])
        tok._tok = hf_Tokenizer.from_file(str(Path(str(path) + "_hf_bpe.json")))
        return tok
