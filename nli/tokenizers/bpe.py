"""BPE tokenizer via HuggingFace ``tokenizers``. OWNER: Neel.

Starting point: ``nlp1-student/part1/experiment.py:32`` ``train_tokenizer`` (BPE model,
``BpeTrainer``, ``Metaspace`` pre-tokenizer, ``[UNK]``). ``tokenizers.Tokenizer`` already
has ``save``/``from_file``; wrap them. Hyperparameters to expose: ``vocab_size``,
``min_frequency``, and the pre-tokenizer choice (see the HF tokenizer summary linked in
the project notes for alternatives such as WordPiece).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import Tokenizer


class BPETokenizer(Tokenizer):
    kind = "bpe"

    def __init__(self, vocab_size: int = 8000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self._tok = None  # tokenizers.Tokenizer once trained

    def train(self, texts: Iterable[str]) -> "BPETokenizer":
        raise NotImplementedError("Neel: train a tokenizers.Tokenizer(BPE) from texts")

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Neel: return self._tok.encode(text).tokens")

    @property
    def vocab(self) -> dict[str, int]:
        raise NotImplementedError("Neel: return self._tok.get_vocab()")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Neel: self._tok.save(str(path) + '.json')")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        raise NotImplementedError("Neel: tokenizers.Tokenizer.from_file(...)")
