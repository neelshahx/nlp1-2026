"""Placeholder for BERT's own WordPiece tokenizer, so a ``bert`` Experiment's id names the tokenizer it really uses.

The ``bert`` Representation tokenizes inside the pretrained encoder (Hugging Face ``AutoTokenizer``, which is WordPiece
for ``bert-base-uncased``). The runner still requires a ``tokenizer`` block and trains it once, so this kind does
nothing: training, saving and loading are no-ops, and ``tokenize`` refuses so it can't be used by mistake.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from .base import Tokenizer


class WordPieceTokenizer(Tokenizer):
    kind = "wordpiece"

    def train(self, texts: Iterable[str]) -> "WordPieceTokenizer":
        return self

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("wordpiece is a placeholder: the bert Representation tokenizes inside BERT")

    @property
    def vocab(self) -> dict[str, int]:
        return {}

    def save(self, path: Path) -> None:
        Path(str(path) + ".json").write_text(json.dumps({"kind": self.kind}))

    @classmethod
    def load(cls, path: Path) -> "WordPieceTokenizer":
        return cls()
