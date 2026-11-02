"""Word-level baseline tokenizer: lowercase, split on words and punctuation."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .base import Tokenizer

_TOKEN = re.compile(r"\w+|[^\w\s]")
UNK = "[UNK]"


class WordLevelTokenizer(Tokenizer):
    kind = "wordlevel"

    def __init__(self, min_frequency: int = 1, lowercase: bool = True):
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self._vocab: dict[str, int] = {}

    def _split(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return _TOKEN.findall(text)

    def train(self, texts: Iterable[str]) -> "WordLevelTokenizer":
        counts = Counter(tok for t in texts for tok in self._split(t))
        kept = sorted(tok for tok, c in counts.items() if c >= self.min_frequency)
        self._vocab = {UNK: 0, **{tok: i + 1 for i, tok in enumerate(kept)}}
        return self

    def tokenize(self, text: str) -> list[str]:
        toks = self._split(text)
        if not self._vocab:
            return toks
        return [t if t in self._vocab else UNK for t in toks]

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    def save(self, path: Path) -> None:
        Path(str(path) + ".json").write_text(json.dumps({
            "min_frequency": self.min_frequency, "lowercase": self.lowercase, "vocab": self._vocab}))

    @classmethod
    def load(cls, path: Path) -> "WordLevelTokenizer":
        d = json.loads(Path(str(path) + ".json").read_text())
        tok = cls(d["min_frequency"], d["lowercase"])
        tok._vocab = d["vocab"]
        return tok
