"""Penn Treebank word tokenizer (NLTK), lowercased: the segmentation GloVe 6B and Bowman et al. (2015) assume.

``tokenize`` always returns the raw tokens (no ``[UNK]`` mapping), so a GloVe Representation can still look up
words that are rare in train. ``train`` records a vocabulary only so count-based Representations work too.
"""

import json
from collections import Counter
from collections.abc import Iterable
from functools import cache
from pathlib import Path

from nltk.tokenize import TreebankWordTokenizer

from .base import UNK, Tokenizer

_PTB = TreebankWordTokenizer()


@cache
def _tokenize(text: str, lowercase: bool) -> tuple[str, ...]:
    # memoised: SNLI repeats each premise ~3 times and every fold re-tokenizes the same train rows
    return tuple(_PTB.tokenize(text.lower() if lowercase else text))


class TreebankTokenizer(Tokenizer):
    kind = "treebank"

    def __init__(self, lowercase: bool = True, min_frequency: int = 1):
        self.lowercase = lowercase
        self.min_frequency = min_frequency
        self._vocab: dict[str, int] = {}

    def train(self, texts: Iterable[str]) -> "TreebankTokenizer":
        counts = Counter(tok for t in texts for tok in self.tokenize(t))
        kept = sorted(tok for tok, c in counts.items() if c >= self.min_frequency)
        self._vocab = {UNK: 0, **{tok: i + 1 for i, tok in enumerate(kept)}}
        return self

    def tokenize(self, text: str) -> list[str]:
        return list(_tokenize(text, self.lowercase))

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    def save(self, path: Path) -> None:
        meta = {"lowercase": self.lowercase, "min_frequency": self.min_frequency, "vocab": self._vocab}
        Path(str(path) + ".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> "TreebankTokenizer":
        d = json.loads(Path(str(path) + ".json").read_text())
        tok = cls(d["lowercase"], d["min_frequency"])
        tok._vocab = d["vocab"]
        return tok
