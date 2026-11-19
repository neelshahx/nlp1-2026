"""GloVe token-id sequences: input to sequence Models (``bowman``) that embed, encode and classify themselves.

``fit`` learns a vocabulary from the fold's training pairs only (Bowman et al. 2015 use a train-only vocab) and
lines it up with GloVe; words outside it map to ``<unk>``. The embedding matrix travels with the Features so
the Model can initialise (and optionally freeze) its embedding layer from it.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..embeddings import glove_matrix, glove_path
from ..tokenizers import Tokenizer
from .base import Features, Representation

PAD, UNK_TOKEN = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1


@dataclass
class SequencePairs:
    """Premise and hypothesis token ids per pair, plus the ``(vocab, dim)`` embedding matrix they index."""

    premise: list[list[int]]
    hypothesis: list[list[int]]
    embeddings: torch.Tensor

    def __len__(self) -> int:
        return len(self.premise)

    def take(self, idx) -> "SequencePairs":
        return SequencePairs([self.premise[i] for i in idx], [self.hypothesis[i] for i in idx], self.embeddings)


class GloveRepresentation(Representation):
    kind = "glove"

    def __init__(self, tokenizer: Tokenizer, dim: int = 300, min_frequency: int = 1, path: str | None = None):
        """``dim`` picks ``data/glove.6B/glove.6B.<dim>d.txt``; ``path`` overrides the file (tests)."""
        super().__init__(tokenizer)
        self.glove_dim = dim
        self.min_frequency = min_frequency
        self.path = path or str(glove_path(dim))
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}
        self.embeddings: torch.Tensor | None = None
        self.stats: dict[str, float] = {}

    def _token_lists(self, pairs: pd.DataFrame):
        return [self.tokenizer.tokenize(t) for t in pairs["premise"]], [
            self.tokenizer.tokenize(t) for t in pairs["hypothesis"]
        ]

    def fit(self, pairs: pd.DataFrame) -> "GloveRepresentation":
        p, h = self._token_lists(pairs)
        counts = Counter(t for toks in p + h for t in toks)
        self.vocab = [PAD, UNK_TOKEN] + sorted(w for w, c in counts.items() if c >= self.min_frequency)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.embeddings, type_coverage = glove_matrix(self.vocab, self.path, seed=0, pad_idx=PAD_IDX)
        self.stats = {"vocab_size": len(self.vocab), "glove_type_coverage": type_coverage}
        return self

    def _ids(self, token_lists: list[list[str]]) -> list[list[int]]:
        # an empty sentence becomes a single <unk> so packed LSTMs never see length 0
        return [[self.index.get(t, UNK_IDX) for t in toks] or [UNK_IDX] for toks in token_lists]

    def transform(self, pairs: pd.DataFrame) -> Features:
        assert self.embeddings is not None, "run fit first"
        p, h = self._token_lists(pairs)
        return Features(SequencePairs(self._ids(p), self._ids(h), self.embeddings), n_rows=len(pairs), dim=None)

    def unk_rate(self, pairs: pd.DataFrame) -> float:
        """Share of tokens in ``pairs`` that fall outside the fitted vocabulary."""
        p, h = self._token_lists(pairs)
        toks = [t for s in p + h for t in s]
        return float(np.mean([t not in self.index for t in toks])) if toks else 0.0
