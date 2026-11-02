"""Bag of words: token counts (optionally normalised) over the Tokenizer's vocabulary. OWNER: Emil.

The tutorial left ``BagOfWords`` as an empty stub. Hyperparameters worth exposing:
``binary``, ``normalize`` (counts vs. frequencies), ``min_df``.
"""
from __future__ import annotations

import scipy.sparse as sp

from .base import VectorRepresentation


class BagOfWordsRepresentation(VectorRepresentation):
    kind = "bow"

    def __init__(self, tokenizer, pairing: str = "concat", normalize: bool = False):
        super().__init__(tokenizer, pairing)
        self.normalize = normalize

    def fit_texts(self, texts: list[str]) -> None:
        raise NotImplementedError("Emil: nothing to fit if using tokenizer.vocab; else learn a vocab here")

    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        raise NotImplementedError("Emil: count matrix (len(texts), side_dim)")

    @property
    def side_dim(self) -> int:
        raise NotImplementedError("Emil")
