"""N-gram counts over Tokenizer tokens. OWNER: Neel.

``fit_texts`` must learn the n-gram vocabulary from the training fold only (ADR-0001).
``sklearn.feature_extraction.text.CountVectorizer(analyzer=callable)`` with a callable
that returns token n-grams from ``self.tokenizer.tokenize`` is the shortest route and
gives ``min_df``, ``binary`` and ``max_features`` for free. Consider a ``tfidf`` flag.
"""
from __future__ import annotations

import scipy.sparse as sp

from .base import VectorRepresentation


class NgramRepresentation(VectorRepresentation):
    kind = "ngram"

    def __init__(self, tokenizer, pairing: str = "concat", n: int = 2, min_df: int = 1,
                 binary: bool = False, max_features: int | None = None):
        super().__init__(tokenizer, pairing)
        self.n = n
        self.min_df = min_df
        self.binary = binary
        self.max_features = max_features
        self._vectorizer = None

    def fit_texts(self, texts: list[str]) -> None:
        raise NotImplementedError("Neel: fit a CountVectorizer over 1..n-grams of tokens")

    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        raise NotImplementedError("Neel: self._vectorizer.transform(texts)")

    @property
    def side_dim(self) -> int:
        raise NotImplementedError("Neel: len(self._vectorizer.vocabulary_)")
