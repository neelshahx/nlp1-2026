"""Bag of words: token counts over the Tokenizer's vocabulary, optionally as 1/0 or as frequencies. OWNER: Emil."""

import numpy as np
import scipy.sparse as sp

from .base import VectorRepresentation, token_count_matrix


class BagOfWordsRepresentation(VectorRepresentation):
    kind = "bow"

    def __init__(self, tokenizer, pairing: str = "concat", binary: bool = False, normalize: bool = False):
        """``binary`` records presence (1/0) instead of counts; ``normalize`` divides each row so it sums to 1."""
        super().__init__(tokenizer, pairing)
        assert not (binary and normalize), "binary and normalize are exclusive"
        self.binary = binary
        self.normalize = normalize

    def fit_texts(self, texts: list[str]) -> None:
        """Nothing to learn: the vocabulary is the tokenizer's."""

    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        """How many times each token appears in each text (or 1/0, or its share of the text)."""
        X = token_count_matrix(self.tokenizer, texts, binary=self.binary)
        if self.normalize:
            totals = np.asarray(X.sum(axis=1)).ravel()
            totals[totals == 0] = 1  # empty texts stay all-zero instead of dividing by zero
            X = sp.csr_matrix(sp.diags(1 / totals) @ X, dtype=np.float32)
        return X

    @property
    def side_dim(self) -> int:
        return len(self.tokenizer.vocab)
