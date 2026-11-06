"""Representation interface: a batch of premise/hypothesis pairs in, Features out."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..tokenizers import Tokenizer

PAIRINGS = ("concat", "joint", "hypothesis_only")
SEP = " [SEP] "


class Features:
    """Output of a Representation for a batch of pairs: a row-indexable ``payload`` plus its shape."""

    def __init__(self, payload: Any, n_rows: int | None = None, dim: int | None = None):
        """``payload`` is usually a sparse ``(n_rows, dim)`` matrix; pass ``dim=None`` for non-matrix payloads."""
        self.payload = payload
        self.n_rows = n_rows if n_rows is not None else payload.shape[0]
        self.dim = dim if dim is not None else getattr(payload, "shape", (None, None))[1]

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        return f"Features(n_rows={self.n_rows}, dim={self.dim}, payload={type(self.payload).__name__})"


def sides(pairs: pd.DataFrame, pairing: str) -> list[pd.Series]:
    """Pick which text column(s) to featurise, one entry per feature block; ``joint`` merges both with a separator."""
    if pairing == "concat":
        return [pairs["premise"], pairs["hypothesis"]]
    if pairing == "joint":
        return [pairs["premise"] + SEP + pairs["hypothesis"]]
    if pairing == "hypothesis_only":
        return [pairs["hypothesis"]]
    raise ValueError(f"unknown pairing {pairing!r}; choose from {PAIRINGS}")


def token_count_matrix(tokenizer: Tokenizer, texts: list[str], binary: bool = False) -> sp.csr_matrix:
    """One row per text, one column per ``tokenizer.vocab`` entry; tokens outside the vocab are dropped."""
    vocab = tokenizer.vocab
    rows, cols = [], []
    for i, text in enumerate(texts):
        for tok in tokenizer.tokenize(text):
            if tok in vocab:
                rows.append(i)
                cols.append(vocab[tok])
    ones = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((ones, (rows, cols)), shape=(len(texts), len(vocab)))  # duplicates sum to counts
    if binary:
        X.data[:] = 1
    return X


class Representation(ABC):
    """Turns premise/hypothesis pairs into Features; decides how the two sides are combined."""

    kind: str  # the value of config["representation"]["kind"]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def fit(self, pairs: pd.DataFrame) -> "Representation":
        """Learn any vocabulary from the training pairs of one fold only. Return self."""

    @abstractmethod
    def transform(self, pairs: pd.DataFrame) -> Features:
        """Featurise a batch of pairs."""

    @property
    def dim(self) -> int | None:
        """Number of features per pair, or ``None`` if the output isn't a fixed-width vector."""
        return None

    def fit_transform(self, pairs: pd.DataFrame) -> Features:
        """``fit`` then ``transform`` on the same pairs."""
        return self.fit(pairs).transform(pairs)


class VectorRepresentation(Representation):
    """Base for Representations that turn each text into a sparse vector; subclasses only handle one text at a time."""

    def __init__(self, tokenizer: Tokenizer, pairing: str = "concat"):
        """``pairing`` chooses which sides are featurised and how (see ``sides``)."""
        super().__init__(tokenizer)
        if pairing not in PAIRINGS:
            raise ValueError(f"unknown pairing {pairing!r}; choose from {PAIRINGS}")
        self.pairing = pairing

    def fit(self, pairs: pd.DataFrame) -> "VectorRepresentation":
        """Learn one shared vocabulary from all selected sides' texts pooled together."""
        texts = [t for side in sides(pairs, self.pairing) for t in side.tolist()]
        self.fit_texts(texts)
        return self

    def transform(self, pairs: pd.DataFrame) -> Features:
        """Vectorise each selected side separately, then place the blocks side by side."""
        blocks = [self.transform_texts(side.tolist()) for side in sides(pairs, self.pairing)]
        return Features(sp.hstack(blocks, format="csr"), dim=self.dim)

    @property
    def dim(self) -> int:
        """``side_dim`` times the number of blocks the pairing produces."""
        return self.side_dim * len(sides(pd.DataFrame({"premise": [""], "hypothesis": [""]}), self.pairing))

    @abstractmethod
    def fit_texts(self, texts: list[str]) -> None:
        """Learn the vocabulary from a list of texts."""

    @abstractmethod
    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        """One sparse row per text, ``side_dim`` columns."""

    @property
    @abstractmethod
    def side_dim(self) -> int:
        """Number of features per single text."""
