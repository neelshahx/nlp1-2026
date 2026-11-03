"""Representation interface: a Pair batch in, Features out.

A Representation receives both premise and hypothesis and decides how to combine them.
Its vocabulary (if any) is fitted per fold on the training portion only (ADR-0001).
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import scipy.sparse as sp

from ..tokenizers import Tokenizer

PAIRINGS = ("concat", "joint", "hypothesis_only")
SEP = " [SEP] "


class Features:
    """Output of a Representation for a batch of Pairs.

    ``payload`` is today a ``scipy.sparse.csr_matrix`` of shape ``(n_rows, dim)``. Any
    row-indexable object is allowed, so a logic-based Representation may hand the Pairs
    DataFrame straight through with ``dim=None``.
    """

    def __init__(self, payload: Any, n_rows: int | None = None, dim: int | None = None):
        self.payload = payload
        self.n_rows = n_rows if n_rows is not None else payload.shape[0]
        self.dim = dim if dim is not None else getattr(payload, "shape", (None, None))[1]

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        return f"Features(n_rows={self.n_rows}, dim={self.dim}, payload={type(self.payload).__name__})"


def sides(pairs: pd.DataFrame, pairing: str) -> list[pd.Series]:
    """The text columns a vector Representation featurises, one entry per feature block."""
    if pairing == "concat":
        return [pairs["premise"], pairs["hypothesis"]]
    if pairing == "joint":
        return [pairs["premise"] + SEP + pairs["hypothesis"]]
    if pairing == "hypothesis_only":
        return [pairs["hypothesis"]]
    raise ValueError(f"unknown pairing {pairing!r}; choose from {PAIRINGS}")


class Representation(ABC):
    kind: str  # the value of config["representation"]["kind"]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def fit(self, pairs: pd.DataFrame) -> "Representation":
        """Learn any vocabulary from the fold's training Pairs only. Return self."""

    @abstractmethod
    def transform(self, pairs: pd.DataFrame) -> Features:
        """Featurise a batch of Pairs."""

    @property
    def dim(self) -> int | None:
        return None

    def fit_transform(self, pairs: pd.DataFrame) -> Features:
        return self.fit(pairs).transform(pairs)


class VectorRepresentation(Representation):
    """Base for Representations that produce one sparse block per side and hstack them.

    Subclasses implement the one-sided ``fit_texts`` / ``transform_texts`` / ``side_dim``;
    this class handles ``pairing``. With ``concat`` the vocabulary is fitted on both sides'
    texts pooled, but premise and hypothesis get separate feature blocks. Override
    ``fit``/``transform`` directly if your Representation doesn't fit this shape.
    """

    def __init__(self, tokenizer: Tokenizer, pairing: str = "concat"):
        super().__init__(tokenizer)
        if pairing not in PAIRINGS:
            raise ValueError(f"unknown pairing {pairing!r}; choose from {PAIRINGS}")
        self.pairing = pairing

    def fit(self, pairs: pd.DataFrame) -> "VectorRepresentation":
        texts = [t for side in sides(pairs, self.pairing) for t in side.tolist()]
        self.fit_texts(texts)
        return self

    def transform(self, pairs: pd.DataFrame) -> Features:
        blocks = [self.transform_texts(side.tolist()) for side in sides(pairs, self.pairing)]
        return Features(sp.hstack(blocks, format="csr"), dim=self.dim)

    @property
    def dim(self) -> int:
        return self.side_dim * len(sides(pd.DataFrame({"premise": [""], "hypothesis": [""]}), self.pairing))

    @abstractmethod
    def fit_texts(self, texts: list[str]) -> None:
        """Learn the vocabulary for one side from a list of texts."""

    @abstractmethod
    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        """``(len(texts), side_dim)`` sparse matrix."""

    @property
    @abstractmethod
    def side_dim(self) -> int:
        """Width of one side's block."""
