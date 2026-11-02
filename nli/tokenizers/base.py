"""Tokenizer interface. A Tokenizer segments one text into tokens; it knows nothing about pairs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class Tokenizer(ABC):
    """Trained once per Dataset on the train split's premises and hypotheses (ADR-0001).

    Implementations must be deterministic given their hyperparameters and training texts,
    because the cached file is keyed by the config hash, not by the trained vocabulary.
    """

    kind: str  # the value of config["tokenizer"]["kind"]

    @abstractmethod
    def train(self, texts: Iterable[str]) -> "Tokenizer":
        """Learn the vocabulary. Return self. No-op for untrainable tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Segment one text into tokens."""

    @property
    @abstractmethod
    def vocab(self) -> dict[str, int]:
        """Token -> id. Empty before training."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write to ``path`` (a file path without extension; add your own)."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Tokenizer":
        """Inverse of ``save``."""

    def tokenize_many(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]
