"""Frozen BERT sentence embeddings (mean pooling over real tokens). Extracted from emil/bert-encoder.py.

The encoder has its own WordPiece tokenizer (use the placeholder ``wordpiece`` tokenizer kind in the config).
Premise and hypothesis are encoded separately, never as one ``[CLS] P [SEP] H [SEP]`` input, so BERT stays a
sentence encoder: the two sentences only meet in the classifier.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..tokenizers import Tokenizer
from .base import Features, Representation

BERT_PAIRINGS = ("concat", "matching", "hypothesis_only", "sentences")


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BertEncoder:
    """Sentences -> ``(n, hidden_size)`` float16 mean-pooled last-layer embeddings."""

    def __init__(
        self, model_name: str = "bert-base-uncased", max_length: int = 64, batch_size: int = 256, device: str = "auto"
    ):
        from transformers import AutoModel, AutoTokenizer  # slow import; only pay for it when used

        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length  # SNLI sentences are short; 64 tokens covers almost all
        self.batch_size = batch_size
        self.dim: int = self.model.config.hidden_size  # 768 for bert-base

    @torch.no_grad()
    def _encode_batch(self, sentences: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        # fp16 on GPU is ~2x faster; on CPU/MPS stay in fp32
        with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
            out = self.model(**enc).last_hidden_state  # (batch, tokens, hidden)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.float() * mask).sum(dim=1)
        return (summed / mask.sum(dim=1).clamp(min=1e-9)).cpu().numpy()

    def transform(self, sentences: list[str]) -> np.ndarray:
        """Duplicates are encoded once; batches are length-sorted to cut padding."""
        unique = list(dict.fromkeys(sentences))
        index = {s: i for i, s in enumerate(unique)}
        order = sorted(range(len(unique)), key=lambda i: len(unique[i]))
        emb = np.zeros((len(unique), self.dim), dtype=np.float16)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            emb[idx] = self._encode_batch([unique[i] for i in idx])
        return emb[[index[s] for s in sentences]]


@dataclass
class SentenceVectorPairs:
    """Premise and hypothesis sentence vectors, ``(n, hidden)`` each, kept apart for a model that combines them."""

    premise: np.ndarray
    hypothesis: np.ndarray

    def __len__(self) -> int:
        return len(self.premise)

    def take(self, idx) -> "SentenceVectorPairs":
        return SentenceVectorPairs(self.premise[idx], self.hypothesis[idx])


def matching_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``[u, v, |u - v|, u * v]`` (Mou et al. 2016 heuristic matching), float32."""
    u, v = u.astype(np.float32), v.astype(np.float32)
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=1)


# (model_name, max_length) -> encoder, and -> {sentence: vector}; shared by every BertRepresentation in the process
_ENCODERS: dict[tuple[str, int], BertEncoder] = {}
_MEMO: dict[tuple[str, int], dict[str, np.ndarray]] = {}


class BertRepresentation(Representation):
    kind = "bert"

    def __init__(
        self,
        tokenizer: Tokenizer,
        pairing: str = "matching",
        model_name: str = "bert-base-uncased",
        max_length: int = 64,
        batch_size: int = 256,
        device: str = "auto",
    ):
        """``pairing``: ``concat`` = [u, v], ``matching`` = [u, v, |u-v|, u*v], ``hypothesis_only`` = v (dense
        matrices for feature models), or ``sentences`` = u and v kept apart (``SentenceVectorPairs``, for
        ``bowman_bert``, which projects and combines them itself)."""
        super().__init__(tokenizer)
        if pairing not in BERT_PAIRINGS:
            raise ValueError(f"unknown pairing {pairing!r}; choose from {BERT_PAIRINGS}")
        self.pairing = pairing
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    @property
    def _key(self) -> tuple[str, int]:
        return (self.model_name, self.max_length)

    def _encoder(self) -> BertEncoder:
        if self._key not in _ENCODERS:
            _ENCODERS[self._key] = BertEncoder(self.model_name, self.max_length, self.batch_size, self.device)
        return _ENCODERS[self._key]

    def encode(self, sentences: list[str]) -> np.ndarray:
        """``(n, hidden)`` float16, reusing vectors already computed in this process."""
        memo = _MEMO.setdefault(self._key, {})
        todo = list(dict.fromkeys(s for s in sentences if s not in memo))
        if todo:
            memo.update(zip(todo, self._encoder().transform(todo), strict=True))
        return np.stack([memo[s] for s in sentences])

    def fit(self, pairs: pd.DataFrame) -> "BertRepresentation":
        return self  # frozen encoder: nothing to learn

    def transform(self, pairs: pd.DataFrame) -> Features:
        v = self.encode(pairs["hypothesis"].tolist())
        if self.pairing == "hypothesis_only":
            return Features(v.astype(np.float32))
        u = self.encode(pairs["premise"].tolist())
        if self.pairing == "sentences":
            return Features(SentenceVectorPairs(u, v), n_rows=len(pairs), dim=None)
        X = matching_features(u, v) if self.pairing == "matching" else np.concatenate([u, v], axis=1, dtype=np.float32)
        return Features(X)

    @property
    def dim(self) -> int | None:
        return None  # hidden size is only known once the encoder is loaded; Features carries the real width
