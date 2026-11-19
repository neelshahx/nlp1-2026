"""Pretrained word vectors (GloVe 6B: uncased, Wikipedia + Gigaword). Files live in ``data/`` (gitignored)."""

from functools import cache
from pathlib import Path

import numpy as np
import torch

GLOVE_DIR = Path(__file__).resolve().parents[1] / "data" / "glove.6B"
GLOVE_300D = GLOVE_DIR / "glove.6B.300d.txt"


def glove_path(dim: int = 300) -> Path:
    """``data/glove.6B/glove.6B.<dim>d.txt``; 6B ships 50, 100, 200 and 300."""
    return GLOVE_DIR / f"glove.6B.{dim}d.txt"


@cache
def load_glove(path: str | Path = GLOVE_300D) -> tuple[dict[str, int], np.ndarray]:
    """``(word -> row, (n_words, dim) float32)``; parsed once per process (~1 min, ~0.5 GB for 300d)."""
    words, rows = [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            word, rest = line.rstrip("\n").split(" ", 1)
            words.append(word)
            rows.append(np.array(rest.split(), dtype=np.float32))
    return {w: i for i, w in enumerate(words)}, np.stack(rows)


def glove_words(path: str | Path = GLOVE_300D) -> set[str]:
    """Every word that has a vector in the GloVe file."""
    return set(load_glove(path)[0])


def glove_matrix(vocab: list[str], path: str | Path = GLOVE_300D, seed: int = 0, pad_idx: int = 0):
    """``(len(vocab), dim)`` float tensor, row i = GloVe vector of ``vocab[i]``, plus the share of vocab found.

    Words without a vector get small random N(0, 0.1) vectors; row ``pad_idx`` is zero.
    The coverage ignores the first two entries, assumed to be the pad and unk specials.
    """
    index, table = load_glove(path)
    rng = np.random.default_rng(seed)
    vectors = rng.normal(0, 0.1, (len(vocab), table.shape[1])).astype(np.float32)
    found = 0
    for i, w in enumerate(vocab):
        if w in index:
            vectors[i] = table[index[w]]
            found += 1
    vectors[pad_idx] = 0.0
    return torch.from_numpy(vectors), found / max(len(vocab) - 2, 1)
