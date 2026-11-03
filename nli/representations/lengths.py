"""Token-count-per-side baseline. Exists so the pipeline runs end to end before the real
Representations land, and doubles as the length-artifact baseline (Gururangan et al. 2018)."""

import numpy as np
import scipy.sparse as sp

from .base import VectorRepresentation


class LengthsRepresentation(VectorRepresentation):
    kind = "lengths"

    def fit_texts(self, texts: list[str]) -> None:
        pass

    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        counts = np.array([[len(self.tokenizer.tokenize(t))] for t in texts], dtype=float)
        return sp.csr_matrix(counts)

    @property
    def side_dim(self) -> int:
        return 1
