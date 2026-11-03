"""Token indicators: a binary vector over the Tokenizer's vocabulary. OWNER: Emil.

Starting point: ``nlp1-student/part1/clslib.py:172`` ``TokenIndicator``. KNOWN ISSUES
there: ``__len__`` reads a global ``tokenizer`` instead of ``self.tokenizer``; it returns a
dense row per text which callers then convert to CSR. Build the CSR matrix directly.
The vocabulary is the Tokenizer's, so ``fit_texts`` is a no-op.
"""

import scipy.sparse as sp

from .base import VectorRepresentation


class IndicatorRepresentation(VectorRepresentation):
    kind = "indicator"

    def fit_texts(self, texts: list[str]) -> None:
        pass

    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        raise NotImplementedError("Emil: binary (len(texts), |vocab|) CSR from tokenizer.vocab")

    @property
    def side_dim(self) -> int:
        raise NotImplementedError("Emil: len(self.tokenizer.vocab)")
