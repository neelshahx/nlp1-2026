"""N-gram counts over Tokenizer tokens. OWNER: Neel.

Every word sequence of length 1..n seen in training becomes one count feature.
"""

import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .base import VectorRepresentation


class NgramRepresentation(VectorRepresentation):
    kind = "ngram"

    def __init__(
        self,
        tokenizer,
        pairing: str = "concat",
        n: int = 2,
        min_df: int = 1,
        binary: bool = False,
        max_features: int | None = None,
    ):
        super().__init__(tokenizer, pairing)
        assert n > 0, "n > 0"
        self.n = n
        self.min_df = min_df
        self.binary = binary
        self.max_features = max_features
        self._vectorizer = None

    def n_gram_finder(self, doc: str) -> list[str]:
        """Split a text into words, then list every run of 1 to n consecutive words."""
        tokens = self.tokenizer.tokenize(doc)
        n_grams = []
        for k in range(1, self.n + 1):
            n_grams.extend(map(" ".join, zip(*[tokens[i:] for i in range(k)], strict=False)))
        return n_grams

    def fit_texts(self, texts: list[str]) -> None:
        """Learn the set of n-grams from these texts only; rare ones can be dropped via min_df/max_features."""
        vectorizer = CountVectorizer(
            min_df=self.min_df,
            binary=self.binary,
            max_features=self.max_features,
            analyzer=self.n_gram_finder,
        )
        vectorizer.fit(texts)
        self._vectorizer = vectorizer

    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        """Count how often each known n-gram appears in each text (1/0 if binary); unknown ones are ignored."""
        assert self._vectorizer is not None, "run fit_texts first"
        return self._vectorizer.transform(texts)

    @property
    def side_dim(self) -> int:
        """Number of n-grams learned, i.e. the length of one text's count vector."""
        return len(self._vectorizer.vocabulary_)
