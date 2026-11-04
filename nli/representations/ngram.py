"""N-gram counts over Tokenizer tokens. OWNER: Neel.

``fit_texts`` must learn the n-gram vocabulary from the training fold only.
``sklearn.feature_extraction.text.CountVectorizer(analyzer=callable)`` with a callable
that returns token n-grams from ``self.tokenizer.tokenize`` is the shortest route and
gives ``min_df``, ``binary`` and ``max_features`` for free. Consider a ``tfidf`` flag.
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
        tokens = self.tokenizer.tokenize(doc)
        n_grams = []
        for k in range(1, self.n + 1):
            n_grams.extend(map(" ".join, zip(*[tokens[i:] for i in range(k)], strict=False)))
        return n_grams

    def fit_texts(self, texts: list[str]) -> None:
        vectorizer = CountVectorizer(
            min_df=self.min_df,
            binary=self.binary,
            max_features=self.max_features,
            analyzer=self.n_gram_finder,
        )
        vectorizer.fit(texts)
        self._vectorizer = vectorizer

    def transform_texts(self, texts: list[str]) -> sp.csr_matrix:
        assert self._vectorizer is not None, "run fit_texts first"
        return self._vectorizer.transform(texts)

    @property
    def side_dim(self) -> int:
        return len(self._vectorizer.vocabulary_)
