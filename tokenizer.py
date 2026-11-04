import re
from collections import Counter

import numpy as np
from scipy import sparse

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BowTokenizer:
    UNK = "<unk>"
 
    def __init__(self, min_freq=2, max_size=20000, binary=False):
        self.min_freq = min_freq    # drop words seen fewer times than this
        self.max_size = max_size    # keep at most this many words (most frequent first)
        self.binary = binary        # presence (1/0) instead of counts
        self.vocab = None
 
    def fit(self, sentences):
        counts = Counter(tok for s in sentences for tok in tokenize(s))
        words = [w for w, c in counts.most_common(self.max_size) if c >= self.min_freq]
        self.vocab = {self.UNK: 0}
        for w in words:
            self.vocab[w] = len(self.vocab)
        return self
 
    def __len__(self):
        return len(self.vocab)
 
    def encode(self, sentence):
        return [self.vocab.get(tok, 0) for tok in tokenize(sentence)]
 
    def transform(self, sentences):
        if self.vocab is None:
            raise RuntimeError("Call fit() before transform().")
        rows, cols, vals = [], [], []
        for i, s in enumerate(sentences):
            for j, c in Counter(self.encode(s)).items():
                rows.append(i)
                cols.append(j)
                vals.append(1.0 if self.binary else float(c))
        return sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(sentences), len(self.vocab)),
            dtype=np.float32,
        )


def pair_features(u, v):
    return sparse.hstack([u, v, abs(u - v), u.multiply(v)]).tocsr()
