from collections import Counter
import numpy as np
from nli.data import load_snli
from nli.tokenizers import TreebankTokenizer
from nli.embeddings import load_glove

d = load_snli()
tr, dev = d["train"], d["dev"]
tok = TreebankTokenizer()
index, table = load_glove()

sents = set(tr["premise"]) | set(tr["hypothesis"])
print("label col sample:", tr.columns.tolist(), len(tr))
print("unique train sentences:", len(sents))
counts = Counter(t for s in sents for t in tok.tokenize(s))
ntok = sum(counts.values())
types_in = [w for w in counts if w in index]
tok_in = sum(c for w, c in counts.items() if w in index)
print(f"tokens {ntok:,}  types {len(counts):,}")
print(f"type coverage {len(types_in)/len(counts):.4f}  token coverage {tok_in/ntok:.4f}")
missing = sorted(((c, w) for w, c in counts.items() if w not in index), reverse=True)
print("top missing:", missing[:20])
print("missing types seen once:", sum(1 for c, w in missing if c == 1), "of", len(missing))
lens = [len(tok.tokenize(s)) for s in sents]
print("mean len", np.mean(lens), "p95", np.percentile(lens, 95), "max", max(lens))
# dev tokens not in train vocab (become <unk> in our pipeline)
dsents = set(dev["premise"]) | set(dev["hypothesis"])
dtoks = [t for s in dsents for t in tok.tokenize(s)]
print(f"dev tokens not in train vocab: {np.mean([t not in counts for t in dtoks]):.4f}")

def cos(a, b):
    u, v = table[index[a]], table[index[b]]
    return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
pairs = [("man","guy"),("dog","puppy"),("running","jogging"),("sitting","standing"),
         ("asleep","awake"),("man","woman"),("dog","car"),("guitar","beach")]
for a, b in pairs: print(f"{a}/{b}: {cos(a,b):.3f}")
