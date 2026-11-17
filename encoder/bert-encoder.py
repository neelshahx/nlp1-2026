"""Frozen BERT sentence embeddings (mean pooling) for SNLI.

Run once to precompute everything:
    python bert_encoder.py

This writes, for each split (train / validation / test), into cache/:
    bert_<split>_premise.npy     (n, 768) float16
    bert_<split>_hypothesis.npy  (n, 768) float16
    labels_<split>.npy           (n,)     int64   0=entailment 1=neutral 2=contradiction
Row i of all three files is the same sentence pair. Pairs without a gold
label (-1) are already removed.

Loading in any model:
    import numpy as np
    U = np.load("cache/bert_train_premise.npy").astype(np.float32)
    V = np.load("cache/bert_train_hypothesis.npy").astype(np.float32)
    y = np.load("cache/labels_train.npy")
"""
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-uncased"
CACHE_DIR = "cache"


class BertEncoder:
    def __init__(self, model_name=MODEL_NAME, max_length=64, batch_size=256, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length      # SNLI sentences are short; 64 tokens covers almost all
        self.batch_size = batch_size
        self.dim = self.model.config.hidden_size   # 768 for bert-base

    @torch.no_grad()
    def _encode_batch(self, sentences):
        enc = self.tokenizer(sentences, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
        # fp16 on GPU is ~2x faster; on CPU stay in fp32
        with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
            out = self.model(**enc).last_hidden_state            # (batch, tokens, 768)
        # Mean pooling over real tokens only (padding masked out)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.float() * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).cpu().numpy()

    def transform(self, sentences):
        """List of sentences -> (n, 768) float16 array.

        Duplicate sentences are encoded once. This matters for SNLI, where
        each premise appears with about three different hypotheses.
        """
        unique = list(dict.fromkeys(sentences))
        index = {s: i for i, s in enumerate(unique)}
        # Sort by length so each batch has similar lengths (less padding, faster)
        order = sorted(range(len(unique)), key=lambda i: len(unique[i]))
        emb = np.zeros((len(unique), self.dim), dtype=np.float16)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            emb[idx] = self._encode_batch([unique[i] for i in idx])
            done = min(start + self.batch_size, len(order))
            if (start // self.batch_size) % 100 == 0 or done == len(order):
                print(f"  encoded {done:,}/{len(unique):,} unique sentences")
        return emb[[index[s] for s in sentences]]


def cached_embeddings(encoder, sentences, name):
    """Load cache/<name>.npy if it exists, otherwise encode and save it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.npy")
    if os.path.exists(path):
        emb = np.load(path)
        if len(emb) == len(sentences):
            print(f"Loaded {path}")
            return emb
        print(f"{path} has the wrong number of rows; re-encoding")
    print(f"Encoding {name} ({len(sentences):,} sentences) on {encoder.device}...")
    emb = encoder.transform(sentences)
    np.save(path, emb)
    return emb


def dense_pair_features(u, v):
    """[u, v, |u - v|, u * v] for dense arrays (float32 output)."""
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=1)


if __name__ == "__main__":
    # Precompute embeddings and labels for all SNLI splits into cache/.
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/snli")
    ds = ds.filter(lambda ex: ex["label"] != -1)   # drop pairs without a gold label
    encoder = BertEncoder()
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split in ["train", "validation", "test"]:
        cached_embeddings(encoder, list(ds[split]["premise"]), f"bert_{split}_premise")
        cached_embeddings(encoder, list(ds[split]["hypothesis"]), f"bert_{split}_hypothesis")
        np.save(os.path.join(CACHE_DIR, f"labels_{split}.npy"),
                np.array(ds[split]["label"], dtype=np.int64))
        print(f"Saved {split}: {len(ds[split]):,} pairs")
    print(f"Done. Files are in {CACHE_DIR}/")