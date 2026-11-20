"""Bowman et al. (2015) sentence-encoder classifier (papers/00_bowman2015_snli.pdf, Fig. 3).

(premise, hypothesis) -> Treebank tokens -> GloVe 300d uncased word embeddings -> tanh projection to 100d ->
sentence encoder with shared weights for premise and hypothesis -> [premise; hypothesis] (200d) ->
3 x tanh layers of 200 -> logits for 3 classes -> cross entropy. AdaDelta, dropout and L2, as in the paper.

``encoder`` picks the sentence encoder:
- ``lstm``: 100d LSTM, last hidden state (the paper's 100d LSTM model)
- ``gru``: 100d GRU, last hidden state (same setup with a GRU cell in place of the LSTM)
- ``sum``: sum of projected word vectors (the paper's 100d sum-of-words embed-and-pool baseline)
- ``mean`` / ``max``: other pooling choices over the same projected word vectors

Consumes ``SequencePairs`` Features from the ``glove`` Representation.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from ..representations import Features
from ..representations.glove import PAD_IDX, SequencePairs
from .base import Model

ENCODERS = ("lstm", "gru", "sum", "mean", "max")
EVAL_BATCH = 1024  # pairs per forward pass when no gradients are needed; only bounds memory


class BowmanNet(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,  # (V, E) initial word vectors
        encoder: str = "lstm",
        pad_idx: int = PAD_IDX,
        proj_dim: int = 100,
        hidden_dim: int = 100,
        mlp_dim: int = 200,
        n_layers_mlp: int = 3,
        n_classes: int = 3,
        dropout: float = 0.1,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; choose from {ENCODERS}")
        self.encoder = encoder
        self.pad_idx = pad_idx
        # clone: from_pretrained shares storage, so fine-tuning would otherwise rewrite the caller's matrix
        self.embed = nn.Embedding.from_pretrained(embeddings.clone(), freeze=freeze_embeddings, padding_idx=pad_idx)
        # reduce word embedding dimension from |E|=300d to proj_dim=100d
        self.proj = nn.Sequential(nn.Linear(embeddings.shape[1], proj_dim), nn.Tanh())
        # shared between premise and hypothesis
        self.lstm = nn.LSTM(proj_dim, hidden_dim, batch_first=True) if encoder == "lstm" else None
        self.gru = nn.GRU(proj_dim, hidden_dim, batch_first=True) if encoder == "gru" else None
        sent_dim = hidden_dim if encoder in ["lstm", "gru"] else proj_dim
        self.drop = nn.Dropout(dropout)
        layers: list[nn.Module] = []
        in_dim = 2 * sent_dim
        for _ in range(n_layers_mlp):
            layers += [nn.Linear(in_dim, mlp_dim), nn.Tanh(), nn.Dropout(dropout)]
            in_dim = mlp_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.mlp = nn.Sequential(*layers)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, L) token ids + (B,) sentence lengths -> (B, sentence_dim) sentence vectors."""
        x = self.drop(self.proj(self.embed(ids)))  # (B, L, proj_dim)
        if self.encoder in ["lstm", "gru"]:
            # pack so the LSTM/GRU stops at each sentence's last real token; h_n[-1] is that state
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            if self.lstm is not None:
                _, (h_n, _) = self.lstm(packed)
            elif self.gru is not None:
                _, h_n = self.gru(packed)
            else:
                raise NotImplementedError(f"recurrent encoder {self.encoder!r} has no module set")
            return self.drop(h_n[-1])
        # tanh(W*0 + b) != 0, so padding must be masked out explicitly before pooling
        mask = (ids != self.pad_idx).unsqueeze(-1)
        if self.encoder == "max":
            return x.masked_fill(~mask, float("-inf")).max(dim=1).values
        summed = (x * mask).sum(dim=1)
        return summed if self.encoder == "sum" else summed / lengths.to(x.device).unsqueeze(-1).clamp(min=1)

    def forward(self, p_ids, p_len, h_ids, h_len) -> torch.Tensor:
        u = self.encode(p_ids, p_len)
        v = self.encode(h_ids, h_len)
        return self.mlp(torch.cat([u, v], dim=-1))  # (B, 3) logits


def collate(premise: list[list[int]], hypothesis: list[list[int]], pad_idx: int = PAD_IDX):
    """-> (p_ids, p_len, h_ids, h_len); each side padded separately, true lengths kept for packing/pooling."""
    p_len = torch.tensor([len(s) for s in premise])
    h_len = torch.tensor([len(s) for s in hypothesis])
    p = pad_sequence([torch.tensor(s) for s in premise], batch_first=True, padding_value=pad_idx)
    h = pad_sequence([torch.tensor(s) for s in hypothesis], batch_first=True, padding_value=pad_idx)
    return p, p_len, h, h_len


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BowmanModel(Model):
    kind = "bowman"

    def __init__(
        self,
        seed: int = 0,
        encoder: str = "lstm",
        freeze_embeddings: bool = False,
        proj_dim: int = 100,
        hidden_dim: int = 100,
        mlp_dim: int = 200,
        n_layers_mlp: int = 3,
        dropout: float = 0.1,
        batch_size: int = 128,
        learning_rate: float = 1.0,
        weight_decay: float = 1e-6,
        num_epochs: int = 30,
        validation_fraction: float = 0.1,
        patience: int = 3,
        device: str = "auto",
    ):
        """AdaDelta (``learning_rate`` 1.0 is Zeiler's original) with L2 ``weight_decay``. ``validation_fraction``
        of the fold's training rows is held out; training stops after ``patience`` epochs without a better
        validation accuracy and the best epoch's weights are kept. ``device`` ``auto`` = cuda, else mps, else cpu.
        """
        super().__init__(seed)
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; choose from {ENCODERS}")
        self.encoder = encoder
        self.freeze_embeddings = freeze_embeddings
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.n_layers_mlp = n_layers_mlp
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.device = device
        self._net: BowmanNet | None = None
        self._emb_shape: tuple[int, int] | None = None
        # one {"epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "secs"} per epoch
        self.history: list[dict] = []

    def _hyperparameters(self) -> dict:
        keys = (
            "seed encoder freeze_embeddings proj_dim hidden_dim mlp_dim n_layers_mlp dropout batch_size "
            "learning_rate weight_decay num_epochs validation_fraction patience device"
        ).split()
        return {k: getattr(self, k) for k in keys}

    def _build(self, embeddings: torch.Tensor) -> BowmanNet:
        self._emb_shape = (embeddings.shape[0], embeddings.shape[1])
        net = BowmanNet(
            embeddings,
            encoder=self.encoder,
            proj_dim=self.proj_dim,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            n_layers_mlp=self.n_layers_mlp,
            dropout=self.dropout,
            freeze_embeddings=self.freeze_embeddings,
        )
        return net.to(pick_device(self.device))

    def _batches(self, X: SequencePairs, order: np.ndarray, size: int):
        dev = pick_device(self.device)
        for i in range(0, len(order), size):
            idx = order[i : i + size]
            p, pl, h, hl = collate([X.premise[j] for j in idx], [X.hypothesis[j] for j in idx])
            yield idx, (p.to(dev), pl, h.to(dev), hl)

    def _evaluate(self, X: SequencePairs, y: np.ndarray, loss_fn) -> tuple[float, float]:
        assert self._net is not None
        self._net.eval()
        total_loss, correct = 0.0, 0
        yt = torch.from_numpy(y).long().to(pick_device(self.device))
        with torch.no_grad():
            for idx, batch in self._batches(X, np.arange(len(y)), EVAL_BATCH):
                logits = self._net(*batch)
                total_loss += loss_fn(logits, yt[idx]).item() * len(idx)
                correct += (logits.argmax(dim=1) == yt[idx]).sum().item()
        return total_loss / len(y), correct / len(y)

    def fit(self, features: Features, labels: np.ndarray) -> "BowmanModel":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X: SequencePairs = features.payload
        y = np.asarray(labels)
        self._net = self._build(X.embeddings)
        self.history = []

        order = rng.permutation(len(y))
        n_val = int(round(self.validation_fraction * len(y)))
        val_idx, train_idx = order[:n_val], order[n_val:]
        X_train, y_train = X.take(train_idx), y[train_idx]
        X_val, y_val = X.take(val_idx), y[val_idx]
        yt = torch.from_numpy(y_train).long().to(pick_device(self.device))

        loss_fn = nn.CrossEntropyLoss()
        params = [p for p in self._net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adadelta(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        best_acc, best_state, since_best = -1.0, None, 0
        for epoch in range(1, self.num_epochs + 1):
            t0 = time.monotonic()
            self._net.train()
            total_loss, correct = 0.0, 0
            for idx, batch in self._batches(X_train, rng.permutation(len(y_train)), self.batch_size):
                logits = self._net(*batch)
                loss = loss_fn(logits, yt[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
                correct += (logits.argmax(dim=1) == yt[idx]).sum().item()
            entry = {"epoch": epoch, "train_loss": total_loss / len(y_train), "train_accuracy": correct / len(y_train)}
            if n_val:
                entry["val_loss"], entry["val_accuracy"] = self._evaluate(X_val, y_val, loss_fn)
            entry["secs"] = time.monotonic() - t0
            self.history.append(entry)
            print(f"  [bowman {self.encoder} seed {self.seed}] {json.dumps(entry)}", flush=True)
            if not n_val:
                continue
            if entry["val_accuracy"] > best_acc:
                best_acc, since_best = entry["val_accuracy"], 0
                best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
            else:
                since_best += 1
                if since_best >= self.patience:
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict_proba(self, features: Features) -> np.ndarray:
        """Softmax over the net's output as a numpy ``(n_rows, 3)`` array."""
        assert self._net is not None, "run fit first"
        self._net.eval()
        X: SequencePairs = features.payload
        out = []
        with torch.no_grad():
            for _, batch in self._batches(X, np.arange(len(X)), EVAL_BATCH):
                out.append(torch.softmax(self._net(*batch), dim=1).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    def save(self, path: Path) -> None:
        assert self._net is not None, "run fit first"
        torch.save(self._net.state_dict(), str(path) + ".pt")
        meta = {**self._hyperparameters(), "emb_shape": self._emb_shape, "history": self.history}
        Path(str(path) + ".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> "BowmanModel":
        meta = json.loads(Path(str(path) + ".json").read_text())
        emb_shape, history = meta.pop("emb_shape"), meta.pop("history")
        model = cls(**meta)
        model.history = history
        model._net = model._build(torch.zeros(*emb_shape))
        model._net.load_state_dict(torch.load(str(path) + ".pt", map_location=pick_device(model.device)))
        model._net.eval()
        return model
