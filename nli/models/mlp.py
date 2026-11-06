"""MLP on sparse input (torch): one hidden layer, or a stack of residual blocks; early stopping. OWNER: Emil."""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ..data import LABELS
from ..representations import Features
from .base import Model


class ResBlock(nn.Module):
    """LayerNorm -> Linear -> ReLU -> Dropout, added back onto its input when ``residual``."""

    def __init__(self, dim: int, dropout: float, residual: bool):
        super().__init__()
        self.residual = residual
        self.f = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, h):
        return h + self.f(h) if self.residual else self.f(h)


def build_net(input_dim: int, hidden_dim: int, dropout: float, n_blocks: int, residual: bool) -> nn.Module:
    """``n_blocks=0`` is a single-hidden-layer MLP; otherwise ``n_blocks`` ResBlocks sit between input and output."""
    if n_blocks == 0:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, len(LABELS))
        )
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        *[ResBlock(hidden_dim, dropout, residual) for _ in range(n_blocks)],
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, len(LABELS)),
    )


EVAL_BATCH = 1024  # rows per forward pass when no gradients are needed; only bounds memory


def _dense(X, idx: np.ndarray, device: str) -> torch.Tensor:
    rows = X[idx]
    return torch.from_numpy(rows.toarray() if sp.issparse(rows) else np.asarray(rows)).float().to(device)


class MLPModel(Model):
    kind = "mlp"

    def __init__(
        self,
        seed: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        n_blocks: int = 0,
        residual: bool = True,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 20,
        validation_fraction: float = 0.1,
        patience: int = 3,
        device: str = "cpu",
    ):
        """``validation_fraction`` of the training rows is held out; training stops after ``patience`` epochs
        without a better validation accuracy and the best epoch's weights are kept. ``device`` is any torch
        device string (``"cpu"``, ``"cuda"``, ``"mps"``)."""
        super().__init__(seed)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.residual = residual
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.device = device
        self._net: nn.Module | None = None
        self._input_dim: int | None = None
        # one {"epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"} per epoch
        self.history: list[dict] = []

    def _hyperparameters(self) -> dict:
        return {
            "seed": self.seed,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "n_blocks": self.n_blocks,
            "residual": self.residual,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "validation_fraction": self.validation_fraction,
            "patience": self.patience,
            "device": self.device,
        }

    def _index_batches(self, n: int, rng: np.random.Generator | None = None, size: int | None = None):
        """Row-index arrays covering ``range(n)``, shuffled if ``rng`` is given."""
        order = rng.permutation(n) if rng is not None else np.arange(n)
        size = size or self.batch_size
        for i in range(0, n, size):
            yield order[i : i + size]

    def _batches(self, X, y: np.ndarray, rng: np.random.Generator | None = None, size: int | None = None):
        for idx in self._index_batches(len(y), rng, size):
            yield _dense(X, idx, self.device), torch.from_numpy(y[idx]).long().to(self.device)

    def _evaluate(self, X, y: np.ndarray, loss_fn) -> tuple[float, float]:
        self._net.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in self._batches(X, y, size=EVAL_BATCH):
                logits = self._net(xb)
                total_loss += loss_fn(logits, yb).item() * len(yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
        return total_loss / len(y), correct / len(y)

    def fit(self, features: Features, labels: np.ndarray) -> "MLPModel":
        """Seed torch, split off the validation slice, train with Adam; append one entry per epoch to ``history``."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X, y = features.payload, np.asarray(labels)
        self._input_dim = X.shape[1]
        self._net = build_net(self._input_dim, self.hidden_dim, self.dropout, self.n_blocks, self.residual)
        self._net.to(self.device)
        self.history = []

        order = rng.permutation(len(y))
        n_val = int(round(self.validation_fraction * len(y)))
        val_idx, train_idx = order[:n_val], order[n_val:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_acc, best_state, since_best = -1.0, None, 0
        for epoch in range(1, self.num_epochs + 1):
            self._net.train()
            total_loss, correct = 0.0, 0
            for xb, yb in self._batches(X_train, y_train, rng):
                logits = self._net(xb)
                loss = loss_fn(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
            entry = {"epoch": epoch, "train_loss": total_loss / len(y_train), "train_accuracy": correct / len(y_train)}
            if n_val:
                entry["val_loss"], entry["val_accuracy"] = self._evaluate(X_val, y_val, loss_fn)
            self.history.append(entry)
            if not n_val:
                continue
            if entry["val_accuracy"] > best_acc:
                best_acc, since_best = entry["val_accuracy"], 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
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
        X = features.payload
        out = []
        with torch.no_grad():
            for idx in self._index_batches(X.shape[0], size=EVAL_BATCH):
                out.append(torch.softmax(self._net(_dense(X, idx, self.device)), dim=1).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    def save(self, path: Path) -> None:
        torch.save(self._net.state_dict(), str(path) + ".pt")
        meta = {**self._hyperparameters(), "input_dim": self._input_dim, "history": self.history}
        Path(str(path) + ".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> "MLPModel":
        meta = json.loads(Path(str(path) + ".json").read_text())
        input_dim, history = meta.pop("input_dim"), meta.pop("history")
        model = cls(**meta)
        model._input_dim = input_dim
        model.history = history
        model._net = build_net(input_dim, model.hidden_dim, model.dropout, model.n_blocks, model.residual)
        model._net.load_state_dict(torch.load(str(path) + ".pt", map_location=model.device))
        model._net.to(model.device).eval()
        return model
