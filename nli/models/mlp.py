"""Single-hidden-layer MLP on sparse input (torch). OWNER: Emil.

Starting point: ``nlp1-student/part1/smlp.py`` (``SparseMLP``, ``scipy_to_torch_sparse``,
``SparseDataset``, ``sparse_collate``, ``train_one_epoch``, ``evaluate``). KNOWN ISSUES there:
- torch is never seeded: call ``torch.manual_seed(self.seed)`` before building the net and
  seed the DataLoader generator.
- early stopping and checkpoint selection used the fold that was then reported (leak).
  Carve ``validation_fraction`` out of the training rows passed to ``fit`` instead (ADR-0001).
- four sparse conversions per batch; convert the CSR once up front.
Hyperparameters: ``hidden_dim``, ``dropout``, ``batch_size``, ``learning_rate``,
``weight_decay``, ``num_epochs``, ``validation_fraction``, ``patience``. Record the per-epoch
train/val curve on ``self.history`` so ``plotting.plot_learning_curve`` can draw it.
"""

from pathlib import Path

import numpy as np

from ..representations import Features
from .base import Model


class MLPModel(Model):
    kind = "mlp"

    def __init__(self, seed: int = 0, hidden_dim: int = 128, dropout: float = 0.1,
                 batch_size: int = 64, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 num_epochs: int = 20, validation_fraction: float = 0.1, patience: int = 3):
        super().__init__(seed)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self._net = None  # torch.nn.Module
        self.history: list[dict] = []  # one {"epoch", "train_loss", "val_loss", "val_accuracy"} per epoch

    def fit(self, features: Features, labels: np.ndarray) -> "MLPModel":
        raise NotImplementedError("Emil: seed torch, split off validation_fraction, train SparseMLP")

    def predict_proba(self, features: Features) -> np.ndarray:
        raise NotImplementedError("Emil: softmax over the net's output, as a numpy (n, 3) array")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Emil: torch.save(state_dict) + a json with the architecture")

    @classmethod
    def load(cls, path: Path) -> "MLPModel":
        raise NotImplementedError("Emil")
