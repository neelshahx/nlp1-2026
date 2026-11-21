"""Bowman et al. (2015) classifier over frozen BERT sentence vectors (variant of ``bowman``; same head and training).

(premise, hypothesis) -> frozen bert-base-uncased, mean-pooled, 768d each (the ``bert`` Representation with
``pairing: sentences``) -> shared tanh projection to 100d -> combiner -> 3 x tanh layers of 200 -> 3 logits.
AdaDelta, dropout, L2, validation split and early stopping are inherited from ``BowmanModel`` unchanged, so the only
difference from the GloVe ``bowman`` models is the sentence encoder (and its tokenizer).

``combiner`` builds the classifier input from the projected sentence vectors u (premise) and v (hypothesis):
- ``concat``: [u, v] (Bowman's)
- ``matching``: [u, v, |u - v|, u * v] (Mou et al. 2016; InferSent)
- ``hypothesis_only``: v (artifact baseline: the premise is never seen)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..representations.bert import SentenceVectorPairs
from .bowman import _WIDTH, COMBINERS, BowmanModel, combine, pick_device


class BowmanBertNet(nn.Module):
    def __init__(
        self,
        input_dim: int,  # BERT hidden size, 768 for bert-base
        combiner: str = "concat",
        proj_dim: int = 100,
        mlp_dim: int = 200,
        n_layers_mlp: int = 3,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if combiner not in COMBINERS:
            raise ValueError(f"unknown combiner {combiner!r}; choose from {COMBINERS}")
        self.combiner = combiner
        # shared between premise and hypothesis; 768d -> 100d so the head matches the GloVe models' 100d sentences
        self.proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Tanh())
        self.drop = nn.Dropout(dropout)
        layers: list[nn.Module] = []
        in_dim = _WIDTH[combiner] * proj_dim
        for _ in range(n_layers_mlp):
            layers += [nn.Linear(in_dim, mlp_dim), nn.Tanh(), nn.Dropout(dropout)]
            in_dim = mlp_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u, v = self.drop(self.proj(u)), self.drop(self.proj(v))
        return self.mlp(combine(u, v, self.combiner))  # (B, 3) logits


class BowmanBertModel(BowmanModel):
    kind = "bowman_bert"

    def __init__(
        self,
        seed: int = 0,
        combiner: str = "concat",
        proj_dim: int = 100,
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
        """Training hyperparameters and their defaults are ``BowmanModel``'s; see there."""
        super().__init__(
            seed=seed,
            encoder="sum",  # unused: the sentence encoder is BERT, upstream in the Representation
            proj_dim=proj_dim,
            mlp_dim=mlp_dim,
            n_layers_mlp=n_layers_mlp,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            validation_fraction=validation_fraction,
            patience=patience,
            device=device,
        )
        if combiner not in COMBINERS:
            raise ValueError(f"unknown combiner {combiner!r}; choose from {COMBINERS}")
        self.encoder = "bert"  # only labels the per-epoch log line
        self.combiner = combiner
        self._input_dim: int | None = None

    def _hyperparameters(self) -> dict:
        keys = (
            "seed combiner proj_dim mlp_dim n_layers_mlp dropout batch_size "
            "learning_rate weight_decay num_epochs validation_fraction patience device"
        ).split()
        return {k: getattr(self, k) for k in keys}

    def _build_net(self, input_dim: int) -> BowmanBertNet:
        self._input_dim = input_dim
        net = BowmanBertNet(
            input_dim,
            combiner=self.combiner,
            proj_dim=self.proj_dim,
            mlp_dim=self.mlp_dim,
            n_layers_mlp=self.n_layers_mlp,
            dropout=self.dropout,
        )
        return net.to(pick_device(self.device))

    def _build_for(self, X: SentenceVectorPairs) -> nn.Module:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._build_net(X.premise.shape[1])

    def _batches(self, X: SentenceVectorPairs, order: np.ndarray, size: int):  # pyright: ignore[reportIncompatibleMethodOverride]
        dev = pick_device(self.device)
        for i in range(0, len(order), size):
            idx = order[i : i + size]
            u = torch.from_numpy(np.asarray(X.premise[idx], dtype=np.float32)).to(dev)
            v = torch.from_numpy(np.asarray(X.hypothesis[idx], dtype=np.float32)).to(dev)
            yield idx, (u, v)

    def save(self, path: Path) -> None:
        assert self._net is not None, "run fit first"
        torch.save(self._net.state_dict(), str(path) + ".pt")
        meta = {**self._hyperparameters(), "input_dim": self._input_dim, "history": self.history}
        Path(str(path) + ".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> "BowmanBertModel":
        meta = json.loads(Path(str(path) + ".json").read_text())
        input_dim, history = meta.pop("input_dim"), meta.pop("history")
        model = cls(**meta)
        model.history = history
        net = model._build_net(input_dim)
        net.load_state_dict(torch.load(str(path) + ".pt", map_location=pick_device(model.device)))
        model._net = net.eval()  # pyright: ignore[reportAttributeAccessIssue]
        return model
