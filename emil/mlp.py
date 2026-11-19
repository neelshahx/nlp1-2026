import torch
import torch.nn as nn


class MLP(nn.Module):
 
    def __init__(self, input_dim, hidden_dim=512, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # W1, b1
            nn.ReLU(),                          # nonlinearity
            nn.Dropout(dropout),                # regularization
            nn.Linear(hidden_dim, n_classes),   # W2, b2 -> logits
        )
 
    def forward(self, x):
        # Returns raw logits; nn.CrossEntropyLoss applies the softmax.
        return self.net(x)
 
 
class ResBlock(nn.Module):
 
    def __init__(self, dim, dropout=0.3, residual=True):
        super().__init__()
        self.residual = residual
        self.f = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
 
    def forward(self, h):
        return h + self.f(h) if self.residual else self.f(h)
 
 
class ResMLP(nn.Module):
 
    def __init__(self, input_dim, hidden_dim=512, n_blocks=4, n_classes=3,
                 dropout=0.3, residual=True):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout, residual) for _ in range(n_blocks)]
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_classes))
 
    def forward(self, x):
        h = torch.relu(self.inp(x))
        return self.out(self.blocks(h))
 
 
def build_model(name, input_dim, hidden_dim=512, n_blocks=4):
    if name == "mlp":
        return MLP(input_dim, hidden_dim)
    if name == "deep_mlp":
        return ResMLP(input_dim, hidden_dim, n_blocks, residual=False)
    if name == "res_mlp":
        return ResMLP(input_dim, hidden_dim, n_blocks, residual=True)
    raise ValueError(f"Unknown model: {name}")
