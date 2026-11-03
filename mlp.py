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
        return self.net(x)