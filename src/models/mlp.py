"""PyTorch MLP for tabular input.

Architecture (output is a LOGIT, no sigmoid inside the network):
  Linear(in, h1) -> ReLU -> Dropout(p)
  -> Linear(h1, h2) -> ReLU -> Dropout(p)
  -> Linear(h2, 1)

Loss: BCEWithLogitsLoss (sigmoid + binary cross-entropy in one
numerically stable step). pos_weight = n_negative / n_positive of the
TRAIN labels (optional). Train positives are 47.3%, so the weight is
about 1.1: classes are nearly balanced and the effect is small. The
option exists so the choice can be documented and tested.

Optimizers: Adam (with weight decay) or SGD with momentum 0.9.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

MOMENTUM = 0.9


class MLP(nn.Module):
    def __init__(self, in_features: int, h1: int = 32, h2: int = 16, dropout: float = 0.2):
        super().__init__()
        self.in_features, self.h1, self.h2, self.dropout = in_features, h1, h2, dropout
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits, shape (batch,)


def make_loss(y_train: np.ndarray | None = None, use_pos_weight: bool = False) -> nn.Module:
    if use_pos_weight and y_train is not None:
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32))
    return nn.BCEWithLogitsLoss()


def make_optimizer(params, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}; use 'adam' or 'sgd'")


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Probability of label 1 for a numpy matrix (eval mode, no gradients)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy().astype("float64")