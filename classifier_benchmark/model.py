"""Compact PyTorch classifier over CSP log-variance features.

CSP already does the spatial feature engineering (see `csp_features.py`),
so the classifier itself only needs to discriminate a small, low-noise
feature vector -- a shallow MLP is the right amount of model, not a
deep CNN over raw channels x time.
"""
from __future__ import annotations

import torch
from torch import nn


class CSPClassifier(nn.Module):
    """Two-hidden-layer MLP over CSP log-variance features.

    Parameters
    ----------
    n_features : int
        Number of input CSP features (2 * n_components).
    hidden_dim : int, default 16
        Width of both hidden layers.
    n_classes : int, default 2
        Number of output classes.
    """

    def __init__(self, n_features: int, hidden_dim: int = 16, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CSPClassifier(n_features=8, hidden_dim=16, n_classes=2)
    print(model)

    dummy = torch.randn(5, 8)
    logits = model(dummy)
    print("output shape (expect (5, 2)):", tuple(logits.shape))
    assert logits.shape == (5, 2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_params}")
    print("OK: forward pass produces class logits of the expected shape.")
