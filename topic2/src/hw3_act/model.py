from __future__ import annotations

import torch
from torch import nn


class TorchACTPolicy(nn.Module):
    """A lightweight ACT-style chunking policy.

    It predicts a fixed chunk of future actions from the current state. This keeps
    the assignment pipeline runnable without requiring a heavyweight simulator,
    while preserving ACT's key action-chunking evaluation surface.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        state_token = self.state_proj(state).unsqueeze(1)
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat([state_token, query], dim=1)
        hidden = self.encoder(tokens)[:, 1:, :]
        return self.head(self.norm(hidden))


def build_model(cfg: dict) -> TorchACTPolicy:
    backend = cfg.get("backend", "torch_act")
    if backend != "torch_act":
        raise ValueError(f"Unsupported backend in this code path: {backend}")
    return TorchACTPolicy(
        state_dim=int(cfg["state_dim"]),
        action_dim=int(cfg["action_dim"]),
        chunk_size=int(cfg["chunk_size"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )
