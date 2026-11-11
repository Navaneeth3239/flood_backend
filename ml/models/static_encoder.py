"""Encode normalized terrain, soil, hydrology and explicit availability."""
from torch import nn


class StaticFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )

    def forward(self, static):
        return self.layers(static)
