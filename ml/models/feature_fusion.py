"""Learned fusion of temporal, static and satellite/weather modalities."""
import torch
from torch import nn


class FeatureFusion(nn.Module):
    def __init__(self, hidden_dim: int, satellite_dim: int, dropout: float):
        super().__init__()
        self.satellite_encoder = nn.Sequential(nn.Linear(satellite_dim, hidden_dim), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, temporal, static, satellite):
        return self.fusion(torch.cat((temporal, static, self.satellite_encoder(satellite)), dim=-1))
