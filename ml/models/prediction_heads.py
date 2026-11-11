"""Independent hazard logits and a learned cross-hazard combined-risk head."""
import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, horizon_minutes: float):
        super().__init__()
        self.horizon_minutes = float(horizon_minutes)
        self.flood_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.landslide_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.flood = nn.Linear(hidden_dim, 1)
        self.landslide = nn.Linear(hidden_dim, 1)
        self.combined = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.lead_time = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        flood, landslide = self.flood_embedding(x), self.landslide_embedding(x)
        return {
            "flood_logits": self.flood(flood).squeeze(-1),
            "landslide_logits": self.landslide(landslide).squeeze(-1),
            "combined_risk_logits": self.combined(torch.cat((x, flood, landslide), dim=-1)).squeeze(-1),
            # This sigmoid bounds a regression output, never a BCE input.
            "lead_time": torch.sigmoid(self.lead_time(x).squeeze(-1)) * self.horizon_minutes,
        }
