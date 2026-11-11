"""Edge-aware GATv2 message passing using PyTorch Geometric."""
from torch import nn
from torch_geometric.nn import GATv2Conv


class GATv2Encoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                      dropout=dropout, edge_dim=4, add_self_loops=True, fill_value=0.0)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + self.dropout(self.activation(layer(x, edge_index, edge_attr))))
        return x
