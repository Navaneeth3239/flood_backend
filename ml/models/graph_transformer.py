"""Multi-head graph Transformer with residual attention and feed-forward blocks."""
from torch import nn
from torch_geometric.nn import TransformerConv


class GraphTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attention = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                         dropout=dropout, edge_dim=4, root_weight=False)
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        x = self.norm1(x + self.dropout(self.attention(x, edge_index, edge_attr)))
        return self.norm2(x + self.dropout(self.feedforward(x)))


class GraphTransformer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([GraphTransformerBlock(hidden_dim, heads, dropout) for _ in range(layers)])

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x
