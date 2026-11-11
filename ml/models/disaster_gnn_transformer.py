"""Executable spatio-temporal multi-hazard GNN-Transformer architecture."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from ..config import resolve_config
from ..data.feature_schema import TEMPORAL_INPUT_DIM, STATIC_INPUT_DIM, SATELLITE_INPUT_DIM
from .feature_fusion import FeatureFusion
from .gnn import GATv2Encoder
from .graph_transformer import GraphTransformer
from .prediction_heads import MultiTaskHeads
from .static_encoder import StaticFeatureEncoder
from .temporal_transformer import TemporalTransformer


class DisasterGNNTransformer(nn.Module):
    def __init__(self, config: dict | None = None, *, initialize_pretrained: bool = True):
        super().__init__()
        self.config = resolve_config(config)
        model = self.config["model"]
        hidden, heads, dropout = model["hidden_dim"], model["heads"], model["dropout"]
        self.temporal_encoder = TemporalTransformer(TEMPORAL_INPUT_DIM, model,
                                                    initialize_pretrained=initialize_pretrained)
        self.static_encoder = StaticFeatureEncoder(STATIC_INPUT_DIM, hidden, dropout)
        self.fusion = FeatureFusion(hidden, SATELLITE_INPUT_DIM, dropout)
        self.gat = GATv2Encoder(hidden, heads, model["gat_layers"], dropout)
        self.graph_transformer = GraphTransformer(hidden, heads, model["graph_transformer_layers"], dropout)
        self.heads = MultiTaskHeads(hidden, dropout, self.config["data"]["forecast_horizon_minutes"])

    def checkpoint_config(self) -> dict:
        config = deepcopy(self.config)
        if self.temporal_encoder.backbone_config is not None:
            config["model"]["backbone_config"] = deepcopy(self.temporal_encoder.backbone_config)
        return config

    def forward(self, temporal, static, satellite, edge_index, edge_attr):
        if temporal.ndim != 3 or temporal.shape[-1] != TEMPORAL_INPUT_DIM:
            raise ValueError(f"temporal must have shape [nodes, time, {TEMPORAL_INPUT_DIM}]")
        n = temporal.shape[0]
        if static.shape != (n, STATIC_INPUT_DIM) or satellite.shape != (n, SATELLITE_INPUT_DIM):
            raise ValueError("Static/satellite input does not match the central feature schema")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_attr.shape != (edge_index.shape[1], 4):
            raise ValueError("Expected edge_index [2, E] and edge_attr [E, 4]")
        for tensor in (temporal, static, satellite, edge_attr):
            if not torch.isfinite(tensor).all():
                raise ValueError("Model inputs must be preprocessed finite values with explicit missingness masks")
        x = self.fusion(self.temporal_encoder(temporal), self.static_encoder(static), satellite)
        x = self.gat(x, edge_index, edge_attr)
        return self.heads(self.graph_transformer(x, edge_index, edge_attr))
