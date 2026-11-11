"""Spatio-Temporal Feature Tokenizer Transformer (FT-Transformer) for Flash Flood Forecasting.

Combines:
1. Temporal Multi-Head Attention Encoder over 10-day antecedent rainfall sequence (T1d - T10d).
2. Tabular Feature-Tokenizer (FT) Multi-Head Self-Attention over catchment morphometric,
   climatic, and socio-economic variables with a learnable [CLS] token.
3. Multi-Modal Cross-Stream Fusion with deep classification head.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalRainfallTransformer(nn.Module):
    """Processes sequential antecedent rainfall (T1d to T10d) with self-attention."""

    def __init__(
        self,
        seq_len: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        
        # Learnable temporal positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (Batch, seq_len)
            
        Returns:
            pooled: (Batch, d_model)
            sequence: (Batch, seq_len, d_model)
        """
        # (B, seq_len) -> (B, seq_len, 1) -> (B, seq_len, d_model)
        tokens = self.input_proj(x.unsqueeze(-1)) + self.pos_embedding
        encoded = self.norm(self.encoder(tokens))
        
        # Aggregate temporal representations: Mean + Max + Last step
        mean_pool = encoded.mean(dim=1)
        last_step = encoded[:, -1]
        pooled = (mean_pool + last_step) * 0.5
        return pooled, encoded


class TabularFeatureTokenizer(nn.Module):
    """Feature Tokenizer (FT) for numerical and categorical catchment attributes.
    
    Transforms each scalar numerical feature into a vector token via affine transformation
    and embeds categorical features into dense embedding lookup vectors.
    """

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        d_model: int = 64,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.d_model = d_model
        
        # Numerical feature weight & bias (one vector per feature)
        self.num_weight = nn.Parameter(torch.randn(num_numerical, d_model) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(num_numerical, d_model))
        
        # Categorical feature embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities
        ])

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Tokenize features into sequence of vectors.
        
        Args:
            x_num: (Batch, num_numerical)
            x_cat: (Batch, num_categorical)
            
        Returns:
            tokens: (Batch, num_numerical + num_categorical, d_model)
        """
        # Numerical tokenization: x * W + b
        num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
        
        # Categorical tokenization
        cat_tokens = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        if cat_tokens:
            cat_tokens_cat = torch.cat(cat_tokens, dim=1)
            tokens = torch.cat([num_tokens, cat_tokens_cat], dim=1)
        else:
            tokens = num_tokens
            
        return tokens


class SpatioTemporalFTTransformer(nn.Module):
    """Unified Spatio-Temporal FT-Transformer for Flood Hazard and Severity Prediction."""

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        seq_len: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        temporal_layers: int = 2,
        tabular_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Stream 1: Temporal Transformer
        self.temporal_stream = TemporalRainfallTransformer(
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=temporal_layers,
            dropout=dropout,
        )
        
        # Stream 2: Tabular Feature Tokenizer
        self.tokenizer = TabularFeatureTokenizer(
            num_numerical=num_numerical,
            cat_cardinalities=cat_cardinalities,
            d_model=d_model,
        )
        
        # [CLS] Token for Tabular Aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Tabular Transformer Encoder
        tab_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tabular_transformer = nn.TransformerEncoder(tab_layer, num_layers=tabular_layers)
        self.tab_norm = nn.LayerNorm(d_model)
        
        # Cross-Stream Attention: Query from Temporal, Key/Value from Tabular
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        
        # Classification Head
        fusion_dim = d_model * 3  # [CLS] token + Temporal pool + Cross-attention pool
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_numerical: torch.Tensor,
        x_categorical: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x_temporal: (Batch, 10) antecedent rainfall sequence
            x_numerical: (Batch, num_numerical) catchment physical/climatic metrics
            x_categorical: (Batch, num_categorical) encoded categorical indices
            
        Returns:
            logits: (Batch, num_classes)
        """
        batch_size = x_temporal.size(0)
        
        # 1. Temporal encoding
        t_pool, t_seq = self.temporal_stream(x_temporal)  # (B, d_model), (B, 10, d_model)
        
        # 2. Tabular tokenization and Transformer attention
        tab_tokens = self.tokenizer(x_numerical, x_categorical)
        cls = self.cls_token.expand(batch_size, -1, -1)
        full_tokens = torch.cat([cls, tab_tokens], dim=1)  # (B, 1 + M, d_model)
        
        tab_encoded = self.tab_norm(self.tabular_transformer(full_tokens))
        cls_rep = tab_encoded[:, 0]  # (B, d_model)
        
        # 3. Cross-attention: Temporal surge queries Catchment vulnerability
        cross_out, _ = self.cross_attn(
            query=t_pool.unsqueeze(1),
            key=tab_encoded,
            value=tab_encoded,
        )
        cross_pool = self.cross_norm(cross_out.squeeze(1))  # (B, d_model)
        
        # 4. Multi-modal fusion
        fused = torch.cat([cls_rep, t_pool, cross_pool], dim=-1)  # (B, 3 * d_model)
        
        return self.classifier(fused)
