"""Chronos-T5 embeddings, or an explicitly selected local Transformer.

Chronos reference: https://github.com/amazon-science/chronos-forecasting
The pretrained encoder is saved inside the task checkpoint. Serving therefore
never downloads a backbone or silently substitutes random pretrained weights.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, config: dict, *, initialize_pretrained: bool = True):
        super().__init__()
        self.kind = config["temporal_backbone"]
        self.feature_count = input_dim // 2
        hidden = config["hidden_dim"]
        self.freeze_pretrained = bool(config.get("freeze_pretrained", True))
        self.backbone_config = None
        if self.kind == "local":
            self.input_projection = nn.Linear(input_dim, hidden)
            layer = nn.TransformerEncoderLayer(
                hidden, config["heads"], hidden * 4, config["dropout"],
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, config["temporal_layers"], enable_nested_tensor=False)
            self.output = nn.LayerNorm(hidden)
            return
        if self.kind != "chronos":
            raise ValueError(f"Unsupported temporal backbone: {self.kind}")
        try:
            from chronos import ChronosConfig, ChronosPipeline
            from transformers import T5Config, T5EncoderModel
        except ImportError as exc:
            raise RuntimeError(
                "Chronos requires chronos-forecasting and transformers. Install "
                "backend/requirements-ml.txt or explicitly select temporal_backbone: local "
                "for development; no automatic fallback is used."
            ) from exc
        if initialize_pretrained:
            pipeline = ChronosPipeline.from_pretrained(
                config["pretrained_model_name"], device_map="cpu", torch_dtype=torch.float32,
                trust_remote_code=False,
            )
            if pipeline.model.config.model_type != "seq2seq":
                raise ValueError("The Chronos adapter requires a Chronos-T5 encoder-decoder checkpoint")
            self.encoder = pipeline.model.model.encoder
            self.backbone_config = pipeline.model.model.config.to_dict()
            self.tokenizer = pipeline.tokenizer
        else:
            self.backbone_config = config.get("backbone_config")
            if not isinstance(self.backbone_config, dict) or "chronos_config" not in self.backbone_config:
                raise ValueError("Checkpoint is missing its serialized Chronos backbone configuration")
            t5_config = T5Config.from_dict(self.backbone_config)
            self.encoder = T5EncoderModel(t5_config).encoder
            self.tokenizer = ChronosConfig(**self.backbone_config["chronos_config"]).create_tokenizer()
        self.backbone_hidden = int(self.backbone_config["d_model"])
        self.output = nn.Sequential(
            nn.Linear(self.feature_count * (self.backbone_hidden + 1), hidden),
            nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(config["dropout"]),
        )
        if self.freeze_pretrained:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.kind == "chronos" and self.freeze_pretrained:
            self.encoder.eval()
        return self

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        if self.kind == "local":
            x = self.input_projection(temporal)
            length, hidden = x.shape[1:]
            position = torch.arange(length, dtype=x.dtype, device=x.device).unsqueeze(1)
            divisor = torch.exp(torch.arange(0, hidden, 2, device=x.device, dtype=x.dtype)
                                * (-math.log(10000.0) / hidden))
            encoding = torch.zeros(length, hidden, dtype=x.dtype, device=x.device)
            encoding[:, 0::2] = torch.sin(position * divisor)
            encoding[:, 1::2] = torch.cos(position * divisor[: hidden // 2])
            return self.output(self.encoder(x + encoding.unsqueeze(0)).mean(dim=1))
        nodes, length, _ = temporal.shape
        values = temporal[..., :self.feature_count]
        availability = temporal[..., self.feature_count:]
        # Chronos is univariate: encode each measured feature independently,
        # preserving feature identity when the embeddings are fused below.
        context = values.masked_fill(availability < 0.5, float("nan"))
        context = context.transpose(1, 2).reshape(nodes * self.feature_count, length).detach().cpu()
        token_ids, mask, _ = self.tokenizer.context_input_transform(context)
        device = temporal.device
        token_ids, mask = token_ids.to(device), mask.to(device)
        encoded = self.encoder(input_ids=token_ids, attention_mask=mask).last_hidden_state
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        pooled = pooled.reshape(nodes, self.feature_count, self.backbone_hidden)
        available_fraction = availability.mean(dim=1).unsqueeze(-1)
        pooled = torch.cat((pooled, available_fraction), dim=-1).flatten(start_dim=1)
        return self.output(pooled)
