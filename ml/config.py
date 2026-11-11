"""Validated configuration shared by training and checkpoint reconstruction."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

DEFAULT_CONFIG = {
    "model": {
        "temporal_backbone": "chronos",
        "pretrained_model_name": "amazon/chronos-t5-tiny",
        "freeze_pretrained": True,
        "hidden_dim": 128,
        "heads": 4,
        "temporal_layers": 2,
        "gat_layers": 2,
        "graph_transformer_layers": 1,
        "dropout": 0.2,
    },
    "data": {
        "frequency": "1h",
        "sequence_length": 24,
        "forecast_horizon_minutes": 360,
        "train_fraction": 0.7,
        "val_fraction": 0.15,
    },
    "graph": {"k_neighbors": 4, "max_distance_km": 5.0},
    "risk": {"moderate": 0.30, "high": 0.55, "critical": 0.75},
    "inference": {"max_prediction_age_minutes": 60},
    "training": {
        "epochs": 100,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "patience": 12,
        "gradient_clip": 1.0,
        "lead_time_weight": 0.1,
        "seed": 42,
        "device": "auto",
        "mixed_precision": True,
        "min_calibration_samples": 30,
        "checkpoint_path": "backend/ml/checkpoints/multi_hazard.pt",
    },
}


def resolve_config(overrides: dict | None = None) -> dict:
    result = deepcopy(DEFAULT_CONFIG)
    for section, values in (overrides or {}).items():
        if section not in result or not isinstance(values, dict):
            raise ValueError(f"Unknown configuration section: {section}")
        result[section].update(deepcopy(values))
    model, data, training = result["model"], result["data"], result["training"]
    if model["temporal_backbone"] not in {"chronos", "local"}:
        raise ValueError("temporal_backbone must be 'chronos' or explicitly 'local'")
    for key in ("hidden_dim", "heads", "temporal_layers", "gat_layers", "graph_transformer_layers"):
        if not isinstance(model[key], int) or model[key] <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    if model["hidden_dim"] % model["heads"]:
        raise ValueError("hidden_dim must be divisible by heads")
    if not 0 <= model["dropout"] < 1:
        raise ValueError("dropout must lie in [0, 1)")
    if data["sequence_length"] < 2 or data["forecast_horizon_minutes"] <= 0:
        raise ValueError("sequence_length >= 2 and a positive forecast horizon are required")
    if not (0 < data["train_fraction"] < 1 and 0 < data["val_fraction"] < 1
            and data["train_fraction"] + data["val_fraction"] < 1):
        raise ValueError("Chronological split fractions must leave a nonempty test fraction")
    if training["epochs"] < 1 or training["patience"] < 1 or training["gradient_clip"] <= 0:
        raise ValueError("Invalid training epochs, patience, or gradient clipping")
    return result


def load_config(path: str | Path) -> dict:
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        return resolve_config(yaml.safe_load(stream) or {})
