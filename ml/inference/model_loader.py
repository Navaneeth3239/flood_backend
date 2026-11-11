"""Strict, cached checkpoint loading. A companion JSON flag never grants readiness."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any


class ModelNotReady(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class ModelBundle:
    model: Any
    preprocessor: Any
    checkpoint: dict
    device: str
    identity: str
    ready: bool
    reason: str | None

    @property
    def version(self) -> str:
        return self.checkpoint["model_version"]

    @property
    def config(self) -> dict:
        return self.checkpoint["config"]


_load_lock = RLock()


def _finite_json(value):
    """Metadata must contain JSON-safe scalars, never NaN or serialized objects."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and _finite_json(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(v) for v in value)
    return value is None or isinstance(value, (str, bool, int)) or (
        isinstance(value, float) and math.isfinite(value)
    )


def _trained_reason(training: dict) -> str | None:
    if training.get("completed") is not True or training.get("optimizer_steps", 0) <= 0:
        return "Weights loaded, but completed supervised training is not recorded."
    if training.get("data_provenance") != "real":
        return "Weights loaded for testing only. A checkpoint trained on real labeled data is required."
    counts = training.get("label_counts", {})
    if any(not isinstance(counts.get(key), int) or counts[key] <= 0
           for key in ("flood", "landslide", "combined_risk")):
        return "Weights loaded, but supervised labels are missing for one or more hazard heads."
    return None


def load_checkpoint(path: str | Path, device: str = "cpu") -> ModelBundle:
    """Load trusted local training artifacts with PyTorch's restricted unpickler.

    Serving reconstructs Chronos from the stored HF configuration and strictly
    restores its weights. It never downloads a backbone or accepts random heads.
    Training metadata is provenance, not a substitute for scientific validation.
    """
    import torch
    from ..data.feature_schema import feature_schema
    from ..data.preprocessing import FeaturePreprocessor
    from ..models.disaster_gnn_transformer import DisasterGNNTransformer

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ModelNotReady("checkpoint_missing", "The configured checkpoint file does not exist.")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ModelNotReady("checkpoint_invalid", "Requested inference device is unavailable; choose cpu or an available cuda device.")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format_version")
        required = {"config", "model_state_dict", "preprocessor", "feature_schema", "training", "model_version", "trained_at"}
        if not required.issubset(checkpoint):
            raise ValueError("Checkpoint is missing model, preprocessing, or training metadata")
        if checkpoint["feature_schema"] != feature_schema():
            raise ValueError("Checkpoint feature order does not match the serving feature schema")
        for key in required - {"model_state_dict"}:
            if not _finite_json(checkpoint[key]):
                raise ValueError(f"Invalid checkpoint metadata: {key}")
        if not isinstance(checkpoint["model_version"], str) or not checkpoint["model_version"]:
            raise ValueError("Missing model_version")
        if not isinstance(checkpoint["training"], dict):
            raise ValueError("Invalid training metadata")
        weights = checkpoint["model_state_dict"]
        if not isinstance(weights, dict) or not weights or any(
            not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in weights.values()
        ):
            raise ValueError("Model weights are empty, non-tensor, or non-finite")
        for key in ("calibration", "metrics"):
            if not _finite_json(checkpoint.get(key, {})):
                raise ValueError(f"Invalid {key} metadata")
        calibration = checkpoint.get("calibration", {})
        for temperature in calibration.get("temperatures", {}).values():
            if not isinstance(temperature, (float, int)) or isinstance(temperature, bool) or not 0.01 <= temperature <= 100:
                raise ValueError("Calibration temperature must be finite and positive")
        preprocessor = FeaturePreprocessor.from_dict(checkpoint["preprocessor"])
        model = DisasterGNNTransformer(checkpoint["config"], initialize_pretrained=False)
        model.load_state_dict(weights, strict=True)
        model.to(device).eval()
        # A loadable state dict is insufficient: execute the exact graph model.
        schema = feature_schema()
        sequence = int(checkpoint["config"]["data"]["sequence_length"])
        if not 1 <= sequence <= 2048:
            raise ValueError("Invalid sequence_length")
        with torch.inference_mode():
            outputs = model(
                torch.zeros((2, sequence, len(schema["temporal"])), device=device),
                torch.zeros((2, len(schema["static"])), device=device),
                torch.zeros((2, len(schema["satellite"])), device=device),
                torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device),
                torch.zeros((2, 4), device=device),
            )
        for key in ("flood_logits", "landslide_logits", "combined_risk_logits", "lead_time"):
            if outputs[key].shape != (2,) or not torch.isfinite(outputs[key]).all():
                raise ValueError("Checkpoint forward pass returned invalid predictions")
        reason = _trained_reason(checkpoint["training"])
        stat = checkpoint_path.stat()
        return ModelBundle(model, preprocessor, checkpoint, device,
                           f"{checkpoint_path}:{stat.st_mtime_ns}:{stat.st_size}", not reason, reason)
    except ModelNotReady:
        raise
    except Exception as exc:
        raise ModelNotReady("checkpoint_invalid", f"Checkpoint validation failed: {type(exc).__name__}: {exc}") from exc


@lru_cache(maxsize=1)
def _cached_load(path: str, mtime: int, size: int, device: str) -> ModelBundle:
    return load_checkpoint(path, device)


def get_model(require_ready: bool = True) -> ModelBundle:
    configured = os.environ.get("MULTI_HAZARD_MODEL_CHECKPOINT", "").strip()
    if not configured:
        raise ModelNotReady("checkpoint_missing", "No trained multi-hazard checkpoint is configured. Set MULTI_HAZARD_MODEL_CHECKPOINT after training.")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise ModelNotReady("checkpoint_missing", "The configured multi-hazard checkpoint file was not found.")
    device = os.environ.get("MULTI_HAZARD_DEVICE", "cpu")
    try:
        stat = path.stat()
        with _load_lock:
            bundle = _cached_load(str(path), stat.st_mtime_ns, stat.st_size, device)
    except ModelNotReady:
        raise
    except Exception as exc:
        raise ModelNotReady("checkpoint_invalid", f"Cannot load checkpoint: {type(exc).__name__}: {exc}") from exc
    if require_ready and not bundle.ready:
        raise ModelNotReady("model_loaded", bundle.reason or "Supervised training is required.")
    return bundle


def clear_model_cache():
    with _load_lock:
        _cached_load.cache_clear()
