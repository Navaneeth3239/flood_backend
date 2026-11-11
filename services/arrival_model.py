"""Optional trained GNN + sparse graph Transformer for building-arrival forecasts.

No random model is used for inference. A validated, local Hugging Face snapshot is
required; the client falls back to its hydraulic rollout when unavailable.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

FEATURES = ["relative_elevation_m", "depth_m", "source", "path", "roughness",
            "net_rainfall_mm_h", "storm_remaining_seconds", "river_rise_m", "dx_m", "dy_m", "horizon_seconds"]
SCHEMA = "terrain-arrival-v1"


def validate_config(config):
    if config.get("feature_schema") != SCHEMA or config.get("features") != FEATURES:
        raise ValueError("Checkpoint is not trained for terrain-arrival-v1 inputs")
    if config.get("architecture") != "SparseArrivalGNNTransformer" or config.get("target") != "remaining_seconds_and_reach_logit":
        raise ValueError("Checkpoint lacks the flood-arrival regression and reachability heads")
    if config.get("threshold_m") != 0.1 or not config.get("training_dataset") or not config.get("validation"):
        raise ValueError("Checkpoint must document threshold, training dataset and validation")
    mean, std = np.asarray(config.get("mean", [])), np.asarray(config.get("std", []))
    if mean.shape != (len(FEATURES),) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid checkpoint feature normalization")
    if not 8 <= config.get("hidden_dim", 0) <= 128 or config["hidden_dim"] % 4:
        raise ValueError("Invalid hidden dimension")
    return config


def make_model(hidden_dim):
    import torch
    from torch import nn

    class SparseArrivalGNNTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(len(FEATURES), hidden_dim)
            self.message = nn.Linear(hidden_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True, dropout=0)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, 2)

        def forward(self, x, neighbors, padding):
            # Each terrain node attends to itself and its four valid neighbors.
            # O(nodes * degree), with no dense N×N attention allocation.
            h = torch.relu(self.project(x))
            gathered = h[neighbors]
            valid = (~padding).unsqueeze(-1)
            messages = (gathered * valid).sum(1) / valid.sum(1).clamp_min(1)
            h = torch.relu(h + self.message(messages))
            gathered = h[neighbors]
            attended, _ = self.attention(h.unsqueeze(1), gathered, gathered, key_padding_mask=padding, need_weights=False)
            h = self.norm1(h + attended.squeeze(1))
            h = self.norm2(h + self.ff(h))
            output = self.head(h)
            return torch.nn.functional.softplus(output[:, 0]), output[:, 1].sigmoid()

    return SparseArrivalGNNTransformer()


@lru_cache(maxsize=1)
def load_model(directory):
    import torch
    from safetensors.torch import load_file
    root = Path(directory)
    config = validate_config(json.loads((root / "config.json").read_text()))
    model = make_model(config["hidden_dim"])
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite weights")
    model.eval()
    return model, config


def model_status():
    directory = os.getenv("TWIN_ARRIVAL_MODEL_DIR")
    if not directory:
        return {"available": False, "reason": "No compatible trained Hugging Face arrival checkpoint configured"}
    try:
        _, config = load_model(directory)
        return {"available": True, "model": "gnn_transformer", "source": config.get("source_repo"),
                "revision": config.get("source_revision"), "training_dataset": config["training_dataset"]}
    except Exception:
        return {"available": False, "reason": "Arrival checkpoint failed validation"}


def predict_arrivals(payload):
    import torch
    model, config = load_model(os.environ["TWIN_ARRIVAL_MODEL_DIR"])
    cols, rows = payload["cols"], payload["rows"]
    count = cols * rows
    inside = np.asarray(payload["inside"], dtype=bool)
    indices = np.flatnonzero(inside)
    bed = np.asarray(payload["bed"], dtype=np.float32)
    depth = np.asarray(payload["depth"], dtype=np.float32)
    result = np.full(count, -1.0)
    if not len(indices):
        return result.tolist()
    reverse = np.full(count, -1, dtype=np.int64)
    reverse[indices] = np.arange(len(indices))
    neighbors = np.zeros((len(indices), 5), dtype=np.int64)
    padding = np.ones_like(neighbors, dtype=bool)
    for local, cell in enumerate(indices):
        row, col = divmod(int(cell), cols)
        adjacent = [cell]
        for r, c in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if 0 <= r < rows and 0 <= c < cols and inside[r * cols + c]:
                adjacent.append(r * cols + c)
        neighbors[local, :len(adjacent)] = reverse[adjacent]
        padding[local, :len(adjacent)] = False
    x = np.column_stack([bed[indices] - bed[indices].min(), depth[indices],
        np.asarray(payload["sources"])[indices], np.asarray(payload["paths"])[indices],
        *[np.full(len(indices), payload[key]) for key in ["roughness", "netRainfall", "stormRemaining", "sourceRise", "dx", "dy", "horizonSeconds"]]])
    x = ((x - np.asarray(config["mean"])) / np.asarray(config["std"])).astype(np.float32)
    with torch.inference_mode():
        remaining, reach = model(torch.from_numpy(x), torch.from_numpy(neighbors), torch.from_numpy(padding))
    times, probability = remaining.numpy(), reach.numpy()
    if not np.isfinite(times).all() or not np.isfinite(probability).all():
        raise ValueError("Non-finite model output")
    reached = (probability >= 0.5) & (times <= payload["horizonSeconds"])
    result[indices[reached]] = payload["elapsed"] + times[reached]
    result[inside & (depth >= 0.1)] = payload["elapsed"]
    return result.tolist()
