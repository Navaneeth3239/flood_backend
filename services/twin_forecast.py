"""Terrain graph + hourly weather for the digital twin surface.

No randomly initialized neural output is ever exposed as a forecast.
"""
import os
from datetime import datetime, timezone
from functools import lru_cache

import httpx
import numpy as np

FEATURE_SCHEMA = "terrain-weather-v1"


async def fetch_weather(lat, lng):
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lng, "timezone": "UTC",
            "forecast_hours": 12,
            "hourly": "precipitation,temperature_2m,relative_humidity_2m,wind_speed_10m,soil_moisture_0_to_1cm",
        })
        response.raise_for_status()
        hourly = response.json()["hourly"]
    keys = ["precipitation", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "soil_moisture_0_to_1cm"]
    if len(hourly["time"]) != 12:
        raise ValueError("Incomplete weather forecast")
    frames = []
    for i, time in enumerate(hourly["time"]):
        values = {key: hourly[key][i] for key in keys}
        if any(v is None or not np.isfinite(v) for v in values.values()):
            raise ValueError("Weather provider returned missing data")
        frames.append({"time": time + "Z", **values})
    return frames


@lru_cache(maxsize=1)
def trained_model(path):
    import torch
    from services.risk_model import GNNTransformerFloodModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Checkpoint must use terrain-weather-v1 features")
    model = GNNTransformerFloodModel(node_features=8, hidden_dim=32)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def predict_surface(elevations, south, north, west, east, frames, size):
    elevations = np.asarray(elevations, dtype=np.float32).reshape(size, size)
    dy = max((north - south) * 111132 / (size - 1), 1)
    dx = max((east - west) * 111132 * np.cos(np.radians((north + south) / 2)) / (size - 1), 1)
    gy, gx = np.gradient(elevations, dy, dx)
    slope = np.arctan(np.hypot(gx, gy)) / (np.pi / 2)
    relative = (elevations - elevations.min()) / max(float(np.ptp(elevations)), 1)
    n = size * size
    adjacency = np.eye(n, dtype=np.float32)
    for row in range(size):
        for col in range(size):
            i = row * size + col
            for r, c in [(row + 1, col), (row, col + 1)]:
                if r < size and c < size:
                    j = r * size + c
                    adjacency[i, j] = adjacency[j, i] = 1
    sequences, scores = [], []
    accumulation = 0.0
    for frame in frames:
        rain = max(0, frame["precipitation"])
        accumulation += rain
        moisture = np.clip(frame["soil_moisture_0_to_1cm"] / 0.5, 0, 1)
        features = np.stack([
            relative.ravel(), slope.ravel(),
            *[np.full(n, value) for value in [rain / 100, accumulation / 150,
              frame["relative_humidity_2m"] / 100, frame["temperature_2m"] / 50,
              frame["wind_speed_10m"] / 100, moisture]],
        ], axis=-1).astype(np.float32)
        sequences.append(features)
        # Explicit experimental index; no claim of calibrated flood probability.
        wetness = 1 - np.exp(-(rain / 25 + accumulation / 100) * (0.4 + 0.6 * moisture))
        local = wetness * (0.35 + 0.65 * (1 - relative.ravel())) * (1 - 0.4 * slope.ravel())
        propagated = adjacency @ local / adjacency.sum(axis=1)
        scores.append(np.clip(0.7 * local + 0.3 * propagated, 0, 1))
    path = os.environ.get("TWIN_GNN_CHECKPOINT")
    mode = "experimental_weather_terrain"
    if path:
        import torch
        model = trained_model(path)
        x = torch.from_numpy(np.stack(sequences))  # time, nodes, features
        with torch.inference_mode():
            scores = model(x, torch.from_numpy(adjacency), return_sequence=True)[0].squeeze(-1).numpy()
        mode = "gnn_transformer"
    return {
        "mode": mode, "feature_schema": FEATURE_SCHEMA,
        "source": "Open-Meteo", "weather_scope": "Area centre forecast shared across terrain nodes",
        "fetched_at": datetime.now(timezone.utc).isoformat(), "size": size,
        "frames": [{**frame, "scores": np.round(score, 4).tolist()} for frame, score in zip(frames, scores)],
    }
