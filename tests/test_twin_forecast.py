import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services import twin_forecast
from routers.digital_twin import router


def frames(rain=5):
    return [{"time": f"2026-09-13T{h:02}:00:00Z", "precipitation": rain,
             "temperature_2m": 25, "relative_humidity_2m": 80,
             "wind_speed_10m": 12, "soil_moisture_0_to_1cm": 0.35} for h in range(12)]


def predict(weather):
    return twin_forecast.predict_surface(list(range(9)), 10, 10.01, 77, 77.01, weather, 3)


def test_fallback_is_deterministic_and_explicit(monkeypatch):
    monkeypatch.delenv("TWIN_GNN_CHECKPOINT", raising=False)
    a, b = predict(frames()), predict(frames())
    assert a["mode"] == "experimental_weather_terrain"
    assert a["frames"] == b["frames"]
    scores = np.array([f["scores"] for f in a["frames"]])
    assert scores.shape == (12, 9)
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
    assert scores[-1].mean() > scores[0].mean()
    assert scores[0, 0] > scores[0, -1]
    assert all(not any(f["scores"]) for f in predict(frames(0))["frames"])


def test_temporal_model_uses_history():
    import torch
    from services.risk_model import GNNTransformerFloodModel
    torch.manual_seed(7)
    model = GNNTransformerFloodModel().eval()
    x = torch.randn(4, 9, 8)
    adj = torch.eye(9)
    with torch.no_grad():
        p, s = model(x, adj)
        changed = x.clone()
        changed[0] += 10
        q, _ = model(changed, adj)
        legacy, _ = model(x[-1], adj)
    assert p.shape == legacy.shape == (9, 1) and s.shape == (9, 4)
    assert not torch.allclose(p, q)


def test_configured_checkpoint(monkeypatch, tmp_path):
    import torch
    from services.risk_model import GNNTransformerFloodModel
    path = tmp_path / "test.pt"
    # Synthetic checkpoint only to exercise wiring, never used by the app.
    torch.save({"feature_schema": twin_forecast.FEATURE_SCHEMA,
                "state_dict": GNNTransformerFloodModel().state_dict()}, path)
    monkeypatch.setenv("TWIN_GNN_CHECKPOINT", str(path))
    assert predict(frames())["mode"] == "gnn_transformer"
    twin_forecast.trained_model.cache_clear()


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def payload():
    return dict(south=10, north=10.01, west=77, east=77.01, size=3, elevations=list(range(9)))


def test_endpoint(api, monkeypatch):
    monkeypatch.delenv("TWIN_GNN_CHECKPOINT", raising=False)
    async def weather(*args):
        return frames()
    monkeypatch.setattr(twin_forecast, "fetch_weather", weather)
    response = api.post("/digital-twin/surface-forecast", json=payload())
    assert response.status_code == 200
    assert len(response.json()["frames"]) == 12


@pytest.mark.parametrize("change", [{"north": 9}, {"east": 80}, {"elevations": [1] * 10}, {"size": 22}, {"elevations": [10000] * 9}])
def test_invalid_grid(api, change):
    assert api.post("/digital-twin/surface-forecast", json={**payload(), **change}).status_code == 422


def test_weather_failure_has_no_fake_forecast(api, monkeypatch):
    async def unavailable(*args):
        raise ValueError("missing precipitation")
    monkeypatch.setattr(twin_forecast, "fetch_weather", unavailable)
    response = api.post("/digital-twin/surface-forecast", json=payload())
    assert response.status_code == 503
    assert "frames" not in response.json()


def test_causal_sequence_matches_prefixes_and_cannot_see_future():
    import torch
    from services.risk_model import GNNTransformerFloodModel
    torch.manual_seed(11)
    model = GNNTransformerFloodModel().eval()
    x = torch.randn(4, 9, 8)
    adjacency = torch.eye(9)
    with torch.inference_mode():
        sequence, severity = model(x, adjacency, return_sequence=True)
        for t in range(4):
            prefix, _ = model(x[:t + 1], adjacency)
            assert torch.allclose(sequence[t], prefix, atol=1e-6)
        changed = x.clone()
        changed[2:] += 100
        other, _ = model(changed, adjacency, return_sequence=True)
    assert severity.shape == (4, 9, 4)
    assert torch.allclose(sequence[:2], other[:2], atol=1e-6)
