"""Tests for multi-hazard model inference, training, checkpoint loading, and GeoJSON endpoints."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.ml.config import resolve_config
from backend.ml.data.feature_schema import schema_dict
from backend.ml.inference.model_loader import clear_model_cache, get_model, load_checkpoint, ModelNotReady
from backend.ml.inference.predictor import clear_predictions
from backend.ml.training.train import train_frame
from backend.routers.digital_twin import router


def synthetic_training_data(periods=50, nodes=("NODE_01", "NODE_02")):
    rows = []
    for hour, timestamp in enumerate(pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")):
        for index, node in enumerate(nodes):
            rows.append({
                "node_id": node,
                "timestamp": timestamp,
                "latitude": 11.62 + index * 0.01,
                "longitude": 76.08 + index * 0.01,
                "soil_moisture": 0.2 + (hour % 10) * 0.05,
                "water_level": 1.0 + (hour % 5) * 0.3,
                "rainfall": float(hour % 4),
                "temperature": 22.0 + (hour % 8),
                "humidity": 65.0 + (hour % 15),
                "accelerometer_x": 0.01 * (index + 1),
                "accelerometer_y": 0.02,
                "accelerometer_z": 9.81,
                "tilt": 1.5,
                "vibration": 0.05,
                "elevation": 500.0 - index * 50,
                "slope": 12.0 + index * 2,
                "aspect": 180.0,
                "flow_accumulation": 1200.0,
                "distance_to_river": 150.0,
                "soil_type": "clay_loam" if index == 0 else "sandy_loam",
                "clay_fraction": 0.35,
                "sand_fraction": 0.40,
                "lulc": "forest",
                "historical_flood_frequency": 0.2,
                "historical_landslide_frequency": 0.15,
                "sentinel1_vv": 0.45,
                "sentinel1_vh": 0.15,
                "ndvi": 0.65,
                "ndwi": 0.20,
                "flood_label": float((hour + index) % 2),
                "landslide_label": float((hour * 2 + index) % 2),
                "combined_risk_label": float((hour + index) % 2),
                "lead_time_minutes": float(30 + (hour % 5) * 15),
            })
    return pd.DataFrame(rows)


def sample_prediction_payload(sequence_length=4, as_of=None):
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC").floor("1h").isoformat()
    end_time = pd.Timestamp(as_of)
    timestamps = pd.date_range(end=end_time, periods=sequence_length, freq="1h", tz="UTC")
    nodes_payload = []
    for index, node_id in enumerate(["NODE_01", "NODE_02"]):
        readings = []
        for t in timestamps:
            readings.append({
                "timestamp": t.isoformat(),
                "soil_moisture": 0.35,
                "water_level": 1.5,
                "rainfall": 2.0,
                "temperature": 24.0,
                "humidity": 70.0,
                "accelerometer_x": 0.01,
                "accelerometer_y": 0.02,
                "accelerometer_z": 9.8,
                "tilt": 1.0,
                "vibration": 0.02,
                "sentinel1_vv": 0.5,
                "sentinel1_vh": 0.15,
                "ndvi": 0.6,
                "ndwi": 0.25,
            })
        nodes_payload.append({
            "node_id": node_id,
            "latitude": 11.62 + index * 0.01,
            "longitude": 76.08 + index * 0.01,
            "static_features": {
                "elevation": 450.0,
                "slope": 10.0,
                "aspect": 180.0,
                "flow_accumulation": 1000.0,
                "distance_to_river": 100.0,
                "soil_type": "clay_loam",
                "clay_fraction": 0.3,
                "sand_fraction": 0.4,
                "lulc": "forest",
                "historical_flood_frequency": 0.1,
                "historical_landslide_frequency": 0.05,
            },
            "readings": readings,
        })
    return {
        "timestamp": as_of,
        "area_id": "test_area",
        "nodes": nodes_payload,
    }


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_endpoints_return_503_when_no_checkpoint_configured(api, monkeypatch):
    monkeypatch.delenv("MULTI_HAZARD_MODEL_CHECKPOINT", raising=False)
    clear_model_cache()
    clear_predictions()

    # predict-graph
    res = api.post("/digital-twin/multi-hazard/predict-graph", json=sample_prediction_payload())
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "MODEL_NOT_READY"
    assert res.json()["detail"]["status"] == "checkpoint_missing"

    # predict-node
    node_payload = sample_prediction_payload()["nodes"][0]
    res = api.post("/digital-twin/multi-hazard/predict-node", json={
        "timestamp": sample_prediction_payload()["timestamp"],
        "node": node_payload,
        "area_id": "test_area",
    })
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "MODEL_NOT_READY"

    # heatmap latest
    res = api.get("/digital-twin/multi-hazard/heatmap/latest")
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "MODEL_NOT_READY"

    # metrics
    res = api.get("/digital-twin/multi-hazard/metrics")
    assert res.status_code == 200
    assert res.json()["available"] is False
    assert res.json()["message"] == "metrics unavailable - labeled dataset required"


def test_train_pipeline_and_checkpoint_loader(tmp_path, monkeypatch):
    data = synthetic_training_data(periods=50)
    config = resolve_config({
        "model": {"temporal_backbone": "local", "hidden_dim": 32, "heads": 2, "temporal_layers": 1, "gat_layers": 1, "graph_transformer_layers": 1, "dropout": 0.1},
        "training": {"epochs": 2, "patience": 2, "device": "cpu", "mixed_precision": False, "checkpoint_path": str(tmp_path / "model.pt")},
        "data": {"sequence_length": 4, "forecast_horizon_minutes": 60, "train_fraction": 0.6, "val_fraction": 0.2},
    })
    checkpoint = train_frame(data, config, data_provenance="synthetic", output=tmp_path / "synthetic_model.pt")
    assert checkpoint["format_version"] == 1
    assert "model_state_dict" in checkpoint
    assert checkpoint["training"]["completed"] is True

    # Loading synthetic checkpoint marks ready=False because data_provenance != "real"
    bundle = load_checkpoint(tmp_path / "synthetic_model.pt")
    assert bundle.ready is False
    assert "real labeled data is required" in bundle.reason

    # Train with data_provenance="real"
    real_checkpoint = train_frame(data, config, data_provenance="real", output=tmp_path / "real_model.pt")
    assert real_checkpoint["training"]["data_provenance"] == "real"

    real_bundle = load_checkpoint(tmp_path / "real_model.pt")
    assert real_bundle.ready is True
    assert real_bundle.reason is None


def test_real_inference_pipeline_and_geojson_output(tmp_path, monkeypatch, api):
    data = synthetic_training_data(periods=50)
    config = resolve_config({
        "model": {"temporal_backbone": "local", "hidden_dim": 32, "heads": 2, "temporal_layers": 1, "gat_layers": 1, "graph_transformer_layers": 1, "dropout": 0.1},
        "training": {"epochs": 2, "patience": 2, "device": "cpu", "mixed_precision": False},
        "data": {"sequence_length": 4, "forecast_horizon_minutes": 60, "train_fraction": 0.6, "val_fraction": 0.2},
    })
    ckpt_path = tmp_path / "ready_model.pt"
    train_frame(data, config, data_provenance="real", output=ckpt_path)

    monkeypatch.setenv("MULTI_HAZARD_MODEL_CHECKPOINT", str(ckpt_path))
    clear_model_cache()
    clear_predictions()

    # 1. model-info endpoint
    info_res = api.get("/digital-twin/multi-hazard/model-info")
    assert info_res.status_code == 200
    info = info_res.json()
    assert info["status"] == "model_ready"
    assert info["ready"] is True
    assert info["forward_pass_verified"] is True
    assert info["trained_on"] == "real"

    # 2. predict-graph endpoint
    graph_payload = sample_prediction_payload(sequence_length=4)
    pred_res = api.post("/digital-twin/multi-hazard/predict-graph", json=graph_payload)
    assert pred_res.status_code == 200
    result = pred_res.json()
    assert result["model_version"] == info["model_version"]
    assert len(result["nodes"]) == 2
    for node in result["nodes"]:
        assert "node_id" in node
        assert "latitude" in node
        assert "longitude" in node
        assert 0.0 <= node["flood_probability"] <= 1.0
        assert 0.0 <= node["landslide_probability"] <= 1.0
        assert 0.0 <= node["combined_risk"] <= 1.0
        assert node["confidence"] is None or (0.0 <= node["confidence"] <= 1.0)
        assert node["risk_class"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        assert node["lead_time_minutes"] is not None

    # 3. predict-node endpoint
    single_node_payload = {"timestamp": graph_payload["timestamp"], "node": graph_payload["nodes"][0], "area_id": "test_area"}
    node_res = api.post("/digital-twin/multi-hazard/predict-node", json=single_node_payload)
    assert node_res.status_code == 200
    node_result = node_res.json()
    assert node_result["node_id"] == "NODE_01"
    assert 0.0 <= node_result["flood_probability"] <= 1.0

    # 4. GeoJSON heatmap/latest endpoint
    heatmap_res = api.get("/digital-twin/multi-hazard/heatmap/latest?area_id=test_area")
    assert heatmap_res.status_code == 200
    geojson = heatmap_res.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2
    for feat in geojson["features"]:
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Point"
        coords = feat["geometry"]["coordinates"]
        assert len(coords) == 2
        # Check GeoJSON order: [longitude, latitude]
        assert coords[0] == pytest.approx(feat["properties"]["node_id"] == "NODE_01" and 76.08 or 76.09)
        assert coords[1] == pytest.approx(feat["properties"]["node_id"] == "NODE_01" and 11.62 or 11.63)
        props = feat["properties"]
        assert "flood_probability" in props
        assert "landslide_probability" in props
        assert "combined_risk" in props
        assert "confidence" in props
        assert "lead_time_minutes" in props
        assert props["risk_class"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}

    # 5. metrics endpoint
    metrics_res = api.get("/digital-twin/multi-hazard/metrics")
    assert metrics_res.status_code == 200
    assert metrics_res.json()["available"] is True
    assert "flood" in metrics_res.json()["metrics"]
    assert "accuracy" in metrics_res.json()["metrics"]["flood"]
