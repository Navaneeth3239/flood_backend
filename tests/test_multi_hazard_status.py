from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.digital_twin import router
from services import multi_hazard_status


def test_model_info_never_marks_an_unconfigured_model_ready(monkeypatch):
    monkeypatch.delenv("MULTI_HAZARD_MODEL_CHECKPOINT", raising=False)
    result = multi_hazard_status.model_info()
    assert result["ready"] is False
    assert result["status"] == "checkpoint_missing"
    assert result["heatmap"]["format"] == "GeoJSON FeatureCollection"


def test_model_info_rejects_an_unloadable_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not loaded by status endpoint")
    monkeypatch.setenv("MULTI_HAZARD_MODEL_CHECKPOINT", str(checkpoint))
    result = multi_hazard_status.model_info()
    assert result["ready"] is False
    assert result["status"] == "checkpoint_invalid"


def test_model_info_endpoint_is_available():
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/digital-twin/multi-hazard/model-info")
    assert response.status_code == 200
    assert "architecture" in response.json()
