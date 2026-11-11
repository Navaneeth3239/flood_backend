"""Report persistence contract, retries, validation and role boundaries."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from lib.auth import current_user
from routers import admin


@pytest.fixture
def report_api(monkeypatch):
    saved = {}

    async def find(query, *args):
        return saved.get(query["id"])

    async def insert(document):
        saved[document["id"]] = document

    collection = SimpleNamespace(find_one=AsyncMock(side_effect=find), insert_one=AsyncMock(side_effect=insert))
    monkeypatch.setattr(admin, "db", SimpleNamespace(reports=collection))
    app = FastAPI()
    app.include_router(admin.router)
    app.dependency_overrides[current_user] = lambda: {"id": "officer-1", "role": "gov_officer"}
    return TestClient(app), app, saved, collection


def payload():
    return {"title": "Test flood simulation", "period": "120 simulated seconds", "zone_name": "Test area",
            "simulation_report": {"runId": "865d4323-ce9d-470b-a853-ff1cd080758a",
                "startedAt": "2026-09-17T00:00:00Z", "endedAt": "2026-09-17T00:02:00Z",
                "scenario": {"elapsedSeconds": 120, "rainfallMmH": 100, "graph": {"nodes": 4, "edges": 4}},
                "settingsHistory": [{"elapsedSeconds": 0, "settings": {"rainfallMmH": 100}}],
                "buildings": [{"id": "house", "arrivalSeconds": 50, "peakDepthM": 0.2}],
                "summary": {"loaded": 1, "assessed": 1, "affectedDuringRun": 1}, "method": "Test fixture"}}


def test_persists_complete_snapshot_and_retries_without_duplicates(report_api):
    client, app, saved, collection = report_api
    response = client.post("/reports/simulation", json=payload())
    assert response.status_code == 201
    report = response.json()
    assert report["report_type"] == "Flood Simulation"
    assert report["simulation_report"]["buildings"][0]["arrivalSeconds"] == 50
    assert report["size_kb"] > 0
    retry = client.post("/reports/simulation", json=payload())
    assert retry.json() == report
    assert len(saved) == 1
    assert collection.insert_one.await_count == 1


def test_separate_users_do_not_collide(report_api):
    client, app, saved, collection = report_api
    first = client.post("/reports/simulation", json=payload()).json()
    app.dependency_overrides[current_user] = lambda: {"id": "officer-2", "role": "admin"}
    second = client.post("/reports/simulation", json=payload()).json()
    assert first["id"] != second["id"]
    assert len(saved) == 2


def test_save_failure_is_retryable_and_does_not_claim_success(report_api):
    client, app, saved, collection = report_api
    collection.insert_one.side_effect = RuntimeError("Database unavailable")
    assert client.post("/reports/simulation", json=payload()).status_code == 503
    assert not saved


def test_role_and_invalid_snapshot_rejected(report_api):
    client, app, saved, collection = report_api
    invalid = deepcopy(payload())
    invalid["simulation_report"]["runId"] = "invalid"
    assert client.post("/reports/simulation", json=invalid).status_code == 422
    app.dependency_overrides[current_user] = lambda: {"id": "viewer", "role": "viewer"}
    assert client.post("/reports/simulation", json=payload()).status_code == 403
    assert not saved
