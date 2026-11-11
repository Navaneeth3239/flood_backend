import pytest
from datetime import datetime, timedelta

def test_satellite_status(client):
    """GET /api/satellite/status should return engine health and project id"""
    response = client.get("/satellite/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "project_id" in data
    assert data["project_id"] == "formal-purpose-466115-i8"
