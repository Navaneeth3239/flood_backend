from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.external_sensors import router
from services import external_sensor_service as sensors
import pytest
from unittest.mock import Mock


def test_sensor_outage_returns_unavailable_instead_of_fabricated_readings(monkeypatch):
    def offline():
        raise ConnectionError("offline")
    monkeypatch.setattr(sensors, "get_connection", offline)
    app = FastAPI()
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        for path in ("history?limit=30", "packets?limit=100"):
            response = client.get("/api/external-sensors/" + path)
            assert response.status_code == 503
            assert "SENSOR_DB_URL" in response.json()["detail"]
            assert response.headers["x-sensor-error"] == "SENSOR_DB_UNAVAILABLE"
        assert client.get("/api/external-sensors/summary").json()["connected"] is False
        assert client.get("/api/external-sensors/latest").json()["has_data"] is False


@pytest.mark.parametrize("value", ["", "  ", '""'])
def test_empty_dashboard_variable_never_connects_to_a_local_socket(monkeypatch, value):
    monkeypatch.setenv("SENSOR_DB_URL", value)
    driver = Mock()
    monkeypatch.setattr(sensors, "psycopg", driver)
    sensors.get_connection()
    driver.connect.assert_called_once_with(sensors.DEFAULT_SENSOR_DB_URL, connect_timeout=8)


def test_copied_env_quotes_are_removed(monkeypatch):
    monkeypatch.setenv("SENSOR_DB_URL", '  "postgresql://user:secret@db.example:5432/sensors"  ')
    assert sensors.sensor_database_url() == "postgresql://user:secret@db.example:5432/sensors"


def test_simulations_use_the_same_url_normalization(monkeypatch):
    from services.simulation_pg_service import _conn
    monkeypatch.setenv("SENSOR_DB_URL", "")
    driver = Mock()
    monkeypatch.setattr(sensors, "psycopg", driver)
    _conn()
    driver.connect.assert_called_once_with(sensors.DEFAULT_SENSOR_DB_URL, connect_timeout=8)


def test_invalid_uri_and_auth_failures_do_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("SENSOR_DB_URL", "https://user:supersecret@wrong.example")
    with pytest.raises(sensors.SensorDatabaseUnavailable) as raised:
        sensors.sensor_database_url()
    assert raised.value.code == "SENSOR_DB_CONFIG_INVALID"
    assert "supersecret" not in str(raised.value)
    error = sensors.sensor_database_error(RuntimeError("password authentication failed for supersecret"))
    assert error.code == "SENSOR_DB_AUTH_FAILED"
    assert "supersecret" not in str(error)


def test_missing_driver_is_visible_in_summary(monkeypatch):
    monkeypatch.setattr(sensors, "psycopg", None)
    result = sensors.get_live_sensor_summary()
    assert result["connected"] is False
    assert result["error_code"] == "SENSOR_DB_DRIVER_MISSING"
