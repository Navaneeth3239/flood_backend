"""Criterion: Role-based access control gates every restricted route / admin-only mutations work."""
import uuid


def _login(client, email, password="Gov@12345"):
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_viewer_forbidden_on_admin_endpoints(client):
    _login(client, "viewer@ein.gov.in")
    resp = client.get("/users")
    assert resp.status_code == 403, resp.text


def test_field_officer_can_read_sensors_but_not_users(client):
    _login(client, "field@ein.gov.in")
    sensors_resp = client.get("/sensors")
    assert sensors_resp.status_code == 200, sensors_resp.text
    sensors = sensors_resp.json()
    assert len(sensors) == 36, len(sensors)

    users_resp = client.get("/users")
    assert users_resp.status_code == 403, users_resp.text


def test_admin_can_commission_and_decommission_sensor(client):
    _login(client, "admin@ein.gov.in")
    zones = client.get("/zones").json()
    zone = zones[0]
    code = f"TSCHECK-{uuid.uuid4().hex[:8].upper()}"

    before_count = len(client.get("/sensors").json())

    create_resp = client.post(
        "/sensors",
        json={
            "code": code,
            "name": "TScheck Fixture Sensor",
            "sensor_type": "rainfall",
            "zone_id": zone["id"],
            "lat": zone["lat"],
            "lng": zone["lng"],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    new_sensor = create_resp.json()
    assert new_sensor["code"] == code

    after_count = len(client.get("/sensors").json())
    assert after_count == before_count + 1

    delete_resp = client.delete(f"/sensors/{new_sensor['id']}")
    assert delete_resp.status_code == 200, delete_resp.text

    final_count = len(client.get("/sensors").json())
    assert final_count == before_count


def test_admin_can_change_user_role_and_suspend(client):
    _login(client, "admin@ein.gov.in")
    users = client.get("/users").json()
    officer = next(u for u in users if u["email"] == "officer@ein.gov.in")

    patch_resp = client.patch(f"/users/{officer['id']}", json={"status": "suspended"})
    assert patch_resp.status_code == 200, patch_resp.text
    assert patch_resp.json()["status"] == "suspended"

    restore_resp = client.patch(f"/users/{officer['id']}", json={"status": "active"})
    assert restore_resp.status_code == 200, restore_resp.text
    assert restore_resp.json()["status"] == "active"
