"""Criterion: Login validates input, shows professional errors, and signs an officer in."""


def test_login_wrong_password_rejected(client):
    resp = client.post(
        "/auth/login",
        json={"email": "admin@ein.gov.in", "password": "WrongPass@123"},
    )
    assert resp.status_code in (400, 401), resp.text
    body = resp.json()
    detail = str(body.get("detail", body)).lower()
    assert "invalid" in detail


def test_login_wrong_password_rejected(client):
    resp = client.post(
        "/auth/login",
        json={"email": "admin@ein.gov.in", "password": "WrongPass@123"},
    )
    assert resp.status_code in (400, 401), resp.text
    body = resp.json()
    detail = str(body.get("detail", body)).lower()
    assert "invalid" in detail


def test_login_correct_credentials_returns_admin_profile(client):
    resp = client.post(
        "/auth/login",
        json={"email": "admin@ein.gov.in", "password": "Gov@12345"},
    )
    assert resp.status_code == 200, resp.text
    user = resp.json()
    assert user.get("role") == "admin"
    assert user.get("email") == "admin@ein.gov.in"
    # session cookie must be issued for subsequent authenticated calls
    assert "ein_session" in resp.cookies or "ein_session" in client.cookies
