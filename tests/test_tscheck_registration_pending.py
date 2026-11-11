"""Criterion: Registration creates a PENDING account that cannot log in until admin activates it."""
import uuid


def _unique_email():
    return f"tscheck-reg-{uuid.uuid4().hex[:10]}@ein.gov.in"


def _payload(email, password="Gov@12345", confirm=None):
    return {
        "first_name": "TScheck",
        "last_name": "RegOfficer",
        "email": email,
        "password": password,
        "confirm_password": confirm if confirm is not None else password,
        "designation": "Field Officer",
        "organization": "Environmental Monitoring",
        "phone": "9876543210",
        "state": "Kerala",
        "district": "Wayanad",
    }


def test_register_creates_pending_and_blocks_login(client):
    email = _unique_email()
    resp = client.post("/auth/register", json=_payload(email))
    assert resp.status_code in (200, 201), resp.text
    body = resp.json()
    user = body.get("user", body)
    assert user.get("status") == "pending", body
    assert user.get("email") == email
    assert body.get("requires_verification") is True

    login_resp = client.post("/auth/login", json={"email": email, "password": "Gov@12345"})
    assert login_resp.status_code in (400, 401, 403), login_resp.text
    detail = str(login_resp.json().get("detail", "")).lower()
    assert "verif" in detail or "pending" in detail


# NOTE: password/confirm mismatch is validated client-side only (inline "Passwords do not
# match." before submission) per the criterion; that flow is covered by the
# registration-creates-pending-account browser check instead of a backend assertion here.
