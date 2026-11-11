"""Criterion: Alerts Center filters and the acknowledge/assign/resolve workflow mutate real state.

Note: the app has no endpoint to create alerts (they are seeded), so this test acts on the
seeded EIN-HGH-0002 alert (not the EIN-CRT-0001 alert relied on by the "exactly 1 critical"
filter assertion). `seed.py` is idempotent and wipes+reinserts, so this is restorable.
"""


def _login_admin(client):
    resp = client.post("/auth/login", json={"email": "admin@ein.gov.in", "password": "Gov@12345"})
    assert resp.status_code == 200, resp.text


def _find_alert(client, code):
    alerts = client.get("/alerts").json()
    return next(a for a in alerts if a["code"] == code)


def test_critical_filter_leaves_exactly_one_alert(client):
    _login_admin(client)
    alerts = client.get("/alerts").json()
    critical = [a for a in alerts if a["risk_level"] == "critical"]
    assert len(critical) == 1, critical
    assert critical[0]["code"] == "EIN-CRT-0001"


def test_acknowledge_assign_resolve_workflow_persists(client):
    _login_admin(client)
    alert = _find_alert(client, "EIN-HGH-0002")
    alert_id = alert["id"]
    assert alert["status"] == "open", alert

    ack = client.post(f"/alerts/{alert_id}/action", json={"action": "acknowledge"})
    assert ack.status_code == 200, ack.text
    assert ack.json()["status"] == "acknowledged"

    assign = client.post(
        f"/alerts/{alert_id}/action",
        json={"action": "assign", "assigned_to": "TScheck Duty Officer"},
    )
    assert assign.status_code == 200, assign.text
    assign_body = assign.json()
    assert assign_body["status"] == "assigned"
    assert assign_body["assigned_to"] == "TScheck Duty Officer"

    resolve = client.post(f"/alerts/{alert_id}/action", json={"action": "resolve"})
    assert resolve.status_code == 200, resolve.text
    assert resolve.json()["status"] == "resolved"

    # persists across a fresh read (simulates page reload)
    reread = _find_alert(client, "EIN-HGH-0002")
    assert reread["status"] == "resolved", reread

    stats = client.get("/stats").json()
    assert stats["active_alerts"] == 2, stats  # was 3 unresolved, one just resolved

    # restore seeded state for other checks in this run
    client.post(f"/alerts/{alert_id}/action", json={"action": "reopen"})
