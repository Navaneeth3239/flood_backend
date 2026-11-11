"""Criterion: Dashboard shows the 5 KPI cards backed by real seeded stats."""


def _login_admin(client):
    resp = client.post("/auth/login", json={"email": "admin@ein.gov.in", "password": "Gov@12345"})
    assert resp.status_code == 200, resp.text  # cookie stored on the shared client


def test_dashboard_stats_match_seeded_counts(client):
    _login_admin(client)
    resp = client.get("/stats")
    assert resp.status_code == 200, resp.text
    stats = resp.json()
    assert stats.get("active_sensors") == 31, stats
    assert stats.get("total_sensors") == 36, stats
    assert stats.get("online_gateways") == 5, stats
    assert stats.get("active_alerts") == 3, stats
    assert stats.get("monitoring_zones") == 6, stats
    assert stats.get("data_streams") == 155, stats


def test_alerts_endpoint_lists_seeded_unresolved_alerts(client):
    _login_admin(client)
    resp = client.get("/alerts")
    assert resp.status_code == 200, resp.text
    alerts_list = resp.json()
    codes = {a.get("code") for a in alerts_list}
    assert "EIN-CRT-0001" in codes, alerts_list
    unresolved = [a for a in alerts_list if a.get("status") != "resolved"]
    assert len(unresolved) == 3, alerts_list
