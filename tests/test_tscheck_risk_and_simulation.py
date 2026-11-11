"""Criterion: AI Risk Assessment and the Digital Twin simulator produce real computed output."""


def _login_admin(client):
    resp = client.post("/auth/login", json={"email": "admin@ein.gov.in", "password": "Gov@12345"})
    assert resp.status_code == 200, resp.text


def test_risk_endpoint_orders_zones_and_includes_disclaimer(client):
    _login_admin(client)
    resp = client.get("/risk")
    assert resp.status_code == 200, resp.text
    zones = resp.json()
    assert len(zones) == 6, zones
    scores = [z["risk_score"] for z in zones]
    assert scores == sorted(scores, reverse=True), scores
    top = zones[0]
    assert top["zone_name"] == "Wayanad Ghat Sector", top
    assert len(top["factors"]) == 5, top["factors"]
    assert "do not replace" in top["disclaimer"].lower()
    assert 0 <= top["confidence"] <= 100


def test_digital_twin_simulation_varies_with_params(client):
    _login_admin(client)
    zones = client.get("/zones").json()
    wayanad = next(z for z in zones if z["name"] == "Wayanad Ghat Sector")

    low = client.post(
        "/simulations",
        json={"zone_id": wayanad["id"], "scenario": "landslide", "rainfall_intensity": 20, "soil_saturation": 30},
    )
    high = client.post(
        "/simulations",
        json={"zone_id": wayanad["id"], "scenario": "landslide", "rainfall_intensity": 190, "soil_saturation": 95},
    )
    assert low.status_code == 201, low.text
    assert high.status_code == 201, high.text
    low_body, high_body = low.json(), high.json()
    assert wayanad["name"] in low_body["summary"]
    assert low_body["peak_impact_pct"] != high_body["peak_impact_pct"], (low_body, high_body)
    assert "severity" in low_body and "evacuation_time_min" in low_body
