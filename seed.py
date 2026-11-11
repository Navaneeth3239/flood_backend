"""Idempotent seed: minimal Indian monitoring network + demo accounts.

Run: cd /app/backend && python seed.py
"""
import asyncio

from lib.auth import hash_password
from lib.db import db
from models.schemas import Alert, Gateway, Notification, Report, Sensor, User, Zone

ZONES = [
    dict(name="Chamoli Slope Alpha", state="Uttarakhand", district="Chamoli", lat=30.4083, lng=79.3200,
         radius_km=9.0, hazard_type="landslide", risk_level="high", risk_score=72,
         rainfall_mm=68.0, temperature_c=19.4, humidity_pct=88.0, water_level_m=4.6, soil_moisture_pct=81.0),
    dict(name="Wayanad Ghat Sector", state="Kerala", district="Wayanad", lat=11.6854, lng=76.1320,
         radius_km=7.5, hazard_type="landslide", risk_level="critical", risk_score=84,
         rainfall_mm=96.0, temperature_c=23.1, humidity_pct=93.0, water_level_m=5.4, soil_moisture_pct=92.0),
    dict(name="Brahmaputra Flood Belt", state="Assam", district="Dibrugarh", lat=27.4728, lng=94.9120,
         radius_km=14.0, hazard_type="flood", risk_level="high", risk_score=66,
         rainfall_mm=54.0, temperature_c=28.6, humidity_pct=84.0, water_level_m=6.8, soil_moisture_pct=74.0),
    dict(name="Bandipur Forest Block", state="Karnataka", district="Chamarajanagar", lat=11.7100, lng=76.5300,
         radius_km=11.0, hazard_type="forest_fire", risk_level="medium", risk_score=41,
         rainfall_mm=4.0, temperature_c=36.2, humidity_pct=31.0, water_level_m=0.9, soil_moisture_pct=18.0),
    dict(name="Nilgiri Watershed", state="Tamil Nadu", district="Nilgiris", lat=11.4064, lng=76.6932,
         radius_km=8.0, hazard_type="landslide", risk_level="medium", risk_score=38,
         rainfall_mm=22.0, temperature_c=21.0, humidity_pct=72.0, water_level_m=2.4, soil_moisture_pct=52.0),
    dict(name="Delhi NCR Air Corridor", state="Delhi", district="New Delhi", lat=28.6139, lng=77.2090,
         radius_km=16.0, hazard_type="air_quality", risk_level="low", risk_score=24,
         rainfall_mm=1.0, temperature_c=31.5, humidity_pct=44.0, water_level_m=0.4, soil_moisture_pct=21.0),
]

SENSOR_PLAN = [
    ("rainfall", "Rainfall Gauge", "mm/h", 68.0),
    ("soil_moisture", "Soil Moisture Probe", "%", 81.0),
    ("water_level", "Water Level Radar", "m", 4.6),
    ("temperature", "Temperature Node", "°C", 19.4),
    ("tilt", "Tilt / Landslide Node", "°", 3.2),
    ("smoke", "Smoke & Fire Detector", "ppm", 12.0),
    ("air_quality", "Air Quality Monitor", "AQI", 148.0),
]

ALERTS = [
    dict(code="EIN-CRT-0001", location="Wayanad Ghat Sector, Kerala", hazard_type="landslide",
         risk_level="critical", title="Water level approaching danger threshold",
         detail="Slope-toe water level at 5.4 m against a 5.0 m danger mark; soil saturation at 92%.",
         status="open"),
    dict(code="EIN-HGH-0002", location="Chamoli Slope Alpha, Uttarakhand", hazard_type="landslide",
         risk_level="high", title="Landslide probability detected",
         detail="Random Forest model reports 72% slope-failure probability over the next 12 hours.",
         status="open"),
    dict(code="EIN-MED-0003", location="Brahmaputra Flood Belt, Assam", hazard_type="flood",
         risk_level="medium", title="Heavy rainfall detected",
         detail="54 mm/h sustained rainfall recorded across 6 upstream gauges.",
         status="acknowledged"),
    dict(code="EIN-LOW-0004", location="Bandipur Forest Block, Karnataka", hazard_type="forest_fire",
         risk_level="low", title="Dry fuel load rising",
         detail="Soil moisture at 18% with ambient temperature at 36.2 °C.",
         status="resolved"),
]

USERS = [
    # Convenience test account (password overridden below to "12345678").
    ("Test", "User", "test@gmail.com", "Administrator", "admin", "NEXGI", "Delhi", "New Delhi"),
    ("Arjun", "Mehta", "admin@ein.gov.in", "Administrator", "admin", "National Disaster Management Authority", "Delhi", "New Delhi"),
    ("Priya", "Nair", "officer@ein.gov.in", "Government Official", "gov_officer", "Kerala State Disaster Management Authority", "Kerala", "Wayanad"),
    ("Rakesh", "Bhatt", "field@ein.gov.in", "Forest Officer", "field_officer", "Uttarakhand Forest Department", "Uttarakhand", "Chamoli"),
    ("Meera", "Iyer", "viewer@ein.gov.in", "Environmental Officer", "viewer", "Central Pollution Control Board", "Tamil Nadu", "Nilgiris"),
]


async def main() -> None:
    for coll in ("zones", "gateways", "sensors", "alerts", "notifications", "reports", "simulations"):
        await db[coll].delete_many({})

    # Seed default custom areas if none exist
    custom_areas_count = await db.custom_areas.count_documents({})
    if custom_areas_count == 0:
        await db.custom_areas.insert_many([
            {
                "id": "b3e2c3ce-e6b0-4c37-8f85-5595d23f5d40",
                "name": "Monitored Area 1",
                "district": "Western Himalayas",
                "area_type": "River Basin",
                "risk_category": "High",
                "priority": "Critical (Real-time)",
                "description": "High-risk flash flood and landslide monitored area",
                "lat": 31.039,
                "lng": 78.8938,
                "shape": "Polygon:[[30.948169143221595,78.71978759765625],[31.130844260159883,78.7218475341797],[31.129374846459353,79.06070709228517],[30.947580251654042,79.07272338867189]]",
                "created_at": "2026-09-27T13:31:51.998915+00:00"
            },
            {
                "id": "0004ccb1-ebc6-4c8c-90f6-06ebe52a2e42",
                "name": "Monitored Area 2",
                "district": "Western Himalayas",
                "area_type": "River Basin",
                "risk_category": "Medium",
                "priority": "Normal (Hourly)",
                "description": "Extended catchment monitored boundary",
                "lat": 31.0649,
                "lng": 78.8176,
                "shape": "Polygon:[[31.218673801364655,78.6126708984375],[30.9128292266562,78.61129760742188],[30.9187201197222,79.02877807617189],[31.209277877460135,79.01779174804689]]",
                "created_at": "2026-09-18T10:53:33.348793+00:00"
            }
        ])

    zones = [Zone(**z, sensor_count=0) for z in ZONES]
    await db.zones.insert_many([z.model_dump() for z in zones])

    gateways = []
    sensors = []
    for idx, zone in enumerate(zones):
        gw = Gateway(
            code=f"GW-{idx + 1:02d}",
            name=f"{zone.name} LoRaWAN Gateway",
            zone_name=zone.name,
            lat=zone.lat + 0.02,
            lng=zone.lng + 0.02,
            status="online" if idx != 5 else "offline",
            connected_nodes=0,
            uplink_rate=round(97.0 + idx * 0.4, 1),
        )
        gateways.append(gw)
        count = 5 + (idx % 3)
        for s in range(count):
            stype, label, unit, val = SENSOR_PLAN[s % len(SENSOR_PLAN)]
            sensors.append(
                Sensor(
                    code=f"{zone.district[:3].upper()}-{stype[:3].upper()}-{s + 1:02d}",
                    name=f"{label} {s + 1}",
                    sensor_type=stype,
                    zone_id=zone.id,
                    zone_name=zone.name,
                    lat=round(zone.lat + (s - 2) * 0.012, 5),
                    lng=round(zone.lng + (s - 2) * 0.014, 5),
                    status="online" if (s + idx) % 7 != 0 else "offline",
                    battery=max(22, 100 - (s * 9 + idx * 4)),
                    signal_dbm=-72 - s * 3,
                    last_value=round(val * (0.85 + 0.05 * s), 1),
                    unit=unit,
                    gateway_id=gw.id,
                )
            )
        gw.connected_nodes = count

    await db.gateways.insert_many([g.model_dump() for g in gateways])
    await db.sensors.insert_many([s.model_dump() for s in sensors])
    for zone in zones:
        n = len([s for s in sensors if s.zone_id == zone.id])
        await db.zones.update_one({"id": zone.id}, {"$set": {"sensor_count": n}})

    zone_by_name = {z.name.split(",")[0]: z for z in zones}
    alert_docs = []
    for a in ALERTS:
        z = zone_by_name.get(a["location"].split(",")[0])
        alert_docs.append(Alert(**a, zone_id=z.id if z else "").model_dump())
    await db.alerts.insert_many(alert_docs)

    await db.notifications.insert_many([
        Notification(kind="critical_alert", title="CRITICAL — Wayanad water level",
                     body="Wayanad Ghat Sector · landslide").model_dump(),
        Notification(kind="ai_prediction", title="AI prediction updated",
                     body="Chamoli Slope Alpha risk score raised to 72%.").model_dump(),
        Notification(kind="sensor_offline", title="Sensor offline",
                     body="NEW-RAI-01 has stopped reporting uplinks.").model_dump(),
        Notification(kind="system_update", title="Platform updated to v3.4.2",
                     body="Multi-agent orchestrator refreshed.", read=True).model_dump(),
    ])

    await db.reports.insert_many([
        Report(title="Monsoon Landslide Susceptibility — Chamoli", report_type="Risk Assessment",
               period="Last 30 days", zone_name="Chamoli Slope Alpha", size_kb=412).model_dump(),
        Report(title="Brahmaputra Flood Watch Summary", report_type="Hazard Summary",
               period="Last 7 days", zone_name="Brahmaputra Flood Belt", size_kb=268).model_dump(),
    ])

    for first, last, email, designation, role, org, state, district in USERS:
        existing = await db.users.find_one({"email": email})
        if not existing:
            doc = User(
                first_name=first, last_name=last, email=email, phone="+91 98000 00000",
                organization=org, designation=designation, role=role, state=state,
                district=district, status="active", verified=True,
            ).model_dump()
            doc["password_hash"] = hash_password("12345678" if email == "test@gmail.com" else "Gov@12345")
            await db.users.insert_one(doc)

    print(f"seeded: {len(zones)} zones, {len(gateways)} gateways, {len(sensors)} sensors, "
          f"{len(alert_docs)} alerts, {len(USERS)} users")


if __name__ == "__main__":
    asyncio.run(main())
