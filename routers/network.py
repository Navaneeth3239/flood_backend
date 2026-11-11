"""Sensors, gateways, zones, telemetry and network statistics."""
import asyncio
import math
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from lib.auth import current_user, require_roles
from lib.db import db
from models.schemas import (
    Gateway,
    MessageResponse,
    NetworkStats,
    Sensor,
    SensorCreate,
    SensorUpdate,
    TelemetryPoint,
    TelemetrySeries,
    Zone,
)

router = APIRouter(tags=["network"])

UNITS = {
    "rainfall": "mm/h",
    "soil_moisture": "%",
    "temperature": "°C",
    "water_level": "m",
    "smoke": "ppm",
    "air_quality": "AQI",
    "tilt": "°",
}


@router.get("/zones", response_model=List[Zone])
async def list_zones(user: dict = Depends(current_user)):
    docs = await db.zones.find({}, {"_id": 0}).to_list(500)
    return [Zone(**d) for d in docs]


@router.get("/zones/{zone_id}", response_model=Zone)
async def get_zone(zone_id: str, user: dict = Depends(current_user)):
    doc = await db.zones.find_one({"id": zone_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Monitoring zone not found")
    return Zone(**doc)


@router.get("/gateways", response_model=List[Gateway])
async def list_gateways(user: dict = Depends(current_user)):
    docs = await db.gateways.find({}, {"_id": 0}).to_list(200)
    return [Gateway(**d) for d in docs]


import logging
logger = logging.getLogger(__name__)


@router.get("/sensors", response_model=List[Sensor])
async def list_sensors(
    zone_id: Optional[str] = Query(default=None),
    sensor_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    user: dict = Depends(current_user),
):
    query: dict = {}
    if zone_id:
        query["zone_id"] = zone_id
    if sensor_type:
        query["sensor_type"] = sensor_type
    if status:
        query["status"] = status
    docs = await db.sensors.find(query, {"_id": 0}).to_list(1000)
    sensors = [Sensor(**d) for d in docs]

    # Enrich with live real LoRaWAN hydrology node from external sensor_db
    try:
        from services.external_sensor_service import get_live_sensor_summary
        summary = get_live_sensor_summary()
        if summary.get("connected") and summary.get("devices"):
            for dev in summary["devices"]:
                latest = dev.get("latest", {})
                sensors.insert(0, Sensor(
                    id=f"ext-lora-{dev['device_id']}",
                    code=dev["device_id"],
                    name=dev["name"],
                    sensor_type="water_level",
                    zone_id="zone-pollachi-monitored",
                    zone_name="Monitored Basin (Pollachi Sector)",
                    lat=10.667366,
                    lng=77.016900,
                    status="online",
                    battery=dev.get("battery_pct", 95),
                    signal_dbm=int(latest.get("rssi_dbm") or -108),
                    last_value=float(latest.get("water_level_mm") or 94.0),
                    unit="mm",
                    updated_at=datetime.fromisoformat(latest["created_at"]) if latest.get("created_at") else datetime.now(timezone.utc),
                ))
    except Exception as e:
        logger.warning(f"Could not merge external sensor: {e}")

    return sensors


@router.post("/sensors", response_model=Sensor, status_code=201)
async def create_sensor(payload: SensorCreate, user: dict = Depends(require_roles("admin"))):
    if await db.sensors.find_one({"code": payload.code}):
        raise HTTPException(status_code=409, detail="A sensor with this code already exists.")
    zone = await db.zones.find_one({"id": payload.zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Monitoring zone not found")
    sensor = Sensor(
        **payload.model_dump(exclude={"unit"}),
        zone_name=zone["name"],
        unit=payload.unit or UNITS.get(payload.sensor_type, ""),
    )
    await db.sensors.insert_one(sensor.model_dump())
    await db.zones.update_one({"id": payload.zone_id}, {"$inc": {"sensor_count": 1}})
    return sensor


@router.patch("/sensors/{sensor_id}", response_model=Sensor)
async def update_sensor(
    sensor_id: str, payload: SensorUpdate, user: dict = Depends(require_roles("admin", "field_officer"))
):
    changes = {k: v for k, v in payload.model_dump().items() if v is not None}
    changes["updated_at"] = datetime.now(timezone.utc)
    result = await db.sensors.update_one({"id": sensor_id}, {"$set": changes})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Sensor not found")
    doc = await db.sensors.find_one({"id": sensor_id}, {"_id": 0})
    return Sensor(**doc)


@router.delete("/sensors/{sensor_id}", response_model=MessageResponse)
async def delete_sensor(sensor_id: str, user: dict = Depends(require_roles("admin"))):
    doc = await db.sensors.find_one({"id": sensor_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Sensor not found")
    await db.sensors.delete_one({"id": sensor_id})
    await db.zones.update_one({"id": doc["zone_id"]}, {"$inc": {"sensor_count": -1}})
    return MessageResponse(message=f"Sensor {doc['code']} decommissioned")


@router.get("/stats", response_model=NetworkStats)
async def network_stats(user: dict = Depends(current_user)):
    # Execute all 7 count queries concurrently rather than sequentially
    total, active, gateways, alerts, zones, critical, high = await asyncio.gather(
        db.sensors.count_documents({}),
        db.sensors.count_documents({"status": "online"}),
        db.gateways.count_documents({"status": "online"}),
        db.alerts.count_documents({"status": {"$ne": "resolved"}}),
        db.zones.count_documents({}),
        db.alerts.count_documents({"status": {"$ne": "resolved"}, "risk_level": "critical"}),
        db.alerts.count_documents({"status": {"$ne": "resolved"}, "risk_level": "high"}),
    )
    level = "CRITICAL" if critical else "HIGH" if high else "MODERATE" if alerts else "LOW"
    return NetworkStats(
        active_sensors=active,
        total_sensors=total,
        online_gateways=gateways,
        active_alerts=alerts,
        monitoring_zones=zones,
        data_streams=active * 5,
        national_hazard_level=level,
    )




@router.get("/telemetry", response_model=TelemetrySeries)
async def telemetry(
    zone_id: Optional[str] = Query(default=None),
    hours: int = Query(default=24, ge=6, le=72),
    user: dict = Depends(current_user),
):
    """Deterministic 24h telemetry series derived server-side from the zone's live readings."""
    base = {"rainfall_mm": 12.0, "water_level_m": 3.2, "temperature_c": 24.0, "soil_moisture_pct": 42.0}
    if zone_id:
        zone = await db.zones.find_one({"id": zone_id}, {"_id": 0})
        if not zone:
            raise HTTPException(status_code=404, detail="Monitoring zone not found")
        base = {
            "rainfall_mm": zone.get("rainfall_mm") or 12.0,
            "water_level_m": zone.get("water_level_m") or 3.2,
            "temperature_c": zone.get("temperature_c") or 24.0,
            "soil_moisture_pct": zone.get("soil_moisture_pct") or 42.0,
        }
    points: List[TelemetryPoint] = []
    for i in range(hours):
        hour = (24 - hours + i) % 24
        wave = math.sin((i / hours) * math.pi * 2)
        points.append(
            TelemetryPoint(
                label=f"{hour:02d}:00",
                rainfall_mm=round(max(0.0, base["rainfall_mm"] * (0.55 + 0.45 * math.sin(i / 3.0))), 1),
                water_level_m=round(base["water_level_m"] * (0.9 + 0.12 * wave), 2),
                temperature_c=round(base["temperature_c"] + 4.0 * math.sin((i - 4) / 3.8), 1),
                soil_moisture_pct=round(min(100.0, base["soil_moisture_pct"] * (0.92 + 0.14 * wave)), 1),
                packet_rate=round(96.0 + 3.5 * math.cos(i / 2.5), 1),
            )
        )
    return TelemetrySeries(zone_id=zone_id, points=points)
