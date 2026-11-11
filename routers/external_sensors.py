"""
Router for external PostgreSQL sensor database (sensor_db)
Provides real-time LoRaWAN node telemetry and packet feeds.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Depends, HTTPException

from lib.auth import optional_user
from services.external_sensor_service import (
    get_live_sensor_summary,
    get_sensor_history,
    get_lora_packets,
    get_node_latest_reading,
    record_sensor_alert,
    get_sensor_alerts,
    SensorDatabaseUnavailable,
)

router = APIRouter(prefix="/external-sensors", tags=["external-sensors"])


@router.get("/summary")
def get_summary(user: Optional[dict] = Depends(optional_user)):
    """
    Get live status and telemetry overview from sensor_db.
    """
    return get_live_sensor_summary()


@router.get("/node/{node_id}")
def get_node_telemetry(
    node_id: str,
    user: Optional[dict] = Depends(optional_user),
):
    """
    Get live telemetry data sent by a specific node (e.g. node1, LORA_NODE_1).
    Returns real telemetry or 0s if no data exists.
    """
    return get_node_latest_reading(node_id)


@router.get("/latest")
def get_latest_default_node(
    node_id: str = Query("node1", description="Node identifier"),
    user: Optional[dict] = Depends(optional_user),
):
    """
    Get latest telemetry for default node (node1 / slave 1).
    """
    return get_node_latest_reading(node_id)


@router.get("/history")
def get_history(
    device_id: Optional[str] = Query(None, description="Filter by device ID (e.g. LORA_NODE_1)"),
    limit: int = Query(1000, ge=1, le=2000, description="Max rows to return"),
    user: Optional[dict] = Depends(optional_user),
):
    """
    Get historical sensor telemetry readings from sensor_data table.
    """
    try:
        return get_sensor_history(device_id=device_id, limit=limit)
    except SensorDatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"X-Sensor-Error": exc.code}) from exc


@router.get("/packets")
def get_packets(
    limit: int = Query(50, ge=1, le=200, description="Max packets to return"),
    user: Optional[dict] = Depends(optional_user),
):
    """
    Get raw LoRaWAN packet logs from lora_packets table.
    """
    try:
        return get_lora_packets(limit=limit)
    except SensorDatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"X-Sensor-Error": exc.code}) from exc


@router.post("/alerts")
async def create_sensor_alert(payload: dict, user: Optional[dict] = Depends(optional_user)):
    """
    Log a disaster alert (Flash Flood or Landslide) triggered by live sensor readings into PostgreSQL & MongoDB.
    """
    return record_sensor_alert(
        disaster_type=payload.get("disaster_type", "flash_flood"),
        alert_level=payload.get("alert_level", "critical"),
        message=payload.get("message", ""),
        zone_name=payload.get("zone_name", "Digital Twin Monitored Basin"),
        sensor_id=payload.get("sensor_id", "LORA_NODE_1"),
        soil_moisture=float(payload.get("soil_moisture") or 0.0),
        water_level_mm=float(payload.get("water_level_mm") or 0.0),
        tilt=float(payload.get("tilt") or 0.0),
        imu_mag=float(payload.get("imu_mag") or 0.0),
    )


@router.get("/alerts")
async def list_sensor_alerts(
    limit: int = Query(50, ge=1, le=200),
    user: Optional[dict] = Depends(optional_user),
):
    """
    Retrieve all disaster alerts logged in PostgreSQL database.
    """
    return get_sensor_alerts(limit=limit)
