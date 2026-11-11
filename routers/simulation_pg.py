"""
FastAPI router — saves flash flood simulation results to external PostgreSQL
sensor_db (same DB as LoRaWAN node data) and exposes read endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import uuid

from lib.auth import current_user, require_roles
from services.simulation_pg_service import (
    ensure_tables,
    save_simulation_run,
    get_simulation_runs,
    get_flood_alerts,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulation-pg", tags=["simulation-pg"])

# Run DDL once at import time (non-fatal if DB unreachable)
try:
    ensure_tables()
except Exception as _exc:
    logger.warning("simulation_pg router: ensure_tables skipped: %s", _exc)


# ─── Request / Response models ────────────────────────────────────────────────

class SimPgScenario(BaseModel):
    rainfallMmH:        float = 150.0
    floodIntensity:     float = 100.0
    durationMinutes:    int   = 60
    soilSaturation:     float = 80.0
    infiltrationMmH:    float = 5.0
    roughness:          float = 0.035
    windSpeedKmh:       float = 20.0
    riverRiseM:         float = 1.2
    flowModel:          str   = "physics"


class SimPgSummary(BaseModel):
    elapsed_seconds:    int   = 0
    spread_area_ha:     float = 0.0
    max_depth_m:        float = 0.0
    total_volume_m3:    float = 0.0
    buildings_exposed:  int   = 0
    buildings_critical: int   = 0
    buildings_safe:     int   = 0


class SimPgSaveRequest(BaseModel):
    run_id:           str   = Field(default_factory=lambda: str(uuid.uuid4()))
    title:            str   = "Flash Flood Simulation"
    period:           str   = ""
    zone_name:        str   = ""
    started_at:       datetime
    ended_at:         datetime
    scenario:         SimPgScenario = Field(default_factory=SimPgScenario)
    summary:          SimPgSummary  = Field(default_factory=SimPgSummary)
    buildings:        List[Dict[str, Any]] = Field(default_factory=list, max_length=100000)
    settings_history: List[Dict[str, Any]] = Field(default_factory=list, max_length=10000)
    method:           str   = "local-inertial-shallow-water"
    center_lat:       Optional[float] = None
    center_lng:       Optional[float] = None
    polygon_coords:   Optional[Any]   = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/save", status_code=201)
async def save_to_postgres(
    payload: SimPgSaveRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(current_user),
):
    """
    Save a completed flash flood simulation run to the external
    PostgreSQL sensor_db alongside LoRaWAN telemetry data.
    Heavy blobs (buildings array) are processed in a background task
    so the HTTP response returns instantly.
    """
    def _save():
        try:
            save_simulation_run(
                run_id          = payload.run_id,
                user_id         = user.get("id", "anonymous"),
                title           = payload.title,
                period          = payload.period,
                zone_name       = payload.zone_name,
                started_at      = payload.started_at,
                ended_at        = payload.ended_at,
                scenario        = payload.scenario.model_dump(),
                buildings       = payload.buildings,
                settings_history= payload.settings_history,
                summary         = payload.summary.model_dump(),
                method          = payload.method,
                center_lat      = payload.center_lat,
                center_lng      = payload.center_lng,
                polygon_coords  = payload.polygon_coords,
            )
        except Exception as exc:
            logger.error("background save_to_postgres failed: %s", exc)

    background_tasks.add_task(_save)
    return {
        "status": "queued",
        "run_id": payload.run_id,
        "database": "Simulation Database (PostgreSQL)",
        "message": "Simulation results are being written to PostgreSQL in the background.",
    }


@router.post("/save-sync", status_code=201)
async def save_to_postgres_sync(
    payload: SimPgSaveRequest,
    user: dict = Depends(require_roles("admin", "gov_officer")),
):
    """
    Synchronous version — waits for the DB write to complete.
    Use this when you need the simulation_id in the response.
    """
    try:
        sim_id = save_simulation_run(
            run_id          = payload.run_id,
            user_id         = user.get("id", "anonymous"),
            title           = payload.title,
            period          = payload.period,
            zone_name       = payload.zone_name,
            started_at      = payload.started_at,
            ended_at        = payload.ended_at,
            scenario        = payload.scenario.model_dump(),
            buildings       = payload.buildings,
            settings_history= payload.settings_history,
            summary         = payload.summary.model_dump(),
            method          = payload.method,
            center_lat      = payload.center_lat,
            center_lng      = payload.center_lng,
            polygon_coords  = payload.polygon_coords,
        )
        return {
            "status": "saved",
            "simulation_id": sim_id,
            "run_id": payload.run_id,
            "database": "Simulation Database (PostgreSQL)",
        }
    except Exception as exc:
        logger.error("save_to_postgres_sync failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"PostgreSQL write failed: {exc}")


@router.get("/runs")
async def list_runs(
    limit: int = Query(50, ge=1, le=500),
    user: dict = Depends(current_user),
):
    """List recent simulation runs stored in database."""
    return get_simulation_runs(limit=limit)


@router.get("/alerts")
async def list_flood_alerts(
    limit: int = Query(100, ge=1, le=500),
    user: dict = Depends(current_user),
):
    """List auto-triggered flood alerts from simulation runs."""
    return get_flood_alerts(limit=limit)


@router.get("/status")
async def db_status(user: dict = Depends(current_user)):
    """Check connectivity to database and return table row counts."""
    from services.simulation_pg_service import _conn
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM public.simulation_runs")
                sim_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM public.flood_alerts_log")
                alert_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM public.building_flood_exposure")
                bfe_count = cur.fetchone()[0]
        return {
            "connected": True,
            "database": "Simulation Database (PostgreSQL)",
            "simulation_runs": sim_count,
            "flood_alerts": alert_count,
            "building_exposures": bfe_count,
        }
    except Exception as exc:
        return {"connected": False, "error": str(exc)}
