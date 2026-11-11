"""
Writes flash flood simulation results to the external PostgreSQL sensor_db.
Connection reuses the same DB used for LoRaWAN telemetry:
  postgresql://sensor_user:nexgi@db.nishanth.qzz.io:5432/sensor_db

Tables written:
  public.simulation_runs           -- one row per simulation run
  public.flood_alerts_log          -- auto-triggered when max_depth >= 0.5 m
  public.building_flood_exposure   -- per-building exposure (top 500 most exposed)
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.external_sensor_service import get_connection

logger = logging.getLogger(__name__)


def _conn():
    # Share URL normalization with telemetry, including empty Vercel env values.
    return get_connection()


def ensure_tables() -> None:
    """Create tables if they do not yet exist (idempotent)."""
    ddl = """
    CREATE TABLE IF NOT EXISTS public.simulation_runs (
        id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        run_id              UUID         NOT NULL UNIQUE,
        user_id             TEXT         NOT NULL,
        zone_name           TEXT         NOT NULL DEFAULT '',
        title               TEXT         NOT NULL,
        period              TEXT         NOT NULL DEFAULT '',
        rainfall_mm_h       DOUBLE PRECISION NOT NULL DEFAULT 0,
        flood_intensity_pct DOUBLE PRECISION NOT NULL DEFAULT 100,
        duration_minutes    INT          NOT NULL DEFAULT 60,
        soil_saturation_pct DOUBLE PRECISION NOT NULL DEFAULT 80,
        infiltration_mm_h   DOUBLE PRECISION NOT NULL DEFAULT 5,
        roughness           DOUBLE PRECISION NOT NULL DEFAULT 0.035,
        wind_speed_kmh      DOUBLE PRECISION NOT NULL DEFAULT 20,
        river_rise_m        DOUBLE PRECISION NOT NULL DEFAULT 1.2,
        flow_model          TEXT         NOT NULL DEFAULT 'physics',
        elapsed_seconds     INT          NOT NULL DEFAULT 0,
        spread_area_ha      DOUBLE PRECISION NOT NULL DEFAULT 0,
        max_depth_m         DOUBLE PRECISION NOT NULL DEFAULT 0,
        total_volume_m3     DOUBLE PRECISION NOT NULL DEFAULT 0,
        buildings_exposed   INT          NOT NULL DEFAULT 0,
        buildings_critical  INT          NOT NULL DEFAULT 0,
        buildings_safe      INT          NOT NULL DEFAULT 0,
        center_lat          DOUBLE PRECISION,
        center_lng          DOUBLE PRECISION,
        polygon_coords      JSONB,
        buildings_json      JSONB,
        settings_history    JSONB,
        scenario            JSONB,
        status              TEXT         NOT NULL DEFAULT 'completed',
        method              TEXT         NOT NULL DEFAULT 'local-inertial-shallow-water',
        started_at          TIMESTAMPTZ  NOT NULL,
        ended_at            TIMESTAMPTZ  NOT NULL,
        created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_sim_runs_user    ON public.simulation_runs (user_id);
    CREATE INDEX IF NOT EXISTS idx_sim_runs_zone    ON public.simulation_runs (zone_name);
    CREATE INDEX IF NOT EXISTS idx_sim_runs_started ON public.simulation_runs (started_at DESC);

    CREATE TABLE IF NOT EXISTS public.flood_alerts_log (
        id              BIGSERIAL    PRIMARY KEY,
        simulation_id   UUID         REFERENCES public.simulation_runs(id) ON DELETE CASCADE,
        zone_name       TEXT,
        alert_level     TEXT         NOT NULL DEFAULT 'warning',
        max_depth_m     DOUBLE PRECISION,
        spread_area_ha  DOUBLE PRECISION,
        buildings_at_risk INT,
        triggered_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_flood_alerts_sim  ON public.flood_alerts_log (simulation_id);
    CREATE INDEX IF NOT EXISTS idx_flood_alerts_zone ON public.flood_alerts_log (zone_name);
    CREATE INDEX IF NOT EXISTS idx_flood_alerts_time ON public.flood_alerts_log (triggered_at DESC);

    CREATE TABLE IF NOT EXISTS public.building_flood_exposure (
        id            BIGSERIAL    PRIMARY KEY,
        simulation_id UUID         REFERENCES public.simulation_runs(id) ON DELETE CASCADE,
        building_id   TEXT,
        building_name TEXT,
        lat           DOUBLE PRECISION,
        lng           DOUBLE PRECISION,
        peak_depth_m  DOUBLE PRECISION,
        arrival_seconds INT,
        risk_level    TEXT,
        recorded_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_bfe_simulation ON public.building_flood_exposure (simulation_id);
    CREATE INDEX IF NOT EXISTS idx_bfe_risk       ON public.building_flood_exposure (risk_level);
    """
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        logger.info("simulation_pg_service: tables ensured in sensor_db")
    except Exception as exc:
        logger.warning("simulation_pg_service: ensure_tables failed (non-fatal): %s", exc)


def save_simulation_run(
    *,
    run_id: str,
    user_id: str,
    title: str,
    period: str,
    zone_name: str,
    started_at: datetime,
    ended_at: datetime,
    scenario: Dict[str, Any],
    buildings: List[Dict[str, Any]],
    settings_history: List[Dict[str, Any]],
    summary: Dict[str, Any],
    method: str = "local-inertial-shallow-water",
    center_lat: Optional[float] = None,
    center_lng: Optional[float] = None,
    polygon_coords: Optional[Any] = None,
) -> str:
    """
    Upsert one simulation run into sensor_db and auto-trigger
    a flood alert row when max depth >= 0.5 m.
    Returns the simulation UUID string.
    """
    sim_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sim:{user_id}:{run_id}"))
    sc = scenario or {}

    rainfall     = float(sc.get("rainfallMmH",    sc.get("rainfall_mm_h",      0)))
    intensity    = float(sc.get("floodIntensity",  sc.get("flood_intensity_pct",100)))
    duration     = int(sc.get("durationMinutes",   sc.get("duration_minutes",    60)))
    saturation   = float(sc.get("soilSaturation",  sc.get("soil_saturation_pct", 80)))
    infiltration = float(sc.get("infiltrationMmH", sc.get("infiltration_mm_h",    5)))
    roughness    = float(sc.get("roughness",         0.035))
    wind         = float(sc.get("windSpeedKmh",    sc.get("wind_speed_kmh",     20)))
    river_rise   = float(sc.get("riverRiseM",      sc.get("river_rise_m",       1.2)))
    flow_model   = str(sc.get("flowModel",         sc.get("flow_model",     "physics")))

    elapsed   = int(summary.get("elapsed_seconds",    0))
    spread    = float(summary.get("spread_area_ha",   0))
    max_depth = float(summary.get("max_depth_m",      0))
    volume    = float(summary.get("total_volume_m3",  0))
    exposed   = int(summary.get("buildings_exposed",  0))
    critical  = int(summary.get("buildings_critical", 0))
    safe_b    = int(summary.get("buildings_safe",     0))

    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.simulation_runs (
                    id, run_id, user_id, zone_name, title, period,
                    rainfall_mm_h, flood_intensity_pct, duration_minutes,
                    soil_saturation_pct, infiltration_mm_h, roughness,
                    wind_speed_kmh, river_rise_m, flow_model,
                    elapsed_seconds, spread_area_ha, max_depth_m,
                    total_volume_m3, buildings_exposed, buildings_critical, buildings_safe,
                    center_lat, center_lng, polygon_coords,
                    buildings_json, settings_history, scenario,
                    method, started_at, ended_at, created_at
                ) VALUES (
                    %s, %s::uuid, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb,
                    %s, %s, %s, NOW()
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    max_depth_m        = EXCLUDED.max_depth_m,
                    spread_area_ha     = EXCLUDED.spread_area_ha,
                    total_volume_m3    = EXCLUDED.total_volume_m3,
                    buildings_exposed  = EXCLUDED.buildings_exposed,
                    buildings_critical = EXCLUDED.buildings_critical,
                    buildings_safe     = EXCLUDED.buildings_safe,
                    elapsed_seconds    = EXCLUDED.elapsed_seconds,
                    ended_at           = EXCLUDED.ended_at
            """, (
                sim_id, run_id, user_id, zone_name, title, period,
                rainfall, intensity, duration,
                saturation, infiltration, roughness,
                wind, river_rise, flow_model,
                elapsed, spread, max_depth,
                volume, exposed, critical, safe_b,
                center_lat, center_lng,
                json.dumps(polygon_coords) if polygon_coords else None,
                json.dumps(buildings[:5000]),
                json.dumps(settings_history[:1000]),
                json.dumps(sc),
                method, started_at, ended_at,
            ))

            if max_depth >= 0.5:
                level = (
                    "critical" if max_depth >= 2.0
                    else "high"    if max_depth >= 1.0
                    else "warning"
                )
                cur.execute("""
                    INSERT INTO public.flood_alerts_log
                        (simulation_id, zone_name, alert_level,
                         max_depth_m, spread_area_ha, buildings_at_risk)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s)
                """, (sim_id, zone_name, level, max_depth, spread, exposed))

            if buildings:
                top = sorted(
                    buildings,
                    key=lambda b: float(b.get("peakDepthM", b.get("peak_depth_m", 0))),
                    reverse=True,
                )[:500]
                for b in top:
                    cur.execute("""
                        INSERT INTO public.building_flood_exposure
                            (simulation_id, building_id, building_name,
                             lat, lng, peak_depth_m, arrival_seconds, risk_level)
                        VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        sim_id,
                        str(b.get("id", "")),
                        str(b.get("name", b.get("buildingName", ""))),
                        b.get("lat"), b.get("lng"),
                        float(b.get("peakDepthM", b.get("peak_depth_m", 0))),
                        int(b.get("arrivalSeconds", b.get("arrival_seconds", -1))),
                        str(b.get("riskLevel", b.get("risk_level", "safe"))),
                    ))

        conn.commit()

    logger.info(
        "simulation_pg saved: id=%s zone=%s depth=%.2fm spread=%.1fha buildings=%d",
        sim_id, zone_name, max_depth, spread, exposed,
    )
    return sim_id


def get_simulation_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent simulation runs from sensor_db."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, run_id, user_id, zone_name, title, period,
                           rainfall_mm_h, flood_intensity_pct, duration_minutes,
                           elapsed_seconds, spread_area_ha, max_depth_m,
                           total_volume_m3, buildings_exposed, buildings_critical, buildings_safe,
                           flow_model, method, status, started_at, ended_at, created_at
                    FROM public.simulation_runs
                    ORDER BY started_at DESC
                    LIMIT %s
                """, (limit,))
                cols = [d[0] for d in cur.description]
                rows = []
                for r in cur.fetchall():
                    item = dict(zip(cols, r))
                    for k in ("started_at", "ended_at", "created_at"):
                        v = item.get(k)
                        if v and hasattr(v, "isoformat"):
                            item[k] = v.isoformat()
                    rows.append(item)
                return rows
    except Exception as exc:
        logger.error("get_simulation_runs: %s", exc)
        return []


def get_flood_alerts(limit: int = 100) -> List[Dict[str, Any]]:
    """Return recent auto-triggered flood alerts from sensor_db."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, simulation_id, zone_name, alert_level,
                           max_depth_m, spread_area_ha, buildings_at_risk, triggered_at
                    FROM public.flood_alerts_log
                    ORDER BY triggered_at DESC
                    LIMIT %s
                """, (limit,))
                cols = [d[0] for d in cur.description]
                rows = []
                for r in cur.fetchall():
                    item = dict(zip(cols, r))
                    v = item.get("triggered_at")
                    if v and hasattr(v, "isoformat"):
                        item["triggered_at"] = v.isoformat()
                    rows.append(item)
                return rows
    except Exception as exc:
        logger.error("get_flood_alerts: %s", exc)
        return []
