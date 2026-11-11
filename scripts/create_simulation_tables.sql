-- =====================================================================
--  Flash Flood Simulation PostgreSQL Schema
--  Run against: sensor_db @ db.nishanth.qzz.io:5432
--
--  Execute with:
--    psql postgresql://sensor_user:nexgi@db.nishanth.qzz.io:5432/sensor_db \
--         -f backend/scripts/create_simulation_tables.sql
-- =====================================================================

-- 1. Simulation Runs — one row per completed simulation
CREATE TABLE IF NOT EXISTS public.simulation_runs (
    id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID         NOT NULL UNIQUE,
    user_id             TEXT         NOT NULL,
    zone_name           TEXT         NOT NULL DEFAULT '',
    title               TEXT         NOT NULL,
    period              TEXT         NOT NULL DEFAULT '',

    -- Storm parameters (from UI sliders)
    rainfall_mm_h       DOUBLE PRECISION NOT NULL DEFAULT 0,
    flood_intensity_pct DOUBLE PRECISION NOT NULL DEFAULT 100,
    duration_minutes    INT          NOT NULL DEFAULT 60,
    soil_saturation_pct DOUBLE PRECISION NOT NULL DEFAULT 80,
    infiltration_mm_h   DOUBLE PRECISION NOT NULL DEFAULT 5,
    roughness           DOUBLE PRECISION NOT NULL DEFAULT 0.035,
    wind_speed_kmh      DOUBLE PRECISION NOT NULL DEFAULT 20,
    river_rise_m        DOUBLE PRECISION NOT NULL DEFAULT 1.2,
    flow_model          TEXT         NOT NULL DEFAULT 'physics',

    -- Simulation results
    elapsed_seconds     INT          NOT NULL DEFAULT 0,
    spread_area_ha      DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_depth_m         DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_volume_m3     DOUBLE PRECISION NOT NULL DEFAULT 0,
    buildings_exposed   INT          NOT NULL DEFAULT 0,
    buildings_critical  INT          NOT NULL DEFAULT 0,
    buildings_safe      INT          NOT NULL DEFAULT 0,

    -- Location
    center_lat          DOUBLE PRECISION,
    center_lng          DOUBLE PRECISION,
    polygon_coords      JSONB,

    -- Full data blobs
    buildings_json      JSONB,          -- per-building array (capped at 5000)
    settings_history    JSONB,          -- parameter changes during run
    scenario            JSONB,          -- full scenario snapshot

    -- Metadata
    status              TEXT         NOT NULL DEFAULT 'completed',
    method              TEXT         NOT NULL DEFAULT 'local-inertial-shallow-water',
    started_at          TIMESTAMPTZ  NOT NULL,
    ended_at            TIMESTAMPTZ  NOT NULL,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sim_runs_user    ON public.simulation_runs (user_id);
CREATE INDEX IF NOT EXISTS idx_sim_runs_zone    ON public.simulation_runs (zone_name);
CREATE INDEX IF NOT EXISTS idx_sim_runs_started ON public.simulation_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sim_runs_depth   ON public.simulation_runs (max_depth_m DESC);

-- 2. Flood Alerts Log — auto-inserted when max_depth >= 0.5 m
CREATE TABLE IF NOT EXISTS public.flood_alerts_log (
    id              BIGSERIAL    PRIMARY KEY,
    simulation_id   UUID         REFERENCES public.simulation_runs(id) ON DELETE CASCADE,
    zone_name       TEXT,
    alert_level     TEXT         NOT NULL DEFAULT 'warning',  -- warning | high | critical
    max_depth_m     DOUBLE PRECISION,
    spread_area_ha  DOUBLE PRECISION,
    buildings_at_risk INT,
    triggered_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_flood_alerts_sim  ON public.flood_alerts_log (simulation_id);
CREATE INDEX IF NOT EXISTS idx_flood_alerts_zone ON public.flood_alerts_log (zone_name);
CREATE INDEX IF NOT EXISTS idx_flood_alerts_time ON public.flood_alerts_log (triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_flood_alerts_lvl  ON public.flood_alerts_log (alert_level);

-- 3. Building Flood Exposure — per-building rows (top 500 per run)
CREATE TABLE IF NOT EXISTS public.building_flood_exposure (
    id              BIGSERIAL    PRIMARY KEY,
    simulation_id   UUID         REFERENCES public.simulation_runs(id) ON DELETE CASCADE,
    building_id     TEXT,
    building_name   TEXT,
    lat             DOUBLE PRECISION,
    lng             DOUBLE PRECISION,
    peak_depth_m    DOUBLE PRECISION,
    arrival_seconds INT,
    risk_level      TEXT,   -- safe | low | moderate | high | critical
    recorded_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bfe_simulation ON public.building_flood_exposure (simulation_id);
CREATE INDEX IF NOT EXISTS idx_bfe_risk       ON public.building_flood_exposure (risk_level);
CREATE INDEX IF NOT EXISTS idx_bfe_depth      ON public.building_flood_exposure (peak_depth_m DESC);

-- Useful verification query
-- SELECT tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
-- FROM pg_tables WHERE schemaname = 'public'
-- ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
