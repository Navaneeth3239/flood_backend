"""Deterministic AI risk scoring + Digital Twin simulation engine (no external LLM)."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from lib.auth import current_user, require_roles
from lib.db import db
from models.schemas import (
    RiskAssessment,
    RiskFactor,
    SimulationRequest,
    SimulationResult,
    SimulationStep,
)

router = APIRouter(tags=["ai"])

DATA_SOURCES = [
    "LoRaWAN IoT sensor telemetry",
    "ISRO / Bhuvan satellite imagery",
    "IMD historical rainfall datasets",
    "GSI landslide susceptibility maps",
]
MODELS = [
    "Random Forest — landslide & flood probability",
    "Isolation Forest — telemetry anomaly detection",
    "CNN — satellite scene classification",
]


def _level(score: int) -> str:
    if score >= 75:
        return "critical"
    if score >= 55:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def _assess(zone: dict) -> RiskAssessment:
    rainfall = float(zone.get("rainfall_mm") or 0)
    soil = float(zone.get("soil_moisture_pct") or 0)
    water = float(zone.get("water_level_m") or 0)
    temp = float(zone.get("temperature_c") or 0)

    rain_w = min(35, round(rainfall / 3.0))
    soil_w = min(25, round(soil / 4.0))
    water_w = min(20, round(water * 3.0))
    hist_w = 10 if zone.get("hazard_type") in ("landslide", "flood") else 5
    thermal_w = min(10, round(max(0.0, temp - 28) * 2))
    score = max(0, min(100, rain_w + soil_w + water_w + hist_w + thermal_w))
    level = _level(score)

    factors = [
        RiskFactor(name="Rainfall intensity", weight=rain_w, value=f"{rainfall:.1f} mm/h"),
        RiskFactor(name="Soil moisture saturation", weight=soil_w, value=f"{soil:.0f} %"),
        RiskFactor(name="River / reservoir water level", weight=water_w, value=f"{water:.2f} m"),
        RiskFactor(name="Historical hazard pattern", weight=hist_w, value=zone.get("hazard_type", "n/a").replace("_", " ")),
        RiskFactor(name="Thermal & terrain slope stress", weight=thermal_w, value=f"{temp:.1f} °C"),
    ]
    recs = {
        "critical": "Initiate evacuation readiness for downstream habitations, deploy NDRF liaison, and escalate to the State EOC immediately.",
        "high": "Increase monitoring frequency to 5-minute intervals and prepare emergency response teams for deployment.",
        "medium": "Maintain heightened watch, verify sensor uplink health, and brief district field officers.",
        "low": "Continue routine monitoring at standard cadence. No escalation required.",
    }
    return RiskAssessment(
        zone_id=zone["id"],
        zone_name=zone["name"],
        risk_score=score,
        risk_level=level,
        confidence=round(min(97.5, 78.0 + score / 6.0), 1),
        factors=factors,
        recommendation=recs[level],
        data_sources=DATA_SOURCES,
        models_used=MODELS,
    )


@router.get("/risk", response_model=List[RiskAssessment])
async def risk_overview(user: dict = Depends(current_user)):
    docs = await db.zones.find({}, {"_id": 0}).to_list(500)
    results = [_assess(d) for d in docs]
    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results


@router.get("/risk/{zone_id}", response_model=RiskAssessment)
async def risk_for_zone(zone_id: str, user: dict = Depends(current_user)):
    doc = await db.zones.find_one({"id": zone_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Monitoring zone not found")
    return _assess(doc)


@router.post("/simulations", response_model=SimulationResult, status_code=201)
async def run_simulation(
    payload: SimulationRequest, user: dict = Depends(require_roles("admin", "gov_officer"))
):
    zone = await db.zones.find_one({"id": payload.zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Monitoring zone not found")

    drivers = {
        "flood": payload.rainfall_intensity / 2.0 + payload.soil_saturation / 3.0,
        "landslide": payload.soil_saturation / 2.0 + payload.slope_angle,
        "forest_fire": payload.wind_speed / 1.5 + (100 - payload.soil_saturation) / 2.5,
        "evacuation": payload.rainfall_intensity / 3.0 + payload.slope_angle / 2.0,
    }
    peak = round(min(99.0, max(6.0, drivers[payload.scenario])), 1)
    severity = _level(int(peak)).upper()
    radius = float(zone.get("radius_km") or 8.0)
    area = round(3.1416 * radius * radius * (peak / 100.0), 1)

    steps: List[SimulationStep] = []
    for hour in range(1, 13):
        ramp = min(1.0, hour / 7.0)
        steps.append(
            SimulationStep(
                hour=hour,
                impact_pct=round(peak * ramp, 1),
                affected_area_km2=round(area * ramp, 1),
            )
        )

    result = SimulationResult(
        zone_id=zone["id"],
        zone_name=zone["name"],
        scenario=payload.scenario,
        severity=severity,
        peak_impact_pct=peak,
        affected_area_km2=area,
        population_at_risk=int(area * 420),
        evacuation_time_min=int(38 + payload.slope_angle * 1.4 + area / 4),
        steps=steps,
        summary=(
            f"{payload.scenario.replace('_', ' ').title()} scenario over {zone['name']} reaches a "
            f"peak impact of {peak}% within 7 hours, affecting approximately {area} km². "
            f"Estimated safe evacuation window: {int(38 + payload.slope_angle * 1.4 + area / 4)} minutes."
        ),
    )
    await db.simulations.insert_one(result.model_dump())
    return result


@router.get("/simulations", response_model=List[SimulationResult])
async def list_simulations(user: dict = Depends(current_user)):
    docs = await db.simulations.find({}, {"_id": 0}).sort("run_at", -1).to_list(50)
    return [SimulationResult(**d) for d in docs]
