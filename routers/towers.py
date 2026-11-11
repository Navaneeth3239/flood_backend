"""
Mobile Network Towers API Router: Shared tower information, OpenStreetMap extraction,
GeoJSON import, and flood-risk integration.
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from services.tower_service import tower_service, VALID_OPERATORS, VALID_RISK_LEVELS

router = APIRouter(prefix="/towers", tags=["mobile-towers"])


class TowerPayload(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    lat: float = Field(..., description="Latitude coordinate")
    lng: float = Field(..., description="Longitude coordinate")
    height: Optional[float] = Field(None, description="Tower height in meters")
    operator: Optional[str] = Field("Unknown", description="Network operator (Jio, Airtel, Vi, BSNL, or Unknown)")
    source: Optional[str] = Field("manual", description="Data source (osm, geojson_import, manual)")
    flood_risk_level: Optional[str] = Field(None, description="Flood risk level (Low, Moderate, High, Critical)")
    properties: Optional[Dict[str, Any]] = None


class RecalculateRiskPayload(BaseModel):
    water_level_m: Optional[float] = None


@router.get("", response_model=List[Dict[str, Any]])
@router.get("/", response_model=List[Dict[str, Any]])
async def list_towers(
    minLat: Optional[float] = Query(None, description="Minimum latitude (south)"),
    minLon: Optional[float] = Query(None, description="Minimum longitude (west)"),
    maxLat: Optional[float] = Query(None, description="Maximum latitude (north)"),
    maxLon: Optional[float] = Query(None, description="Maximum longitude (east)"),
    north: Optional[float] = Query(None, description="North latitude alias"),
    south: Optional[float] = Query(None, description="South latitude alias"),
    east: Optional[float] = Query(None, description="East longitude alias"),
    west: Optional[float] = Query(None, description="West longitude alias"),
    lat: Optional[float] = Query(None, description="Center latitude for radius query"),
    lng: Optional[float] = Query(None, description="Center longitude for radius query"),
    radius_km: Optional[float] = Query(None, description="Search radius in kilometers"),
    operator: Optional[str] = Query(None, description="Filter by network operator (Jio, Airtel, Vi, BSNL, Unknown, or All)"),
    flood_risk: Optional[str] = Query(None, description="Filter by flood risk level (Low, Moderate, High, Critical, or All)"),
    source: Optional[str] = Query(None, description="Filter by data source (osm, geojson_import, manual)"),
    limit: int = Query(2000, description="Maximum number of towers to return"),
):
    """
    Retrieves mobile network towers matching spatial bounding box or radius,
    with network operator and flood risk filters.
    Used identically by both Leaflet 2D Map and Cesium 3D Digital Twin.
    """
    effective_min_lat = minLat if minLat is not None else south
    effective_max_lat = maxLat if maxLat is not None else north
    effective_min_lon = minLon if minLon is not None else west
    effective_max_lon = maxLon if maxLon is not None else east

    try:
        return await tower_service.query_towers(
            min_lat=effective_min_lat,
            min_lon=effective_min_lon,
            max_lat=effective_max_lat,
            max_lon=effective_max_lon,
            center_lat=lat,
            center_lng=lng,
            radius_km=radius_km,
            operator=operator,
            flood_risk=flood_risk,
            source=source,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query mobile towers: {e}")


@router.get("/stats")
async def get_tower_stats():
    """Returns aggregated breakdown by network operator and flood risk."""
    try:
        return await tower_service.get_tower_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tower stats: {e}")


@router.get("/{tower_id}")
async def get_tower(tower_id: str):
    """Retrieves full details of a specific mobile network tower by ID."""
    tower = await tower_service.get_tower_by_id(tower_id)
    if not tower:
        raise HTTPException(status_code=404, detail=f"Tower '{tower_id}' not found")
    return tower


@router.post("", status_code=201)
@router.post("/", status_code=201)
async def create_or_update_tower(payload: TowerPayload = Body(...)):
    """Creates or updates a single mobile tower record."""
    try:
        return await tower_service.upsert_tower(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save tower: {e}")


@router.post("/import-geojson")
async def import_geojson_towers(geojson: Dict[str, Any] = Body(...)):
    """
    Imports mobile communication towers from a GeoJSON FeatureCollection or Feature.
    Normalizes coordinates and operator without inventing data.
    """
    try:
        return await tower_service.import_geojson(geojson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GeoJSON import error: {e}")


@router.post("/recalculate-risk")
async def recalculate_flood_risk(payload: RecalculateRiskPayload = Body(default=RecalculateRiskPayload())):
    """
    Re-evaluates flood risk level for all towers based on current prediction telemetry,
    monitored zone status, or simulation water level.
    """
    try:
        return await tower_service.recalculate_flood_risks(water_level_m=payload.water_level_m)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recalculate tower risk: {e}")

