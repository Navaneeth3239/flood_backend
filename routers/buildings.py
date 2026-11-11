"""
Buildings API Router: Real Microsoft Global ML Building Footprints
Fetches and serves building footprint polygons extruded in 3D.
"""

from typing import Optional, List
from fastapi import APIRouter, Query, Body, HTTPException
from pydantic import BaseModel, Field

from services.ms_building_service import ms_building_service

router = APIRouter(prefix="/buildings", tags=["buildings"])


class BuildingBboxRequest(BaseModel):
    minLat: Optional[float] = None
    minLon: Optional[float] = None
    maxLat: Optional[float] = None
    maxLon: Optional[float] = None
    north: Optional[float] = None
    south: Optional[float] = None
    east: Optional[float] = None
    west: Optional[float] = None
    polygon: Optional[List[List[float]]] = None
    water_level_m: Optional[float] = 0.0
    max_buildings: Optional[int] = 2500


@router.get("")
@router.get("/")
async def get_buildings(
    minLat: Optional[float] = Query(None, description="Minimum latitude (south)"),
    minLon: Optional[float] = Query(None, description="Minimum longitude (west)"),
    maxLat: Optional[float] = Query(None, description="Maximum latitude (north)"),
    maxLon: Optional[float] = Query(None, description="Maximum longitude (east)"),
    north: Optional[float] = Query(None),
    south: Optional[float] = Query(None),
    east: Optional[float] = Query(None),
    west: Optional[float] = Query(None),
    lat: Optional[float] = Query(None),
    lng: Optional[float] = Query(None),
    radius_km: Optional[float] = Query(1.5),
    water_level_m: float = Query(0.0, description="Water level elevation for flood simulation"),
    max_buildings: int = Query(2500, description="Maximum buildings to return"),
    polygon: Optional[str] = Query(None, description="JSON string of polygon points [[lat, lon], ...]"),
):
    """
    Load real Microsoft Global ML Building Footprint polygons inside bounding box or polygon.
    Returns GeoJSON FeatureCollection with 3D height, elevation, river distance, and flood risk.
    """
    import json
    poly_list = None
    if polygon:
        try:
            poly_list = json.loads(polygon)
            if not (isinstance(poly_list, list) and len(poly_list) >= 3):
                poly_list = None
        except Exception:
            poly_list = None

    n = north if north is not None else maxLat
    s = south if south is not None else minLat
    e = east if east is not None else maxLon
    w = west if west is not None else minLon

    if None in (n, s, e, w):
        if poly_list:
            lats = [p[0] for p in poly_list]
            lngs = [p[1] for p in poly_list]
            n, s = max(lats), min(lats)
            e, w = max(lngs), min(lngs)
        elif lat is not None and lng is not None:
            r = radius_km or 1.5
            d_lat = r / 111.0
            d_lng = r / (111.0 * 0.98)
            n, s = lat + d_lat, lat - d_lat
            e, w = lng + d_lng, lng - d_lng
        else:
            raise HTTPException(
                status_code=400,
                detail="Must supply minLat/minLon/maxLat/maxLon or polygon or lat/lng/radius_km",
            )

    return await ms_building_service.get_buildings_for_bbox(
        min_lat=float(s),
        min_lon=float(w),
        max_lat=float(n),
        max_lon=float(e),
        polygon=poly_list,
        max_buildings=max_buildings,
        water_level_m=water_level_m,
    )



@router.post("")
@router.post("/")
async def extract_buildings_post(payload: BuildingBboxRequest = Body(...)):
    """POST endpoint supporting custom polygon filtering for Microsoft building footprints."""
    n = payload.north if payload.north is not None else payload.maxLat
    s = payload.south if payload.south is not None else payload.minLat
    e = payload.east if payload.east is not None else payload.maxLon
    w = payload.west if payload.west is not None else payload.minLon

    if None in (n, s, e, w):
        if payload.polygon and len(payload.polygon) >= 3:
            lats = [p[0] for p in payload.polygon]
            lngs = [p[1] for p in payload.polygon]
            n, s = max(lats), min(lats)
            e, w = max(lngs), min(lngs)
        else:
            raise HTTPException(status_code=400, detail="Bounding box or polygon required")

    return await ms_building_service.get_buildings_for_bbox(
        min_lat=float(s),
        min_lon=float(w),
        max_lat=float(n),
        max_lon=float(e),
        polygon=payload.polygon,
        max_buildings=payload.max_buildings or 2500,
        water_level_m=payload.water_level_m or 0.0,
    )
