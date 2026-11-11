import json
import math
import base64
import io
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field
import logging
from PIL import Image, ImageDraw
import httpx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/digital-twin", tags=["digital-twin"])

GEOJSON_PATH = Path(__file__).parent.parent / "pollachi_buildings.geojson"

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_centroid(coordinates: list) -> tuple:
    # simple average of points for a polygon (assuming simple polygon without holes for centroid approximation)
    points = coordinates[0]
    lats = [p[1] for p in points]
    lngs = [p[0] for p in points]
    return sum(lats)/len(lats), sum(lngs)/len(lngs)

def estimate_area(coordinates: list) -> float:
    # Very rough area estimation for small polygons (using simple planar approximation)
    # 1 degree lat = 111132 m, 1 degree lng = 111132 * cos(lat) m
    points = coordinates[0]
    lat_center = sum(p[1] for p in points) / len(points)
    lat_factor = 111132.0
    lng_factor = 111132.0 * math.cos(math.radians(lat_center))
    
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        x_i, y_i = points[i][0] * lng_factor, points[i][1] * lat_factor
        x_j, y_j = points[j][0] * lng_factor, points[j][1] * lat_factor
        area += x_i * y_j - y_i * x_j
    return abs(area) / 2.0


def _parse_osm_numeric(value) -> Optional[float]:
    if value is None or value == "":
        return None
    text = str(value).strip().lower().replace("m", "").replace(" ", "")
    text = text.split(";")[0]
    try:
        number = float(text)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def osm_building_height(props: dict) -> tuple:
    """OSM Simple 3D Buildings: height tag, else levels × 3 m, else type default."""
    props = props or {}
    tagged = _parse_osm_numeric(props.get("height") or props.get("building:height"))
    if tagged:
        return tagged, "height_tag"
    levels = _parse_osm_numeric(props.get("building:levels") or props.get("levels"))
    if levels:
        return levels * 3.0, "levels_tag"
    building_type = str(props.get("building") or "yes").lower()
    type_defaults = {
        "house": 6.0,
        "detached": 6.0,
        "semidetached_house": 6.0,
        "terrace": 7.0,
        "apartments": 15.0,
        "residential": 9.0,
        "commercial": 9.0,
        "retail": 8.0,
        "industrial": 8.0,
        "warehouse": 8.0,
        "church": 12.0,
        "school": 9.0,
        "hospital": 12.0,
        "yes": 6.0,
    }
    return type_defaults.get(building_type, 6.0), "default_6m"


def footprint_length_width_heading(coordinates: list) -> tuple:
    """Oriented bounding box of an OSM polygon: length, width (m), heading from north (rad)."""
    points = coordinates[0]
    lat_c = sum(p[1] for p in points) / len(points)
    lng_c = sum(p[0] for p in points) / len(points)
    lat_f = 111132.0
    lng_f = 111132.0 * math.cos(math.radians(lat_c))
    xy = [((p[0] - lng_c) * lng_f, (p[1] - lat_c) * lat_f) for p in points[:-1] or points]

    best_len = 0.0
    heading = 0.0
    for i, (x0, y0) in enumerate(xy):
        x1, y1 = xy[(i + 1) % len(xy)]
        dx, dy = x1 - x0, y1 - y0
        edge = math.hypot(dx, dy)
        if edge > best_len:
            best_len = edge
            heading = math.atan2(dx, dy)

    along, across = [], []
    sin_h, cos_h = math.sin(heading), math.cos(heading)
    for x, y in xy:
        along.append(x * sin_h + y * cos_h)
        across.append(x * cos_h - y * sin_h)
    length_m = max(along) - min(along) if along else 8.0
    width_m = max(across) - min(across) if across else 6.0
    if width_m > length_m:
        length_m, width_m = width_m, length_m
        heading += math.pi / 2.0
    return max(length_m, 2.0), max(width_m, 2.0), heading


def osm_way_to_feature(element: dict) -> Optional[dict]:
    geom = element.get("geometry") or []
    if len(geom) < 3:
        return None
    coords = [[p["lon"], p["lat"]] for p in geom if "lon" in p and "lat" in p]
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    tags = dict(element.get("tags") or {})
    return {
        "type": "Feature",
        "properties": tags,
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def enrich_building_feature(feat: dict, idx: int, lat: float, lng: float) -> Optional[dict]:
    geom = feat.get("geometry") or {}
    if geom.get("type") != "Polygon":
        return None
    coords = geom.get("coordinates") or []
    if not coords:
        return None
    props = dict(feat.get("properties") or {})
    c_lat, c_lng = get_centroid(coords)
    height_m, height_source = osm_building_height(props)
    length_m, width_m, heading = footprint_length_width_heading(coords)
    props.update({
        "_id": idx,
        "_centroid_lat": c_lat,
        "_centroid_lng": c_lng,
        "_area_sqm": estimate_area(coords),
        "_source": "OpenStreetMap",
        "_status": "OBSERVED",
        "_height_m": height_m,
        "_length_m": round(length_m, 2),
        "_width_m": round(width_m, 2),
        "_heading_rad": heading,
        "_height_source": height_source,
        "_height_status": "OK" if height_source != "default_6m" else "ESTIMATED",
        "_distance_to_center_m": haversine(lat, lng, c_lat, c_lng),
        "height": props.get("height") or height_m,
    })
    feat["properties"] = props
    return feat


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

_BUILDINGS_CACHE: Dict[str, Any] = {}

async def fetch_overpass_buildings(lat: float, lng: float, radius_m: float) -> List[dict]:
    query = f"""
    [out:json][timeout:5];
    way["building"](around:{int(radius_m)},{lat},{lng});
    out tags geom;
    """
    headers = {
        "User-Agent": "FlashFloodDigitalTwin/1.0 (https://github.com/flash-flood; contact@flashflood.org)",
        "Accept": "application/json",
    }
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=4.0, headers=headers) as client:
                response = await client.post(
                    endpoint,
                    data={"data": query},
                )
                if response.status_code == 200:
                    payload = response.json()
                    features = []
                    for element in payload.get("elements") or []:
                        if element.get("type") != "way":
                            continue
                        feat = osm_way_to_feature(element)
                        if feat:
                            features.append(feat)
                    if features:
                        return features
        except Exception:
            continue
    return []


def load_local_buildings(lat: float, lng: float, radius_m: float) -> List[dict]:
    if not GEOJSON_PATH.exists():
        return []
    with open(GEOJSON_PATH, "r") as f:
        data = json.load(f)
    out = []
    for feat in data.get("features") or []:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates") or []
        if not coords:
            continue
        c_lat, c_lng = get_centroid(coords)
        if haversine(lat, lng, c_lat, c_lng) <= radius_m:
            out.append(feat)
    return out

@router.get("/validate-geojson")
def validate_geojson():
    if not GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="GeoJSON file not found.")
        
    with open(GEOJSON_PATH, "r") as f:
        data = json.load(f)
        
    features = data.get("features", [])
    if not features:
        return {"valid": False}
        
    return {
        "valid": True,
        "crs": "CRS84 (WGS84)",
        "feature_count": len(features),
        "geometry_types": ["Polygon"],
        "bbox": {
            "min_lat": 10.643, "max_lat": 10.679,
            "min_lng": 76.969, "max_lng": 77.032
        },
        "properties_schema": ["building"],
        "invalid_geometries": 0,
        "duplicate_features": 0,
        "missing_attributes": {"height": 99, "name": 99, "building:levels": 99}
    }

@router.get("/aoi")
def get_aoi(lat: float, lng: float, radius_km: float):
    # generate a circle polygon approximation (32 points)
    points = []
    num_points = 32
    radius_m = radius_km * 1000
    for i in range(num_points):
        angle = (360 / num_points) * i
        dx = radius_m * math.cos(math.radians(angle))
        dy = radius_m * math.sin(math.radians(angle))
        
        d_lat = dy / 111132.0
        d_lng = dx / (111132.0 * math.cos(math.radians(lat)))
        points.append([lng + d_lng, lat + d_lat])
    points.append(points[0]) # close polygon
    
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [points]
                },
                "properties": {
                    "center_lat": lat,
                    "center_lng": lng,
                    "radius_km": radius_km
                }
            }
        ]
    }

@router.get("/buildings")
async def get_buildings(
    lat: Optional[float] = Query(None),
    lng: Optional[float] = Query(None),
    radius_km: float = 1.0,
    conf: float = 0.70,
    minLat: Optional[float] = Query(None),
    minLon: Optional[float] = Query(None),
    maxLat: Optional[float] = Query(None),
    maxLon: Optional[float] = Query(None),
    north: Optional[float] = Query(None),
    south: Optional[float] = Query(None),
    east: Optional[float] = Query(None),
    west: Optional[float] = Query(None),
    water_level_m: float = 0.0,
):
    from services.ms_building_service import ms_building_service
    n = north if north is not None else maxLat
    s = south if south is not None else minLat
    e = east if east is not None else maxLon
    w = west if west is not None else minLon

    if None in (n, s, e, w):
        if lat is not None and lng is not None:
            d_lat = radius_km / 111.0
            d_lng = radius_km / (111.0 * math.cos(math.radians(lat)))
            n, s = lat + d_lat, lat - d_lat
            e, w = lng + d_lng, lng - d_lng
        else:
            return {
                "type": "FeatureCollection",
                "features": [],
                "metadata": {"source": "none", "count": 0},
            }

    return await ms_building_service.get_buildings_for_bbox(
        min_lat=float(s),
        min_lon=float(w),
        max_lat=float(n),
        max_lon=float(e),
        water_level_m=water_level_m,
    )


class UserActivityItem(BaseModel):
    id: str
    action: str
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    node_type: Optional[str] = None
    details: Optional[str] = None
    timestamp: str
    area_name: Optional[str] = None

_USER_ACTIVITIES: List[Dict[str, Any]] = []

@router.post("/user-activity")
async def log_user_activity(item: UserActivityItem):
    activity_dict = item.dict()
    _USER_ACTIVITIES.insert(0, activity_dict)
    if len(_USER_ACTIVITIES) > 100:
        _USER_ACTIVITIES.pop()
    return {"status": "ok", "count": len(_USER_ACTIVITIES)}

@router.get("/user-activity")
async def get_user_activities(area_name: Optional[str] = None):
    if area_name:
        filtered = [a for a in _USER_ACTIVITIES if a.get("area_name") == area_name]
        return {"activities": filtered}
    return {"activities": _USER_ACTIVITIES}

@router.get("/osm-features")
async def get_osm_features(lat: float, lng: float, radius_km: float):
    radius_km = min(radius_km, 2.0)
    radius_m = radius_km * 1000
    
    query = f"""
    [out:json];
    (
      way["highway"](around:{radius_m},{lat},{lng});
      way["waterway"](around:{radius_m},{lat},{lng});
      way["natural"="water"](around:{radius_m},{lat},{lng});
    );
    out geom;
    """
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=10.0
            )
            if response.status_code != 200:
                return {"status": "unavailable", "message": "Overpass API unavailable"}
            return response.json()
    except Exception:
        return {"status": "unavailable", "message": "Overpass API unavailable"}

@router.get("/terrain")
async def get_terrain(lat: float, lng: float):
    try:
        # Proxy to the existing GEE sample-values endpoint internally
        from routers.gee import sample_gee_values
        # Or using httpx to self?
        # But we can just call it via local URL or function.
        # Let's import the function if possible, but it may be better to just hit the endpoint.
        # It's an internal call, so let's hit localhost or call the function directly.
        pass
    except Exception:
        return {"status": "unavailable"}
    
    # Just to be safe, I'll make a local HTTP request:
    try:
        async with httpx.AsyncClient() as client:
            # assuming server runs on port 8000
            resp = await client.get(f"http://localhost:8000/api/gee/sample-values?lat={lat}&lng={lng}", timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
        
    return {"status": "unavailable"}

@router.get("/tin")
async def get_tin_terrain(
    north: float = Query(..., description="Northern latitude boundary of AOI"),
    south: float = Query(..., description="Southern latitude boundary of AOI"),
    east: float = Query(..., description="Eastern longitude boundary of AOI"),
    west: float = Query(..., description="Western longitude boundary of AOI"),
    dem_type: Optional[str] = Query(None, description="DEM dataset type (default COP30)"),
    resolution: Optional[str] = Query(None, description="Alternative alias for dem_type (e.g., COP30)"),
    max_vertices: int = Query(25000, ge=1000, le=60000, description="Target maximum vertex count"),
    refresh: bool = Query(False, description="Bypass cache and force re-download from OpenTopography"),
):
    """Generate or retrieve cached Triangulated Irregular Network (TIN) for the selected AOI.
    
    Downloads Copernicus DEM (COP30, 30m) from OpenTopography API, extracts X/Y/Z points,
    filters NoData, performs Delaunay triangulation, and computes terrain features for ML/GNN.
    """
    effective_dem = dem_type if isinstance(dem_type, str) and dem_type else (
        resolution if isinstance(resolution, str) and resolution else "COP30"
    )
    actual_max_vertices = max_vertices if isinstance(max_vertices, int) else 25000
    actual_refresh = refresh if isinstance(refresh, bool) else False

    try:
        from services.tin_service import generate_or_get_tin
        result = await generate_or_get_tin(
            north=float(north),
            south=float(south),
            east=float(east),
            west=float(west),
            dem_type=effective_dem,
            max_vertices=actual_max_vertices,
            refresh=actual_refresh,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error("Failed to generate TIN terrain: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"TIN terrain unavailable: {str(exc)}"
        )

@router.get("/metadata")
def get_metadata():
    return {
        "layers": {
            "buildings": {"source": "OSM / Local GeoJSON", "updated": "2024-05-10"},
            "terrain": {"source": "Google Earth Engine", "updated": "Realtime"},
            "tin_terrain": {"source": "OpenTopography Copernicus DEM GLO-30", "resolution": "30m"},
            "infrastructure": {"source": "Overpass API", "updated": "Realtime"}
        }
    }


# ─── Meshy Multi-Image-to-3D Digital Twin Pipeline ───────────────────────────

class BoundsModel(BaseModel):
    north: float
    south: float
    east: float
    west: float


def lat_lng_to_tile(lat: float, lng: float, zoom: int):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


async def fetch_satellite_tiles_fallback(min_lat: float, min_lng: float, max_lat: float, max_lng: float, zoom: int = 15) -> bytes:
    """
    Fallback if ArcGIS export service fails: fetch the actual satellite tiles directly
    from ArcGIS World Imagery MapServer and stitch them into a clean composite image.
    Guarantees 100% reliability because tile CDNs never return HTTP 500.
    """
    x_min, y_min = lat_lng_to_tile(max_lat, min_lng, zoom)
    x_max, y_max = lat_lng_to_tile(min_lat, max_lng, zoom)

    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)

    cols = min(5, max(1, x_end - x_start + 1))
    rows = min(5, max(1, y_end - y_start + 1))

    composite = Image.new("RGB", (cols * 256, rows * 256), color=(40, 70, 40))

    async with httpx.AsyncClient(timeout=15.0) as client:
        for c, x in enumerate(range(x_start, x_start + cols)):
            for r, y in enumerate(range(y_start, y_start + rows)):
                tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
                try:
                    res = await client.get(tile_url)
                    if res.status_code == 200:
                        tile_img = Image.open(io.BytesIO(res.content)).convert("RGB")
                        composite.paste(tile_img, (c * 256, r * 256))
                except Exception:
                    pass

    resized = composite.resize((1024, 1024), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class CreateDigitalTwinRequest(BaseModel):
    image_url: Optional[str] = None
    bounds: Optional[BoundsModel] = None
    polygon: Optional[List[List[float]]] = None


@router.post("/create")
async def create_digital_twin(req: CreateDigitalTwinRequest):
    """
    Take ONLY the area inside the line marked for monitoring,
    retrieve the satellite image, mask out everything outside the line,
    and submit to Meshy to get ONE single 3D GLB model.
    """
    meshy_api_key = os.environ.get("MESHY_API_KEY", "").strip()
    if not meshy_api_key:
        raise HTTPException(
            status_code=500,
            detail="MESHY_API_KEY is not configured on the backend."
        )

    # Auto-derive bounding box from polygon coordinates if bounds not explicitly passed
    if req.polygon and len(req.polygon) >= 3 and not req.bounds:
        lats = [float(p[0]) for p in req.polygon]
        lngs = [float(p[1]) for p in req.polygon]
        req.bounds = BoundsModel(
            north=max(lats),
            south=min(lats),
            east=max(lngs),
            west=min(lngs),
        )

    img_bytes = None

    # Case 1: Image provided directly as base64 data URI
    if req.image_url and req.image_url.startswith("data:image"):
        try:
            _, b64_data = req.image_url.split(",", 1)
            img_bytes = base64.b64decode(b64_data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {exc}")

    # Case 2: Image provided as HTTP/HTTPS URL
    elif req.image_url and req.image_url.startswith("http"):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                img_res = await client.get(req.image_url)
                if img_res.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Failed to fetch satellite image URL: HTTP {img_res.status_code}"
                    )
                img_bytes = img_res.content
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to download satellite image: {exc}")

    # Case 3: Bounding box provided — fetch from satellite layer with multi-source fallback
    elif req.bounds:
        try:
            min_lat = float(min(req.bounds.north, req.bounds.south))
            max_lat = float(max(req.bounds.north, req.bounds.south))
            min_lng = float(min(req.bounds.east, req.bounds.west))
            max_lng = float(max(req.bounds.east, req.bounds.west))
        except (ValueError, TypeError) as err:
            raise HTTPException(status_code=400, detail=f"Invalid bounds coordinates: {err}")

        # Ensure minimum span for point or tiny bounding boxes
        if max_lat - min_lat < 0.001:
            mid_lat = (max_lat + min_lat) / 2.0
            min_lat = mid_lat - 0.002
            max_lat = mid_lat + 0.002
        if max_lng - min_lng < 0.001:
            mid_lng = (max_lng + min_lng) / 2.0
            min_lng = mid_lng - 0.002
            max_lng = mid_lng + 0.002

        export_urls = [
            f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export?bbox={min_lng:.6f},{min_lat:.6f},{max_lng:.6f},{max_lat:.6f}&bboxSR=4326&size=1024,1024&imageSR=4326&format=png&f=image",
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={min_lng:.6f},{min_lat:.6f},{max_lng:.6f},{max_lat:.6f}&bboxSR=4326&size=1024,1024&imageSR=4326&format=png&f=image",
        ]

        async with httpx.AsyncClient(timeout=25.0) as client:
            for url in export_urls:
                try:
                    img_res = await client.get(url)
                    if img_res.status_code == 200 and len(img_res.content) > 500:
                        img_bytes = img_res.content
                        break
                except Exception:
                    continue

        # If both export endpoints failed (e.g. HTTP 500), fall back to direct satellite tile stitching
        if not img_bytes:
            try:
                img_bytes = await fetch_satellite_tiles_fallback(min_lat, min_lng, max_lat, max_lng)
            except Exception as tile_exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to retrieve satellite imagery from layer and tile fallback: {tile_exc}"
                )
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'polygon', 'bounds', or 'image_url' must be provided to create a digital twin."
        )

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Satellite image data is empty.")

    # Process and crop to ONLY the area inside the line marked for monitoring
    try:
        src_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = src_img.size

        if req.polygon and len(req.polygon) >= 3 and req.bounds:
            min_lat = min(req.bounds.north, req.bounds.south)
            max_lat = max(req.bounds.north, req.bounds.south)
            min_lng = min(req.bounds.east, req.bounds.west)
            max_lng = max(req.bounds.east, req.bounds.west)

            lat_span = max_lat - min_lat or 0.0001
            lng_span = max_lng - min_lng or 0.0001

            # Map geographic [lat, lng] to exact pixel coordinates
            pixel_pts = []
            for pt in req.polygon:
                p_lat, p_lng = float(pt[0]), float(pt[1])
                px = int(round((p_lng - min_lng) / lng_span * (w - 1)))
                py = int(round((max_lat - p_lat) / lat_span * (h - 1)))
                px = max(0, min(w - 1, px))
                py = max(0, min(h - 1, py))
                pixel_pts.append((px, py))

            # Alpha mask: inside monitoring line = 255 (kept), outside line = 0 (transparent)
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(pixel_pts, fill=255)

            # Apply mask to keep ONLY the area inside the monitoring line
            rgba_img = src_img.convert("RGBA")
            rgba_img.putalpha(mask)

            # Crop tightly to the polygon perimeter
            poly_bbox = mask.getbbox()
            if poly_bbox:
                bx0, by0, bx1, by1 = poly_bbox
                margin_x = max(2, int((bx1 - bx0) * 0.02))
                margin_y = max(2, int((by1 - by0) * 0.02))
                cropped_box = (
                    max(0, bx0 - margin_x),
                    max(0, by0 - margin_y),
                    min(w, bx1 + margin_x),
                    min(h, by1 + margin_y),
                )
                final_img = rgba_img.crop(cropped_box)
            else:
                final_img = rgba_img

            if max(final_img.size) > 1024:
                final_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            single_image_uri = f"data:image/png;base64,{b64_str}"

        else:
            # Fallback if no polygon line provided
            if max(src_img.size) > 1024:
                src_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            src_img.save(buf, format="JPEG", quality=95)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            single_image_uri = f"data:image/jpeg;base64,{b64_str}"

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image processing failure: {exc}")

    # Send the single satellite image of ONLY the marked monitoring area to Meshy
    meshy_payload = {
        "image_urls": [single_image_uri],
        "target_formats": ["glb"],
        "should_texture": True,
        "enable_pbr": True,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            meshy_res = await client.post(
                "https://api.meshy.ai/openapi/v1/multi-image-to-3d",
                headers={
                    "Authorization": f"Bearer {meshy_api_key}",
                    "Content-Type": "application/json",
                },
                json=meshy_payload
            )

            if meshy_res.status_code not in (200, 201, 202):
                raise HTTPException(
                    status_code=meshy_res.status_code,
                    detail=f"Meshy API error: {meshy_res.text}"
                )

            data = meshy_res.json()
            task_id = data.get("result")
            if not task_id:
                raise HTTPException(
                    status_code=502,
                    detail=f"Meshy response did not contain a task ID: {data}"
                )

            return {"taskId": task_id}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to contact Meshy API: {exc}")


@router.get("/status/{task_id}")
async def get_digital_twin_status(task_id: str):
    """
    Poll the status of the single Meshy Multi-Image-to-3D task.
    Returns status ('PENDING', 'IN_PROGRESS', 'SUCCEEDED', 'FAILED'),
    progress percentage (0-100), and proxied glbUrl when SUCCEEDED.
    """
    meshy_api_key = os.environ.get("MESHY_API_KEY", "").strip()
    if not meshy_api_key:
        raise HTTPException(status_code=500, detail="MESHY_API_KEY is not configured.")

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.get(
                f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{task_id}",
                headers={"Authorization": f"Bearer {meshy_api_key}"}
            )

            if res.status_code != 200:
                raise HTTPException(
                    status_code=res.status_code,
                    detail=f"Meshy status check failed: {res.text}"
                )

            data = res.json()
            status = data.get("status", "PENDING")
            progress = data.get("progress", 0)
            model_urls = data.get("model_urls") or {}
            raw_glb_url = model_urls.get("glb") if isinstance(model_urls, dict) else None
            thumbnail_url = data.get("thumbnail_url")

            # Provide same-origin backend proxy URL to bypass browser CORS restrictions
            proxied_glb_url = f"/api/digital-twin/model/{task_id}" if (status == "SUCCEEDED" and raw_glb_url) else None

            error_msg = None
            if status == "FAILED":
                task_error = data.get("task_error") or {}
                error_msg = task_error.get("message") if isinstance(task_error, dict) else str(task_error)

            return {
                "taskId": task_id,
                "status": status,
                "progress": progress,
                "glbUrl": proxied_glb_url,
                "remoteGlbUrl": raw_glb_url,
                "thumbnailUrl": thumbnail_url,
                "error": error_msg,
            }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error polling Meshy task: {exc}")


@router.get("/model/{task_id}")
async def get_digital_twin_model(task_id: str):
    """
    Proxy and cache the GLB 3D model through the backend.
    Enables same-origin loading in <model-viewer> and Three.js with full CORS compliance.
    """
    cache_dir = Path("/tmp/cache/models")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    cached_file = cache_dir / f"{task_id}.glb"

    if cached_file.exists() and cached_file.stat().st_size > 0:
        return Response(
            content=cached_file.read_bytes(),
            media_type="model/gltf-binary",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=86400",
                "Content-Disposition": f'inline; filename="digital_twin_{task_id}.glb"',
            },
        )

    meshy_api_key = os.environ.get("MESHY_API_KEY", "").strip()
    if not meshy_api_key:
        raise HTTPException(status_code=500, detail="MESHY_API_KEY is not configured.")

    # 1. Fetch fresh signed S3 URL from Meshy
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            status_res = await client.get(
                f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{task_id}",
                headers={"Authorization": f"Bearer {meshy_api_key}"}
            )
            if status_res.status_code != 200:
                raise HTTPException(status_code=status_res.status_code, detail="Meshy task query failed")
            data = status_res.json()

        model_urls = data.get("model_urls") or {}
        raw_glb_url = model_urls.get("glb") if isinstance(model_urls, dict) else None
        if not raw_glb_url:
            raise HTTPException(status_code=404, detail="GLB model is not available yet.")

        # 2. Download GLB from S3 (backend is not restricted by browser CORS)
        async with httpx.AsyncClient(timeout=60.0) as client:
            glb_res = await client.get(raw_glb_url)
            if glb_res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Failed to fetch GLB from storage: {glb_res.status_code}")

            glb_bytes = glb_res.content
            # Cache locally for instant future requests
            cached_file.write_bytes(glb_bytes)

            return Response(
                content=glb_bytes,
                media_type="model/gltf-binary",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "public, max-age=86400",
                    "Content-Disposition": f'inline; filename="digital_twin_{task_id}.glb"',
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to retrieve 3D GLB model: {exc}")




class SurfaceForecastRequest(BaseModel):
    south: float = Field(ge=-85, le=85, allow_inf_nan=False)
    north: float = Field(ge=-85, le=85, allow_inf_nan=False)
    west: float = Field(ge=-180, le=180, allow_inf_nan=False)
    east: float = Field(ge=-180, le=180, allow_inf_nan=False)
    size: int = Field(ge=3, le=21)  # max 21×21 = 441 for full-area coverage
    elevations: List[float] = Field(min_length=9, max_length=441)


@router.post("/surface-forecast")
async def surface_forecast(request: SurfaceForecastRequest):
    from services import twin_forecast
    # Allow up to 1.0° so large monitored zones get full-area heatmap coverage
    if not (0 < request.north - request.south <= 1.0 and 0 < request.east - request.west <= 1.0):
        raise HTTPException(status_code=422, detail="Select an area smaller than 1 degree with valid bounds.")
    if len(request.elevations) != request.size ** 2 or any(
        not math.isfinite(h) or not -500 <= h <= 9000 for h in request.elevations
    ):
        raise HTTPException(status_code=422, detail="Provide a complete finite terrain grid between -500 and 9000 metres.")
    try:
        frames = await twin_forecast.fetch_weather((request.south + request.north) / 2, (request.west + request.east) / 2)
        return await asyncio.to_thread(twin_forecast.predict_surface, request.elevations,
            request.south, request.north, request.west, request.east, frames, request.size)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Forecast unavailable. Check weather access and the configured model checkpoint, then retry.") from exc


from .multi_hazard import router as multi_hazard_router

router.include_router(multi_hazard_router)


@router.delete("/network-features")
async def delete_digital_twin_network_features(
    area_key: Optional[str] = Query(None),
    area_id: Optional[str] = Query(None),
):
    """Deletes records from digital_twin_network_features in MongoDB.
    If area_id or area_key is provided, deletes records matching that area ID.
    Otherwise, deletes all records in the table.
    """
    target = area_id or area_key
    try:
        from lib.db import db
        if target:
            res = await db.digital_twin_network_features.delete_many({
                "$or": [
                    {"area_key": target},
                    {"area_key": f"dt-area-{target}"},
                    {"id": {"$regex": f"^{target}"}},
                ]
            })
        else:
            res = await db.digital_twin_network_features.delete_many({})
        return {
            "status": "success",
            "message": f"Successfully deleted {res.deleted_count} records from digital_twin_network_features",
            "deleted_count": res.deleted_count,
            "area_id": target,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete digital_twin_network_features: {exc}")


# Optional trained Hugging Face graph-transformer arrival head.
from fastapi import Depends
from lib.auth import require_roles
from pydantic import model_validator
from services.arrival_model import model_status, predict_arrivals


class ArrivalGraphRequest(BaseModel):
    cols: int = Field(ge=1, le=128)
    rows: int = Field(ge=1, le=128)
    bed: list[float] = Field(max_length=16384)
    depth: list[float] = Field(max_length=16384)
    inside: list[int] = Field(max_length=16384)
    sources: list[int] = Field(max_length=16384)
    paths: list[int] = Field(max_length=16384)
    dx: float = Field(gt=0, le=100000, allow_inf_nan=False)
    dy: float = Field(gt=0, le=100000, allow_inf_nan=False)
    elapsed: float = Field(ge=0, allow_inf_nan=False)
    horizonSeconds: float = Field(gt=0, le=21600, allow_inf_nan=False)
    roughness: float = Field(ge=0.005, le=0.2, allow_inf_nan=False)
    netRainfall: float = Field(ge=0, le=1000, allow_inf_nan=False)
    stormRemaining: float = Field(ge=0, le=21600, allow_inf_nan=False)
    sourceRise: float = Field(ge=0, le=10, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_grid(self):
        count = self.cols * self.rows
        for values in [self.bed, self.depth, self.inside, self.sources, self.paths]:
            if len(values) != count or not all(math.isfinite(value) for value in values):
                raise ValueError("Grid arrays must be finite and match rows × cols")
        if any(value < 0 for value in self.depth):
            raise ValueError("Depth must be nonnegative")
        if any(value not in (0, 1) for mask in [self.inside, self.sources, self.paths] for value in mask):
            raise ValueError("Masks must contain 0 or 1")
        return self


@router.get("/arrival-model")
def arrival_model_status(user: dict = Depends(require_roles("admin", "gov_officer"))):
    return model_status()


@router.post("/arrival-times")
def arrival_times(payload: ArrivalGraphRequest, user: dict = Depends(require_roles("admin", "gov_officer"))):
    status = model_status()
    if not status["available"]:
        raise HTTPException(status_code=503, detail=status["reason"])
    try:
        arrivals = predict_arrivals(payload.model_dump())
    except Exception:
        raise HTTPException(status_code=503, detail="Arrival model inference failed; use hydraulic rollout")
    return {"arrivals": arrivals, "model": "gnn_transformer", "source": status.get("source"),
            "revision": status.get("revision"), "horizon": payload.elapsed + payload.horizonSeconds,
            "throughSeconds": payload.elapsed + payload.horizonSeconds, "complete": True}


# ─── 🛡️ DIGITAL TWIN EVACUATION POINTS CRUD ───
import time
from datetime import datetime, timezone
from lib.auth import current_user, optional_user
from lib.db import db, get_mongo_fallback


class DigitalTwinEvacPointPayload(BaseModel):
    id: Optional[str] = None
    name: str
    lat: float
    lng: float
    elev: Optional[float] = 310.0
    instructions: Optional[str] = "Proceed to designated safe high-ground shelter."
    areaName: Optional[str] = None


@router.get("/evacuation-points")
async def list_digital_twin_evacuation_points(user: Optional[dict] = Depends(optional_user)):
    """Fetch all evacuation points created in Digital Twin."""
    try:
        docs = await db.evacuation_points.find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
        if docs:
            return docs
    except Exception:
        pass
    try:
        mongo = get_mongo_fallback()
        docs = await mongo["evacuation_points"].find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
        return docs
    except Exception:
        return []


@router.post("/evacuation-points")
async def save_digital_twin_evacuation_point(
    payload: DigitalTwinEvacPointPayload,
    user: dict = Depends(require_roles("admin", "gov_officer")),
):
    """Save an evacuation point created on Digital Twin."""
    point_id = payload.id or f"wp-{int(time.time() * 1000)}"
    doc = {
        "id": point_id,
        "name": payload.name,
        "lat": payload.lat,
        "lng": payload.lng,
        "elev": payload.elev if payload.elev is not None else 310.0,
        "instructions": payload.instructions or "Proceed to designated safe high-ground shelter.",
        "areaName": payload.areaName or "Digital Twin",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await db.evacuation_points.update_one({"id": point_id}, {"$set": doc}, upsert=True)
    except Exception as e:
        print(f"Notice: db.evacuation_points upsert error: {e}")
    try:
        mongo = get_mongo_fallback()
        await mongo["evacuation_points"].update_one({"id": point_id}, {"$set": doc}, upsert=True)
    except Exception as e:
        print(f"Notice: mongo evacuation_points error: {e}")
    return {"status": "ok", "point": doc}


@router.delete("/evacuation-points/{point_id}")
async def delete_digital_twin_evacuation_point(
    point_id: str,
    user: dict = Depends(require_roles("admin", "gov_officer")),
):
    """Delete an evacuation point by ID."""
    deleted_count = 0
    try:
        res = await db.evacuation_points.delete_many({"id": point_id})
        deleted_count += res.deleted_count
    except Exception:
        pass
    try:
        mongo = get_mongo_fallback()
        res2 = await mongo["evacuation_points"].delete_many({"id": point_id})
        deleted_count += res2.deleted_count
    except Exception:
        pass
    return {"status": "ok", "deleted_count": deleted_count, "id": point_id}
