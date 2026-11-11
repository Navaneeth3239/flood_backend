"""Location Service: Place Geocoding, Area Selection, and Bounding Box Derivation."""
import math
from typing import Dict, Any, List, Optional, Tuple
import httpx


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two points on Earth in meters."""
    R = 6371000.0  # Earth's radius in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def bbox_from_radius(lat: float, lng: float, radius_km: float = 2.0) -> Dict[str, float]:
    """Computes bounding box (north, south, east, west) from center coordinate and radius."""
    radius_km = max(0.2, min(radius_km, 25.0))
    lat_delta = radius_km / 111.132
    # Adjust for longitudinal convergence
    cos_lat = math.cos(math.radians(lat))
    lng_delta = radius_km / (111.132 * max(cos_lat, 0.01))

    return {
        "north": round(lat + lat_delta, 6),
        "south": round(lat - lat_delta, 6),
        "east": round(lng + lng_delta, 6),
        "west": round(lng - lng_delta, 6),
        "center_lat": round(lat, 6),
        "center_lng": round(lng, 6),
        "radius_km": round(radius_km, 2),
    }


def bbox_from_polygon(coordinates: List[List[float]], pad_pct: float = 0.10) -> Dict[str, float]:
    """Computes bounding box from a polygon coordinate list [[lat, lng], ...]."""
    if not coordinates:
        raise ValueError("Polygon coordinates cannot be empty")

    lats = [float(p[0]) for p in coordinates]
    lngs = [float(p[1]) for p in coordinates]

    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)

    lat_span = max_lat - min_lat
    lng_span = max_lng - min_lng

    # Guarantee a minimum bbox span (~3.5 km) so rivers, tributaries, and road networks
    # connected to the area are always captured even if the polygon is small.
    min_span = 0.032
    if lat_span < min_span:
        pad_lat = (min_span - lat_span) / 2.0
        min_lat -= pad_lat
        max_lat += pad_lat
    else:
        lat_pad = lat_span * pad_pct
        min_lat -= lat_pad
        max_lat += lat_pad

    if lng_span < min_span:
        pad_lng = (min_span - lng_span) / 2.0
        min_lng -= pad_lng
        max_lng += pad_lng
    else:
        lng_pad = lng_span * pad_pct
        min_lng -= lng_pad
        max_lng += lng_pad

    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)

    return {
        "north": round(max_lat, 6),
        "south": round(min_lat, 6),
        "east": round(max_lng, 6),
        "west": round(min_lng, 6),
        "center_lat": round(center_lat, 6),
        "center_lng": round(center_lng, 6),
        "polygon_points": len(coordinates),
    }


_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = {}


async def geocode_place_name(place_name: str) -> Optional[Dict[str, Any]]:
    """Geocodes a place name to coordinates and bounding box via OSM Nominatim with in-memory caching."""
    trimmed = place_name.strip()
    if not trimmed:
        return None

    cache_key = trimmed.lower()
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]

    url = f"https://nominatim.openstreetmap.org/search?format=json&q={trimmed}&limit=1&addressdetails=1"
    headers = {"User-Agent": "FlashFloodPredictor/2.0 (emergency-response)"}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0:
                    item = data[0]
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    bbox_raw = item.get("boundingbox", [])
                    res = None
                    if len(bbox_raw) == 4:
                        res = {
                            "name": item.get("display_name", place_name),
                            "lat": lat,
                            "lng": lon,
                            "north": float(bbox_raw[1]),
                            "south": float(bbox_raw[0]),
                            "east": float(bbox_raw[3]),
                            "west": float(bbox_raw[2]),
                        }
                    else:
                        res = bbox_from_radius(lat, lon, 2.0)
                    if res:
                        _GEOCODE_CACHE[cache_key] = res
                    return res
    except Exception as e:
        print(f"Geocoding error for '{place_name}': {e}")

    return None



def point_in_polygon(lat: float, lng: float, polygon: Optional[List[List[float]]]) -> bool:
    """Ray casting point-in-polygon algorithm. polygon is [[lat, lng], ...]."""
    if not polygon or len(polygon) < 3:
        return True
    n = len(polygon)
    inside = False
    p1_lat, p1_lng = polygon[0]
    for i in range(1, n + 1):
        p2_lat, p2_lng = polygon[i % n]
        if min(p1_lat, p2_lat) < lat <= max(p1_lat, p2_lat):
            if lng <= max(p1_lng, p2_lng):
                if p1_lat != p2_lat:
                    xinters = (lat - p1_lat) * (p2_lng - p1_lng) / (p2_lat - p1_lat) + p1_lng
                if p1_lng == p2_lng or lng <= xinters:
                    inside = not inside
        p1_lat, p1_lng = p2_lat, p2_lng
    return inside
