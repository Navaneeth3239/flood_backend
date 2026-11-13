"""
Permanent Monitoring Areas Service:
Manages protected, permanent geographical monitoring areas (e.g. Devikulam, Kerala).
Ensures idempotent database persistence, deletion safeguards, and pre-warmed layer caching.
"""
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
import json
import gzip
from pathlib import Path

logger = logging.getLogger(__name__)

DEVIKULAM_AREA_ID = "devikulam-kerala-permanent-01"

DEVIKULAM_COORDINATES = {
    "lat": 10.063857,
    "lng": 77.105736,
    "radius_km": 2.0,
    "radius_m": 2000,
}

# 4-corner rectangular bounding box polygon (matching reference monitoring zone)
DEVIKULAM_BOUNDS = [[10.045839, 77.087413], [10.081875, 77.124059]]

DEVIKULAM_POLYGON_COORDS = [
    [10.045839, 77.087413], # South-West
    [10.081875, 77.087413], # North-West
    [10.081875, 77.124059], # North-East
    [10.045839, 77.124059], # South-East
    [10.045839, 77.087413], # Closed loop
]

DEVIKULAM_PERMANENT_AREA: Dict[str, Any] = {
    "id": "devikulam-kerala-permanent-01",
    "name": "Devikulam, Kerala",
    "district": "Idukki",
    "state": "Kerala",
    "area_type": "Highland Watershed & Basin",
    "risk_category": "High",
    "priority": "Critical (Permanent)",
    "description": "Permanent Digital Twin monitoring area for Devikulam, Idukki District, Kerala.",
    "lat": DEVIKULAM_COORDINATES["lat"],
    "lng": DEVIKULAM_COORDINATES["lng"],
    "radius_km": DEVIKULAM_COORDINATES["radius_km"],
    "radius_m": DEVIKULAM_COORDINATES["radius_m"],
    "shape": "Polygon:[[10.045839,77.087413],[10.081875,77.087413],[10.081875,77.124059],[10.045839,77.124059]]",
    "polygon": DEVIKULAM_POLYGON_COORDS,
    "bounds": DEVIKULAM_BOUNDS,
    "area_sq_meters": 15760000,
    "is_permanent": True,
    "is_protected": True,
    "layer_config": {
        "rivers_color": "#1976D2",
        "roads_color": "#FB8C00",
        "buildings_color": "#E53935",
        "towers_color": "#22C55E"
    },
    "created_at": "2026-11-12T00:00:00.000Z"
}

# ── Munnar – Devikulam Corridor Permanent Monitoring Area ───────────────────────
MUNNAR_DEVIKULAM_AREA_ID = "munnar-devikulam-permanent-02"

MUNNAR_DEVIKULAM_COORDINATES = {
    "lat": 10.0763,
    "lng": 77.0826,
    "radius_km": 6.0,
    "radius_m": 6000,
}

MUNNAR_DEVIKULAM_BOUNDS = [[10.0350, 77.0450], [10.1150, 77.1350]]
MUNNAR_DEVIKULAM_POLYGON_COORDS = [
    [10.0350, 77.0450],
    [10.1150, 77.0450],
    [10.1150, 77.1350],
    [10.0350, 77.1350],
    [10.0350, 77.0450],
]

MUNNAR_DEVIKULAM_PERMANENT_AREA: Dict[str, Any] = {
    "id": "munnar-devikulam-permanent-02",
    "name": "Munnar – Devikulam, Kerala",
    "district": "Idukki",
    "state": "Kerala",
    "area_type": "Highland Mountain Watershed & River Basin",
    "risk_category": "Critical",
    "priority": "Critical (Permanent)",
    "description": "Permanent Digital Twin monitoring area covering Munnar – Devikulam river networks, roads, mobile towers, and building footprints.",
    "lat": MUNNAR_DEVIKULAM_COORDINATES["lat"],
    "lng": MUNNAR_DEVIKULAM_COORDINATES["lng"],
    "radius_km": MUNNAR_DEVIKULAM_COORDINATES["radius_km"],
    "radius_m": MUNNAR_DEVIKULAM_COORDINATES["radius_m"],
    "shape": "Polygon:[[10.035,77.045],[10.115,77.045],[10.115,77.135],[10.035,77.135]]",
    "polygon": MUNNAR_DEVIKULAM_POLYGON_COORDS,
    "bounds": MUNNAR_DEVIKULAM_BOUNDS,
    "area_sq_meters": 88000000,
    "is_permanent": True,
    "is_protected": True,
    "layer_config": {
        "rivers_color": "#1976D2",
        "roads_color": "#FB8C00",
        "buildings_color": "#E53935",
        "towers_color": "#22C55E"
    },
    "created_at": "2026-11-13T00:00:00.000Z"
}

DEVIKULAM_TOWERS = [
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-munnar-bsnl-06",
        "name": "Munnar Town Center Cellular Tower (BSNL)",
        "lat": 10.0889,
        "lng": 77.0595,
        "height": 52.0,
        "operator": "BSNL",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-munnar-jio-07",
        "name": "Old Munnar Bridge Mobile Mast (Jio)",
        "lat": 10.0820,
        "lng": 77.0550,
        "height": 44.0,
        "operator": "Jio",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-munnar-airtel-08",
        "name": "Munnar Tea Estate Cellular Relay (Airtel)",
        "lat": 10.0950,
        "lng": 77.0620,
        "height": 42.0,
        "operator": "Airtel",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-munnar-vi-09",
        "name": "Munnar Mattupetty Junction Tower (Vi)",
        "lat": 10.0910,
        "lng": 77.0720,
        "height": 40.0,
        "operator": "Vi",
        "source": "osm",
        "flood_risk_level": "Low",
        "properties": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
]


async def ensure_permanent_areas_initialized() -> None:
    """
    Idempotent initialization:
    Ensures Devikulam & Munnar–Devikulam permanent monitoring areas exist in MongoDB Atlas,
    mobile towers are registered, and their OSM geographical layers are cached.
    """
    try:
        from lib.db import db

        # 1. Custom Areas Collection
        for permanent_area in (DEVIKULAM_PERMANENT_AREA, MUNNAR_DEVIKULAM_PERMANENT_AREA):
            existing = await db.custom_areas.find_one({"id": permanent_area["id"]})
            if not existing:
                await db.custom_areas.insert_one(dict(permanent_area))
                logger.info("Initialized permanent monitoring area: %s (%s)", permanent_area["name"], permanent_area["id"])
            else:
                await db.custom_areas.update_one(
                    {"id": permanent_area["id"]},
                    {"$set": {
                        "name": permanent_area["name"],
                        "district": permanent_area["district"],
                        "state": permanent_area["state"],
                        "area_type": permanent_area["area_type"],
                        "risk_category": permanent_area["risk_category"],
                        "priority": permanent_area["priority"],
                        "description": permanent_area["description"],
                        "lat": permanent_area["lat"],
                        "lng": permanent_area["lng"],
                        "radius_km": permanent_area["radius_km"],
                        "radius_m": permanent_area["radius_m"],
                        "shape": permanent_area["shape"],
                        "polygon": permanent_area["polygon"],
                        "bounds": permanent_area["bounds"],
                        "area_sq_meters": permanent_area["area_sq_meters"],
                        "is_permanent": True,
                        "is_protected": True,
                        "layer_config": permanent_area["layer_config"],
                    }}
                )
                logger.info("Synchronized permanent & protected area '%s' in MongoDB Atlas", permanent_area["name"])

        # 2. Ensure mobile network towers for Devikulam
        now_str = datetime.now(timezone.utc).isoformat()
        for t in DEVIKULAM_TOWERS:
            existing_tower = await db.towers.find_one({"id": t["id"]})
            if not existing_tower:
                t_record = dict(t)
                t_record["created_at"] = now_str
                t_record["updated_at"] = now_str
                await db.towers.insert_one(t_record)

        # 3. Pre-cache layer GeoJSON in db.area_map_layers if available
        gz_path = Path(__file__).resolve().parent.parent / "data" / "prebaked_zones" / "devikulam-kerala-permanent-01.json.gz"
        if gz_path.exists():
            try:
                from services.area_map_store import boundary_key
                data = json.loads(gzip.decompress(gz_path.read_bytes()))
                b_key = boundary_key(DEVIKULAM_POLYGON_COORDS, data.get("bbox", {}))
                area_id = DEVIKULAM_PERMANENT_AREA["id"]

                for layer_name in ("roads", "rivers", "buildings"):
                    layer_data = data.get(layer_name, {}).get("geojson")
                    if layer_data:
                        await db.area_map_layers.update_one(
                            {"area_id": area_id, "boundary_key": b_key, "layer": layer_name},
                            {"$set": {
                                "area_id": area_id,
                                "boundary_key": b_key,
                                "layer": layer_name,
                                "geojson": layer_data,
                                "updated_at": now_str,
                            }},
                            upsert=True
                        )
                logger.info("Pre-cached Devikulam OSM geographical layers in MongoDB Atlas area_map_layers")
            except Exception as e:
                logger.warning("Could not pre-cache Devikulam layers to db: %s", e)

    except Exception as e:
        logger.error("Failed to ensure permanent monitoring areas initialized: %s", e)
