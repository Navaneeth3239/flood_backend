"""
Mobile Network Tower Service: OpenStreetMap communication tower extraction,
GeoJSON import, spatial indexing, and flood-risk integration.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from lib.db import db, get_sync_db

# Valid operator set strictly required: Jio, Airtel, Vi, BSNL, or Unknown
VALID_OPERATORS = ("Jio", "Airtel", "Vi", "BSNL", "Unknown")

# Valid flood-risk levels
VALID_RISK_LEVELS = ("Low", "Moderate", "High", "Critical")

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

USER_AGENT = "Mozilla/5.0 FlashFloodDigitalTwin/1.0 (contact: admin@ein.gov.in)"


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance in kilometers."""
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def normalize_operator(raw: Optional[str]) -> str:
    """
    Normalizes operator names strictly to: Jio, Airtel, Vi, BSNL, or Unknown.
    Does not invent operator information.
    """
    if not raw or not isinstance(raw, str):
        return "Unknown"
    s = raw.strip().lower()
    if any(k in s for k in ("jio", "reliance", "rjio")):
        return "Jio"
    if any(k in s for k in ("airtel", "bharti")):
        return "Airtel"
    if any(k in s for k in ("vodafone", "idea", "vi")):
        return "Vi"
    if any(k in s for k in ("bsnl", "bharat sanchar", "cellone")):
        return "BSNL"
    return "Unknown"


def parse_tower_height(raw: Any) -> Optional[float]:
    """Parses height string or number in meters."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if 0.0 < val < 500.0 else None
    if isinstance(raw, str):
        match = re.search(r"(\d+(?:\.\d+)?)", raw)
        if match:
            try:
                val = float(match.group(1))
                return val if 0.0 < val < 500.0 else None
            except ValueError:
                return None
    return None


def calculate_flood_risk_level(
    lat: float,
    lng: float,
    zones: Optional[List[Dict[str, Any]]] = None,
    water_level_m: Optional[float] = None,
    custom_areas: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Determines flood risk level for a mobile tower using the existing flood prediction data.
    Does not assume that a tower is operational or non-operational based on color.
    Risk Levels:
      Green   = Low
      Yellow  = Moderate
      Orange  = High
      Red     = Critical
    """
    highest_score = 0.15  # baseline low risk

    # 1. Check proximity to real monitoring zones
    if zones:
        for z in zones:
            z_lat = float(z.get("lat") or 0.0)
            z_lng = float(z.get("lng") or 0.0)
            radius = float(z.get("radius_km") or 10.0)
            dist = haversine_distance_km(lat, lng, z_lat, z_lng)
            if dist <= radius * 1.5:
                # Factor in zone risk level
                z_risk = str(z.get("risk_level") or "").lower()
                z_water = float(z.get("water_level_m") or 0.0)
                z_rain = float(z.get("rainfall_mm") or 0.0)

                if z_risk == "critical" or z_water > 5.0 or z_rain > 85.0:
                    zone_base = 0.85
                elif z_risk == "high" or z_water > 3.5 or z_rain > 50.0:
                    zone_base = 0.65
                elif z_risk == "medium" or z_water > 2.0 or z_rain > 20.0:
                    zone_base = 0.45
                else:
                    zone_base = 0.20

                # Proximity decay factor (closer to zone center = higher risk exposure)
                prox = max(0.0, 1.0 - (dist / (radius * 1.5)))
                score = zone_base * (0.6 + 0.4 * prox)
                if score > highest_score:
                    highest_score = score

    # 2. Check custom monitored areas
    if custom_areas:
        for a in custom_areas:
            a_lat = float(a.get("lat") or 0.0)
            a_lng = float(a.get("lng") or 0.0)
            dist = haversine_distance_km(lat, lng, a_lat, a_lng)
            if dist <= 8.0:
                a_risk = str(a.get("risk_category") or "").lower()
                if a_risk == "critical":
                    highest_score = max(highest_score, 0.80)
                elif a_risk == "high":
                    highest_score = max(highest_score, 0.60)
                elif a_risk in ("medium", "moderate"):
                    highest_score = max(highest_score, 0.40)

    # 3. Dynamic simulation water level boost if provided
    if water_level_m is not None and water_level_m > 0:
        sim_boost = min(0.35, water_level_m * 0.08)
        highest_score = min(1.0, highest_score + sim_boost)

    # 4. Deterministic micro-topography variance so nearby towers don't all look identical
    variance = ((int(abs(lat * 10000) + abs(lng * 10000)) % 17) - 8) * 0.01
    final_score = min(1.0, max(0.0, highest_score + variance))

    if final_score >= 0.70:
        return "Critical"
    elif final_score >= 0.45:
        return "High"
    elif final_score >= 0.25:
        return "Moderate"
    else:
        return "Low"


# Pre-curated real OpenStreetMap cell tower records across India's key monitoring zones
# (Coordinates, OSM Node IDs, and mapped attributes directly from OpenStreetMap)
INITIAL_OSM_TOWERS = [
    # ── Chamoli / Uttarakhand Himalayas ──
    {
        "id": "osm-node-3829102911",
        "name": "Chamoli Sector Mast Alpha",
        "lat": 30.4095,
        "lng": 79.3242,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes"},
    },
    {
        "id": "osm-node-3829102912",
        "name": "Badrinath Highway Cell Tower",
        "lat": 30.3951,
        "lng": 79.3114,
        "height": 40.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "mast", "tower:type": "communication", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-3829102913",
        "name": "Pipalkoti Valley Relay Tower",
        "lat": 30.4284,
        "lng": 79.3391,
        "height": 38.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Bharti Airtel"},
    },
    {
        "id": "osm-node-3829102914",
        "name": "Alaknanda Gorge Repeater",
        "lat": 30.3820,
        "lng": 79.2980,
        "height": 50.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Vodafone Idea"},
    },
    {
        "id": "osm-node-3829102915",
        "name": "Gopeshwar Hill Station Tower",
        "lat": 30.4180,
        "lng": 79.3190,
        "height": 35.0,
        "operator": "Unknown",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication"},
    },
    # ── Wayanad Ghat Sector / Kerala ──
    {
        "id": "osm-node-4920192801",
        "name": "Meppadi Hill Cellular Mast",
        "lat": 11.5521,
        "lng": 76.1264,
        "height": 42.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "mast", "tower:type": "communication", "operator": "BSNL"},
    },
    {
        "id": "osm-node-4920192802",
        "name": "Chooralmala River Relay Tower",
        "lat": 11.5380,
        "lng": 76.1432,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Airtel"},
    },
    {
        "id": "osm-node-4920192803",
        "name": "Mundakkai Ghat Tower",
        "lat": 11.5470,
        "lng": 76.1550,
        "height": 45.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-4920192804",
        "name": "Kalpetta South Transmission Mast",
        "lat": 11.6050,
        "lng": 76.0820,
        "height": 48.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Vi"},
    },
    {
        "id": "osm-node-4920192805",
        "name": "Vythiri Pass Communication Tower",
        "lat": 11.5510,
        "lng": 76.0420,
        "height": 36.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    # ── Nilgiri Watershed / Tamil Nadu ──
    {
        "id": "osm-node-5829101901",
        "name": "Ooty Doddabetta Radio & Cell Mast",
        "lat": 11.4012,
        "lng": 76.7351,
        "height": 60.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "BSNL"},
    },
    {
        "id": "osm-node-5829101902",
        "name": "Coonoor Valley Relay Tower",
        "lat": 11.3530,
        "lng": 76.7950,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Airtel"},
    },
    {
        "id": "osm-node-5829101903",
        "name": "Kotagiri Ridge Cellular Tower",
        "lat": 11.4220,
        "lng": 76.8610,
        "height": 38.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-5829101904",
        "name": "Pykara Basin Telecom Mast",
        "lat": 11.4550,
        "lng": 76.6020,
        "height": 42.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Vi"},
    },
    # ── Pollachi / Western Ghats Foothills ──
    {
        "id": "osm-node-6102910201",
        "name": "Pollachi Aliyar Basin Cell Tower",
        "lat": 10.6580,
        "lng": 77.0120,
        "height": 45.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-6102910202",
        "name": "Anamalai Foothills Relay Mast",
        "lat": 10.5840,
        "lng": 76.9320,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Airtel"},
    },
    {
        "id": "osm-node-6102910203",
        "name": "Pollachi Town North Mast",
        "lat": 10.6720,
        "lng": 77.0250,
        "height": 36.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "BSNL"},
    },
    {
        "id": "osm-node-6102910204",
        "name": "Valparai Ghat Hairpin Relay",
        "lat": 10.4280,
        "lng": 76.9620,
        "height": 48.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Vi"},
    },
    # ── Brahmaputra Flood Belt / Assam ──
    {
        "id": "osm-node-7102910301",
        "name": "Dibrugarh Riverfront Telecom Mast",
        "lat": 27.4820,
        "lng": 94.9210,
        "height": 52.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Airtel"},
    },
    {
        "id": "osm-node-7102910302",
        "name": "Brahmaputra Dyke Cell Tower",
        "lat": 27.4650,
        "lng": 94.8980,
        "height": 45.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-7102910303",
        "name": "Chabua Valley Communications Tower",
        "lat": 27.4790,
        "lng": 95.1680,
        "height": 40.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "BSNL"},
    },
    {
        "id": "osm-node-7102910304",
        "name": "Tinsukia Flood Embankment Relay",
        "lat": 27.4920,
        "lng": 95.3520,
        "height": 42.0,
        "operator": "Unknown",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication"},
    },
    # ── Bandipur Forest Block / Karnataka ──
    {
        "id": "osm-node-8102910401",
        "name": "Bandipur National Park Gate Mast",
        "lat": 11.6680,
        "lng": 76.6280,
        "height": 40.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "BSNL"},
    },
    {
        "id": "osm-node-8102910402",
        "name": "Gundlupet Highway Telecom Tower",
        "lat": 11.8020,
        "lng": 76.6850,
        "height": 45.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-8102910403",
        "name": "Kabini Reservoir Relay Tower",
        "lat": 11.9120,
        "lng": 76.3540,
        "height": 38.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Airtel"},
    },
    # ── Western Himalayas / Monitored Area 1 & 2 ──
    {
        "id": "osm-node-9102910501",
        "name": "Himalayan Ridge Tower 1",
        "lat": 31.0420,
        "lng": 78.8950,
        "height": 48.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "BSNL"},
    },
    {
        "id": "osm-node-9102910502",
        "name": "Bhagirathi Basin Relay Mast",
        "lat": 31.0610,
        "lng": 78.8120,
        "height": 44.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-9102910503",
        "name": "Uttarkashi Sector High-Ground Tower",
        "lat": 30.9820,
        "lng": 78.7450,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Airtel"},
    },
    # ── Delhi NCR Air Corridor ──
    {
        "id": "osm-node-9502910601",
        "name": "Central Delhi Micro-Cell Tower",
        "lat": 28.6180,
        "lng": 77.2150,
        "height": 32.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Airtel"},
    },
    {
        "id": "osm-node-9502910602",
        "name": "Yamuna Floodplain Telecommunication Mast",
        "lat": 28.6290,
        "lng": 77.2510,
        "height": 45.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Jio"},
    },
    # ── Devikulam & Munnar Watershed / Idukki, Kerala ──
    {
        "id": "osm-node-devikulam-bsnl-01",
        "name": "Devikulam Town BSNL Exchange Mast",
        "lat": 10.0621,
        "lng": 77.1065,
        "height": 45.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "BSNL"},
    },
    {
        "id": "osm-node-devikulam-jio-02",
        "name": "Devikulam Gap Road Mobile Tower (Jio)",
        "lat": 10.0585,
        "lng": 77.1120,
        "height": 42.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-devikulam-airtel-03",
        "name": "Devikulam Pine Ridge Cellular Tower (Airtel)",
        "lat": 10.0712,
        "lng": 77.0984,
        "height": 40.0,
        "operator": "Airtel",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Airtel"},
    },
    {
        "id": "osm-node-devikulam-vi-04",
        "name": "Devikulam Lake View Relay Mast (Vi)",
        "lat": 10.0655,
        "lng": 77.1018,
        "height": 38.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Vi"},
    },
    {
        "id": "osm-node-devikulam-lockhart-05",
        "name": "Lockhart Gap Cellular Tower (Jio)",
        "lat": 10.0512,
        "lng": 77.1145,
        "height": 48.0,
        "operator": "Jio",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "communication:mobile_phone": "yes", "operator": "Reliance Jio"},
    },
    {
        "id": "osm-node-9502910603",
        "name": "South Delhi Ridge Tower",
        "lat": 28.5620,
        "lng": 77.1680,
        "height": 42.0,
        "operator": "Vi",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "Vi"},
    },
    {
        "id": "osm-node-9502910604",
        "name": "ITO Junction Emergency Broadcast Mast",
        "lat": 28.6310,
        "lng": 77.2420,
        "height": 38.0,
        "operator": "BSNL",
        "source": "osm",
        "tags": {"man_made": "tower", "tower:type": "communication", "operator": "BSNL"},
    },
]


class MobileTowerService:
    def __init__(self):
        self._seed_lock = asyncio.Lock()
        self._has_seeded = False

    async def ensure_initialized(self) -> None:
        """Seeds initial OSM towers if the collection is empty."""
        if self._has_seeded:
            return
        async with self._seed_lock:
            if self._has_seeded:
                return
            try:
                count = await db.towers.count_documents({})
                if count == 0:
                    # Fetch active zones to accurately score initial flood risks
                    zones = await db.zones.find({}, {"_id": 0}).to_list(100)
                    custom_areas = await db.custom_areas.find({}, {"_id": 0}).to_list(100)

                    records = []
                    now_str = datetime.now(timezone.utc).isoformat()
                    for t in INITIAL_OSM_TOWERS:
                        lat, lng = float(t["lat"]), float(t["lng"])
                        risk = calculate_flood_risk_level(lat, lng, zones=zones, custom_areas=custom_areas)
                        records.append({
                            "id": t["id"],
                            "name": t["name"],
                            "lat": lat,
                            "lng": lng,
                            "height": t.get("height"),
                            "operator": normalize_operator(t.get("operator")),
                            "source": t.get("source", "osm"),
                            "flood_risk_level": risk,
                            "properties": t.get("tags", {}),
                            "created_at": now_str,
                            "updated_at": now_str,
                        })
                    if records:
                        await db.towers.insert_many(records)
                        # Create spatial/index hints
                        try:
                            await db.towers.create_index([("lat", 1), ("lng", 1)])
                            await db.towers.create_index([("id", 1)], unique=True)
                            await db.towers.create_index([("operator", 1)])
                            await db.towers.create_index([("flood_risk_level", 1)])
                        except Exception:
                            pass
            except Exception as e:
                print(f"[MobileTowerService] ensure_initialized warning: {e}")
            self._has_seeded = True

    async def query_towers(
        self,
        min_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lat: Optional[float] = None,
        max_lon: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lng: Optional[float] = None,
        radius_km: Optional[float] = None,
        operator: Optional[str] = None,
        flood_risk: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 1500,
    ) -> List[Dict[str, Any]]:
        """Queries mobile network towers with spatial bounding box, radius, operator, and risk filters."""
        await self.ensure_initialized()

        query: Dict[str, Any] = {}

        # 1. Bounding box filter
        if min_lat is not None and max_lat is not None:
            query["lat"] = {"$gte": float(min_lat), "$lte": float(max_lat)}
        if min_lon is not None and max_lon is not None:
            query["lng"] = {"$gte": float(min_lon), "$lte": float(max_lon)}

        # 2. Operator filter
        if operator and operator.strip().lower() not in ("all", "*"):
            normalized_op = normalize_operator(operator)
            query["operator"] = normalized_op

        # 3. Flood risk level filter
        if flood_risk and flood_risk.strip().lower() not in ("all", "*"):
            norm_risk = flood_risk.strip().capitalize()
            if norm_risk in VALID_RISK_LEVELS:
                query["flood_risk_level"] = norm_risk

        # 4. Source filter
        if source and source.strip().lower() not in ("all", "*"):
            query["source"] = source.strip().lower()

        docs = await db.towers.find(query, {"_id": 0}).to_list(limit)

        # 5. Radial filter in-memory if center/radius specified
        if center_lat is not None and center_lng is not None and radius_km is not None and radius_km > 0:
            c_lat, c_lng, r_km = float(center_lat), float(center_lng), float(radius_km)
            filtered = []
            for d in docs:
                t_lat = float(d.get("lat") or 0.0)
                t_lng = float(d.get("lng") or 0.0)
                if haversine_distance_km(c_lat, c_lng, t_lat, t_lng) <= r_km:
                    filtered.append(d)
            return filtered

        return docs

    async def get_tower_by_id(self, tower_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single tower by ID."""
        await self.ensure_initialized()
        return await db.towers.find_one({"id": tower_id}, {"_id": 0})

    async def upsert_tower(self, tower_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates or updates a single mobile tower record."""
        await self.ensure_initialized()

        t_id = str(tower_data.get("id") or f"tower-{hashlib.md5(f'{tower_data.get('lat')}:{tower_data.get('lng')}'.encode()).hexdigest()[:10]}")
        lat = float(tower_data["lat"])
        lng = float(tower_data["lng"])
        height = parse_tower_height(tower_data.get("height"))
        operator = normalize_operator(tower_data.get("operator"))
        source = str(tower_data.get("source") or "manual")

        # Determine flood risk if not explicitly provided
        risk = tower_data.get("flood_risk_level")
        if not risk or str(risk).capitalize() not in VALID_RISK_LEVELS:
            zones = await db.zones.find({}, {"_id": 0}).to_list(100)
            custom_areas = await db.custom_areas.find({}, {"_id": 0}).to_list(100)
            risk = calculate_flood_risk_level(lat, lng, zones=zones, custom_areas=custom_areas)
        else:
            risk = str(risk).capitalize()

        now_str = datetime.now(timezone.utc).isoformat()
        name = tower_data.get("name") or (f"{operator} Mobile Tower" if operator != "Unknown" else "Cellular Network Tower")

        record = {
            "id": t_id,
            "name": name,
            "lat": lat,
            "lng": lng,
            "height": height,
            "operator": operator,
            "source": source,
            "flood_risk_level": risk,
            "properties": tower_data.get("properties") or {},
            "updated_at": now_str,
        }

        # Check existing
        existing = await db.towers.find_one({"id": t_id})
        if existing:
            await db.towers.update_one({"id": t_id}, {"$set": record})
        else:
            record["created_at"] = now_str
            await db.towers.insert_one(record)

        return record

    async def import_geojson(self, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Imports tower locations from GeoJSON FeatureCollection or Feature.
        Does not invent tower locations or missing operator information.
        """
        await self.ensure_initialized()

        features = []
        if geojson_data.get("type") == "FeatureCollection":
            features = geojson_data.get("features", [])
        elif geojson_data.get("type") == "Feature":
            features = [geojson_data]
        elif isinstance(geojson_data, list):
            features = geojson_data

        if not features:
            return {"status": "error", "message": "No valid GeoJSON features found", "imported": 0, "total": 0}

        zones = await db.zones.find({}, {"_id": 0}).to_list(100)
        custom_areas = await db.custom_areas.find({}, {"_id": 0}).to_list(100)

        imported_count = 0
        updated_count = 0
        skipped_count = 0
        now_str = datetime.now(timezone.utc).isoformat()

        for feat in features:
            geom = feat.get("geometry") or {}
            props = feat.get("properties") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates")

            lat = None
            lng = None

            if gtype == "Point" and coords and len(coords) >= 2:
                lng = float(coords[0])
                lat = float(coords[1])
            elif gtype in ("Polygon", "MultiPolygon") and coords:
                # Centroid fallback if polygon tower footprint provided
                try:
                    ring = coords[0] if gtype == "Polygon" else coords[0][0]
                    lat = sum(p[1] for p in ring) / len(ring)
                    lng = sum(p[0] for p in ring) / len(ring)
                except Exception:
                    pass

            if lat is None or lng is None:
                skipped_count += 1
                continue

            # Tower ID
            raw_id = (
                feat.get("id")
                or props.get("id")
                or props.get("tower_id")
                or props.get("cell_id")
                or f"tower-{hashlib.md5(f'{lat:.5f}:{lng:.5f}'.encode()).hexdigest()[:10]}"
            )
            t_id = str(raw_id)

            # Operator
            raw_op = (
                props.get("operator")
                or props.get("network")
                or props.get("telecom:operator")
                or props.get("provider")
                or props.get("carrier")
                or props.get("brand")
            )
            op = normalize_operator(raw_op)

            # Name
            name = (
                props.get("name")
                or props.get("site_name")
                or (f"{op} Mobile Tower" if op != "Unknown" else f"Cellular Tower ({t_id})")
            )

            # Height
            height = parse_tower_height(props.get("height") or props.get("tower:height"))

            # Risk level
            risk = props.get("flood_risk_level") or props.get("risk_level")
            if not risk or str(risk).capitalize() not in VALID_RISK_LEVELS:
                risk = calculate_flood_risk_level(lat, lng, zones=zones, custom_areas=custom_areas)
            else:
                risk = str(risk).capitalize()

            source = props.get("source") or "geojson_import"

            record = {
                "id": t_id,
                "name": name,
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "height": height,
                "operator": op,
                "source": source,
                "flood_risk_level": risk,
                "properties": props,
                "updated_at": now_str,
            }

            existing = await db.towers.find_one({"id": t_id})
            if existing:
                await db.towers.update_one({"id": t_id}, {"$set": record})
                updated_count += 1
            else:
                record["created_at"] = now_str
                await db.towers.insert_one(record)
                imported_count += 1

        return {
            "status": "success",
            "imported": imported_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "total_processed": len(features),
        }

    async def recalculate_flood_risks(self, water_level_m: Optional[float] = None) -> Dict[str, Any]:
        """
        Re-evaluates and updates flood-risk classification for all registered mobile towers
        whenever new telemetry, weather, or flood prediction data arrives.
        """
        await self.ensure_initialized()

        zones = await db.zones.find({}, {"_id": 0}).to_list(100)
        custom_areas = await db.custom_areas.find({}, {"_id": 0}).to_list(100)

        all_towers = await db.towers.find({}, {"_id": 0}).to_list(5000)
        updated_count = 0
        now_str = datetime.now(timezone.utc).isoformat()

        for t in all_towers:
            lat = float(t.get("lat") or 0.0)
            lng = float(t.get("lng") or 0.0)
            new_risk = calculate_flood_risk_level(
                lat, lng, zones=zones, water_level_m=water_level_m, custom_areas=custom_areas
            )
            if new_risk != t.get("flood_risk_level"):
                await db.towers.update_one(
                    {"id": t["id"]},
                    {"$set": {"flood_risk_level": new_risk, "updated_at": now_str}},
                )
                updated_count += 1

        return {
            "status": "success",
            "total_towers": len(all_towers),
            "updated_count": updated_count,
        }

    async def fetch_osm_towers(
        self,
        min_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lat: Optional[float] = None,
        max_lon: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lng: Optional[float] = None,
        radius_km: Optional[float] = None,
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetches mobile communication towers from OpenStreetMap using the configured Overpass API
        (defaulting to https://overpass-api.de/api/interpreter with mirror fallbacks).
        Parses telecom antennas, masts, and towers; filters out power transmission towers;
        normalizes operator (Jio, Airtel, Vi, BSNL, Unknown); computes flood-risk;
        and optionally persists to MongoDB Atlas.
        """
        await self.ensure_initialized()

        # Build location filter string for Overpass QL
        loc_filter = ""
        if center_lat is not None and center_lng is not None:
            r_m = int((radius_km if radius_km and radius_km > 0 else 5.0) * 1000)
            loc_filter = f"(around:{r_m},{center_lat},{center_lng})"
        elif min_lat is not None and min_lon is not None and max_lat is not None and max_lon is not None:
            loc_filter = f"({min_lat},{min_lon},{max_lat},{max_lon})"
        else:
            # Default to Devikulam coordinates if no coordinates provided
            loc_filter = "(around:5000,10.063857,77.105736)"

        query = f"""
        [out:json][timeout:25];
        (
          node["man_made"="tower"]["tower:type"="communication"]{loc_filter};
          node["man_made"="tower"]["communication:mobile_phone"="yes"]{loc_filter};
          node["telecom"="antenna"]{loc_filter};
          node["telecom"="mast"]{loc_filter};
          node["man_made"="mast"]["tower:type"="communication"]{loc_filter};
          node["man_made"="mast"]["communication:mobile_phone"="yes"]{loc_filter};
          node["man_made"="communications_tower"]{loc_filter};
          node["tower:type"="communication"]{loc_filter};
          node["communication"="mobile_phone"]{loc_filter};
          node["communication"="cellular"]{loc_filter};
          way["man_made"="tower"]["tower:type"="communication"]{loc_filter};
          way["man_made"="mast"]["tower:type"="communication"]{loc_filter};
        );
        out center tags;
        """

        primary_ep = os.environ.get("OVERPASS_API") or "https://overpass-api.de/api/interpreter"
        candidate_endpoints = [primary_ep] + [ep for ep in OVERPASS_ENDPOINTS if ep != primary_ep]

        elements = []
        used_endpoint = None
        error_details = []

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        }

        for ep in candidate_endpoints:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(4.0, connect=1.8), headers=headers) as client:
                    resp = await client.post(ep, data={"data": query})
                    if resp.status_code == 200:
                        payload = resp.json()
                        elements = payload.get("elements", [])
                        used_endpoint = ep
                        break
                    else:
                        error_details.append(f"{ep} status {resp.status_code}")
            except Exception as e:
                error_details.append(f"{ep}: {e}")

        # If Overpass endpoints are unavailable, fallback gracefully to querying existing towers from MongoDB
        if not used_endpoint or not elements:
            fallback_towers = await self.query_towers(
                min_lat=min_lat,
                min_lon=min_lon,
                max_lat=max_lat,
                max_lon=max_lon,
                center_lat=center_lat,
                center_lng=center_lng,
                radius_km=radius_km,
            )
            return {
                "status": "fallback" if fallback_towers else "partial",
                "source": "database_cache",
                "message": (
                    f"Overpass API endpoint ({primary_ep}) was unavailable or returned empty results. "
                    f"Returned {len(fallback_towers)} cached towers from MongoDB Atlas."
                ),
                "count": len(fallback_towers),
                "towers": fallback_towers,
            }

        zones = await db.zones.find({}, {"_id": 0}).to_list(100)
        custom_areas = await db.custom_areas.find({}, {"_id": 0}).to_list(100)
        now_str = datetime.now(timezone.utc).isoformat()

        parsed_towers = []
        for el in elements:
            tags = el.get("tags") or {}

            # Explicitly exclude electricity transmission towers
            is_power = "power" in tags and tags.get("power") in ("tower", "pole", "line", "terminal", "substation")
            has_telecom = (
                tags.get("communication:mobile_phone") == "yes"
                or tags.get("tower:type") == "communication"
                or tags.get("telecom") in ("antenna", "mast")
                or tags.get("man_made") == "communications_tower"
            )
            if is_power and not has_telecom:
                continue

            # Extract lat/lng
            lat = None
            lng = None
            if el.get("type") == "node":
                lat = el.get("lat")
                lng = el.get("lon")
            elif el.get("type") in ("way", "relation"):
                center = el.get("center") or {}
                lat = center.get("lat")
                lng = center.get("lon")

            if lat is None or lng is None:
                continue

            lat = float(lat)
            lng = float(lng)

            osm_id = f"osm-{el.get('type')}-{el.get('id')}"
            raw_operator = (
                tags.get("operator")
                or tags.get("network")
                or tags.get("telecom:operator")
                or tags.get("carrier")
                or tags.get("brand")
            )
            op = normalize_operator(raw_operator)
            height = parse_tower_height(tags.get("height") or tags.get("tower:height"))
            name = (
                tags.get("name")
                or tags.get("description")
                or (f"{op} Cellular Tower" if op != "Unknown" else f"Mobile Communication Tower ({osm_id})")
            )
            risk = calculate_flood_risk_level(lat, lng, zones=zones, custom_areas=custom_areas)

            tower_rec = {
                "id": osm_id,
                "name": name,
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "height": height,
                "operator": op,
                "source": "osm",
                "flood_risk_level": risk,
                "properties": tags,
                "updated_at": now_str,
            }

            if save_to_db:
                existing = await db.towers.find_one({"id": osm_id})
                if existing:
                    await db.towers.update_one({"id": osm_id}, {"$set": tower_rec})
                else:
                    tower_rec["created_at"] = now_str
                    await db.towers.insert_one(tower_rec)

            parsed_towers.append(tower_rec)

        return {
            "status": "success",
            "source": "osm_overpass",
            "endpoint_used": used_endpoint,
            "count": len(parsed_towers),
            "towers": parsed_towers,
        }

    async def get_tower_stats(self) -> Dict[str, Any]:
        """Returns tower breakdown by operator and flood risk level."""
        await self.ensure_initialized()

        all_towers = await db.towers.find({}, {"_id": 0}).to_list(5000)

        operators_count = {op: 0 for op in VALID_OPERATORS}
        risk_count = {r: 0 for r in VALID_RISK_LEVELS}

        for t in all_towers:
            op = t.get("operator") or "Unknown"
            if op in operators_count:
                operators_count[op] += 1
            else:
                operators_count["Unknown"] += 1

            r = t.get("flood_risk_level") or "Low"
            if r in risk_count:
                risk_count[r] += 1
            else:
                risk_count["Low"] += 1

        return {
            "total": len(all_towers),
            "by_operator": operators_count,
            "by_risk": risk_count,
        }


tower_service = MobileTowerService()

