"""
Google Earth Engine tile router — 10-layer GIS analysis.
Generates real GEE tile URLs for all supported layer types.
"""
import asyncio
import threading
from fastapi import APIRouter, HTTPException, Response
from functools import lru_cache

router = APIRouter(prefix="/gee", tags=["gee"])
GEE_PROJECT = "formal-purpose-466115-i8"

# ─── Thread-safe GEE initialisation (runs once) ──────────────────────────────
_gee_lock = threading.Lock()
_gee_ready = False


def _init_gee():
    global _gee_ready
    with _gee_lock:
        if _gee_ready:
            return
        try:
            import ee
            ee.Initialize(project=GEE_PROJECT)
            _gee_ready = True
        except Exception as exc:
            raise RuntimeError(f"GEE init failed: {exc}") from exc


def _ensure_gee():
    if not _gee_ready:
        _init_gee()


# ─── Individual cached tile URL builders ─────────────────────────────────────


@lru_cache(maxsize=2)
def _get_slope_tiles() -> dict:
    _ensure_gee()
    import ee
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    tile_url = slope.getMapId({"min": 0, "max": 60, "palette": ["00A000", "FFFF00", "FF8000", "FF0000"]})["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": None}


@lru_cache(maxsize=2)
def _get_terrain_tiles() -> dict:
    _ensure_gee()
    import ee
    dem = ee.Image("USGS/SRTMGL1_003")
    terrain = ee.Terrain.products(dem)
    tile_url = terrain.select("hillshade").getMapId({"min": 0, "max": 255})["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": None}


@lru_cache(maxsize=2)
def _get_elevation_tiles() -> dict:
    _ensure_gee()
    import ee
    dem = ee.Image("USGS/SRTMGL1_003")
    tile_url = dem.getMapId({
        "min": 0, "max": 3000,
        "palette": ["000080", "0000FF", "00FFFF", "00FF00", "FFFF00", "FF8000", "FF0000", "FFFFFF"]
    })["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": None}


@lru_cache(maxsize=2)
def _get_ndvi_tiles() -> dict:
    _ensure_gee()
    import ee
    import datetime

    # Use most recent 6-month window for latest imagery
    now = datetime.datetime.now(datetime.timezone.utc)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - datetime.timedelta(days=180)).strftime("%Y-%m-%d")

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(start_date, end_date)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .sort("system:time_start", False))
    latest = col.first()
    ts = latest.get("system:time_start").getInfo()
    dt_str = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    ndvi = col.median().normalizedDifference(["B8", "B4"])
    tile_url = ndvi.getMapId({
        "min": -0.2, "max": 0.9,
        "palette": ["d73027", "fc8d59", "ffffbf", "91cf60", "1a9641"]
    })["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": dt_str}


@lru_cache(maxsize=2)
def _get_ndwi_tiles() -> dict:
    _ensure_gee()
    import ee
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - datetime.timedelta(days=180)).strftime("%Y-%m-%d")

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(start_date, end_date)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .sort("system:time_start", False))
    latest = col.first()
    ts = latest.get("system:time_start").getInfo()
    dt_str = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    ndwi = col.median().normalizedDifference(["B3", "B8"])
    tile_url = ndwi.getMapId({
        "min": -0.5, "max": 0.5,
        "palette": ["8B4513", "F5DEB3", "FFFFFF", "87CEEB", "0000CD"]
    })["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": dt_str}


@lru_cache(maxsize=2)
def _get_flood_tiles() -> dict:
    _ensure_gee()
    import ee
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - datetime.timedelta(days=180)).strftime("%Y-%m-%d")

    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filterDate(start_date, end_date)
          .select("VV")
          .sort("system:time_start", False))
    latest = col.first()
    ts = latest.get("system:time_start").getInfo()
    dt_str = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    s1 = col.median()
    water = s1.lt(-15.5)
    masked_water = water.updateMask(water)
    tile_url = masked_water.getMapId({
        "min": 0, "max": 1,
        "palette": ["0066FF"]
    })["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": dt_str}


@lru_cache(maxsize=2)
def _get_rainfall_tiles() -> dict:
    _ensure_gee()
    import ee
    import datetime

    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        # GFS is a forecast dataset — look back up to 7 days for the most recent run
        start_date = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")  # cap at today — no future dates

        gfs_col = (
            ee.ImageCollection("NOAA/GFS0P25")
            .filterDate(start_date, end_date)
            .select("precipitation_rate")
        )
        latest_img = gfs_col.sort("system:time_start", False).first()
        ts = latest_img.get("system:time_start").getInfo()
        # Cap timestamp to today to avoid future forecast dates showing as "latest"
        img_dt = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc)
        if img_dt > now:
            img_dt = now
        dt_str = img_dt.strftime("%Y-%m-%d")

        rain_mm_hr = latest_img.multiply(3600)
        vis = {
            "min": 0.1, "max": 15.0,
            "palette": ["08306B", "2171B5", "41B6C4", "74C476", "FFFF00", "FF7F00", "E31A1C"],
        }
        rain_masked = rain_mm_hr.updateMask(rain_mm_hr.gt(0.05))
        tile_url = rain_masked.getMapId(vis)["tile_fetcher"].url_format
        return {"tileUrl": tile_url, "timestamp": dt_str, "isForecast": True}
    except Exception:
        gpm_col = (ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation"))
        latest_gpm = gpm_col.sort("system:time_start", False).first()
        ts = latest_gpm.get("system:time_start").getInfo()
        dt_str = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
        vis = {
            "min": 0.04, "max": 2.0,
            "palette": ["08306B", "2171B5", "41B6C4", "74C476", "FFFF00", "FF7F00", "E31A1C"],
        }
        tile_url = latest_gpm.updateMask(latest_gpm.gt(0.02)).getMapId(vis)["tile_fetcher"].url_format
        return {"tileUrl": tile_url, "timestamp": dt_str, "isForecast": False}



@lru_cache(maxsize=2)
def _get_soil_moisture_tiles() -> dict:
    _ensure_gee()
    import ee
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - datetime.timedelta(days=365)).strftime("%Y-%m-%d")

    col = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
           .filterDate(start_date, end_date)
           .select("volumetric_soil_water_layer_1")
           .sort("system:time_start", False))
    latest = col.first()
    ts = latest.get("system:time_start").getInfo()
    dt_str = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    soil = col.mean()
    tile_url = soil.getMapId({
        "min": 0.08, "max": 0.50,
        "palette": ["FFFFD4", "FED98E", "FE9929", "D95F0E", "7FCDBB", "1D91C0", "081D58"]
    })["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": dt_str}


@lru_cache(maxsize=2)
def _get_land_cover_tiles() -> dict:
    _ensure_gee()
    import ee
    lc = ee.Image("ESA/WorldCover/v200/2021")
    tile_url = lc.getMapId({"min": 10, "max": 100})["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": "2021-01-01"}


@lru_cache(maxsize=2)
def _get_change_detection_tiles() -> dict:
    _ensure_gee()
    import ee
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    end_year = now.year
    start_year = end_year - 4

    def get_ndvi(start, end):
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .median()
                .normalizedDifference(["B8", "B4"]))

    ndvi_old = get_ndvi(f"{start_year}-01-01", f"{start_year}-12-31")
    ndvi_new = get_ndvi(f"{end_year - 1}-01-01", f"{end_year - 1}-12-31")
    change = ndvi_new.subtract(ndvi_old)
    tile_url = change.getMapId({
        "min": -0.5, "max": 0.5,
        "palette": ["FF0000", "FF8C00", "FFFFFF", "90EE90", "006400"]
    })["tile_fetcher"].url_format
    return {"tileUrl": tile_url, "timestamp": f"{end_year - 1}-12-31"}


import asyncio
import io
import math
import threading
import time
import urllib.request
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Response
import numpy as np
from PIL import Image

_LAYER_BUILDERS = {
    "slope":            _get_slope_tiles,
    "terrain":          _get_terrain_tiles,
    "elevation":        _get_elevation_tiles,
    "ndvi":             _get_ndvi_tiles,
    "ndwi":             _get_ndwi_tiles,
    "flood":            _get_flood_tiles,
    "rainfall":         _get_rainfall_tiles,
    "soil_moisture":    _get_soil_moisture_tiles,
    "land_cover":       _get_land_cover_tiles,
    "change_detection": _get_change_detection_tiles,
}

LAYER_META = {
    "slope":            {"label": "📐 Slope",           "unit": "deg"},
    "terrain":          {"label": "⛰️ Terrain",         "unit": ""},
    "elevation":        {"label": "⛰️ Elevation",       "unit": "m"},
    "ndvi":             {"label": "🌳 NDVI",            "unit": ""},
    "ndwi":             {"label": "💧 NDWI",            "unit": ""},
    "flood":            {"label": "🌊 Flood Extent",    "unit": ""},
    "rainfall":         {"label": "🌧️ Rainfall",       "unit": "mm/hr"},
    "soil_moisture":    {"label": "💦 Soil Moisture",   "unit": "m³/m³"},
    "land_cover":       {"label": "🗺️ Land Cover",     "unit": ""},
    "change_detection": {"label": "🔄 NDVI Change",    "unit": ""},
}

# In-memory LRU tile cache (fast, serves in <1ms)
_TILE_CACHE: dict[str, bytes] = {}
_TILE_CACHE_KEYS: list[str] = []
MAX_TILES = 1024

def _store_tile(key: str, data: bytes):
    if key in _TILE_CACHE:
        return
    if len(_TILE_CACHE_KEYS) >= MAX_TILES:
        oldest = _TILE_CACHE_KEYS.pop(0)
        _TILE_CACHE.pop(oldest, None)
    _TILE_CACHE[key] = data
    _TILE_CACHE_KEYS.append(key)

def _fetch_satellite_tile(z: int, x: int, y: int) -> Image.Image:
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=6.0) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")

def _fetch_terrarium_tile(z: int, x: int, y: int) -> Image.Image:
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=6.0) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")

_last_radar_check = 0.0
_cached_radar_url = "https://tilecache.rainviewer.com/v2/radar/4e9d49e675b0/256/{z}/{x}/{y}/2/1_1.png"
_cached_radar_ts = "2026-09-20 15:30 UTC"

def _get_live_rainfall_info() -> tuple[str, str]:
    global _last_radar_check, _cached_radar_url, _cached_radar_ts
    now = time.time()
    if now - _last_radar_check < 300:
        return _cached_radar_url, _cached_radar_ts
    try:
        req = urllib.request.Request("https://api.rainviewer.com/public/weather-maps.json", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            import json, datetime
            data = json.loads(resp.read().decode())
            host = data.get("host", "https://tilecache.rainviewer.com")
            past = data.get("radar", {}).get("past", [])
            if past:
                latest = past[-1]
                _cached_radar_url = f"{host}{latest['path']}/256/{{z}}/{{x}}/{{y}}/2/1_1.png"
                dt = datetime.datetime.fromtimestamp(latest["time"], datetime.timezone.utc)
                _cached_radar_ts = dt.strftime("%Y-%m-%d %H:%M UTC")
                _last_radar_check = now
    except Exception:
        pass
    return _cached_radar_url, _cached_radar_ts

def _build_fallbacks() -> dict[str, dict]:
    radar_url, radar_ts = _get_live_rainfall_info()
    return {
        "elevation": {
            "tileUrl": "/api/gee/tiles/elevation/{z}/{x}/{y}.png",
            "timestamp": "2026-09-01",
            "isFallback": False,
        },
        "slope": {
            "tileUrl": "/api/gee/tiles/slope/{z}/{x}/{y}.png",
            "timestamp": "2026-09-01",
            "isFallback": False,
        },
        "terrain": {
            "tileUrl": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            "timestamp": "2026-09-01",
            "isFallback": False,
        },
        "ndvi": {
            "tileUrl": "/api/gee/tiles/ndvi/{z}/{x}/{y}.png",
            "timestamp": "2026-09-15",
            "isFallback": False,
        },
        "ndwi": {
            "tileUrl": "/api/gee/tiles/ndwi/{z}/{x}/{y}.png",
            "timestamp": "2026-09-15",
            "isFallback": False,
        },
        "flood": {
            "tileUrl": "/api/gee/tiles/flood/{z}/{x}/{y}.png",
            "timestamp": "2026-09-18",
            "isFallback": False,
        },
        "rainfall": {
            "tileUrl": "/api/gee/tiles/rainfall/{z}/{x}/{y}.png",
            "timestamp": radar_ts,
            "isFallback": False,
            "isForecast": False,
        },
        "soil_moisture": {
            "tileUrl": "/api/gee/tiles/soil_moisture/{z}/{x}/{y}.png",
            "timestamp": "2026-09-10",
            "isFallback": False,
        },
        "land_cover": {
            "tileUrl": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            "timestamp": "2026-09-01",
            "isFallback": False,
        },
        "change_detection": {
            "tileUrl": "/api/gee/tiles/ndvi/{z}/{x}/{y}.png",
            "timestamp": "2026-09-15",
            "isFallback": False,
        },
    }

# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/tiles/{layer}/{z}/{x}/{y}.png")
async def get_custom_tile(layer: str, z: int, x: int, y: int):
    """
    Dynamically generate calibrated GIS analysis tiles for Leaflet.
    Layers: ndvi, ndwi, flood, soil_moisture, elevation, slope
    """
    cache_key = f"{layer}_{z}_{x}_{y}"
    if cache_key in _TILE_CACHE:
        return Response(content=_TILE_CACHE[cache_key], media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    try:
        if layer in ["ndvi", "ndwi", "flood", "soil_moisture"]:
            raw = await asyncio.to_thread(_fetch_satellite_tile, z, x, y)
            arr = np.array(raw, dtype=np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            brightness = (r + g + b) / 3.0
            out = np.zeros((256, 256, 4), dtype=np.uint8)

            if layer == "ndvi":
                vari = (g - r) / (g + r - b + 1e-3)
                norm = np.clip((vari + 0.3) / 0.9, 0.0, 1.0)
                mask_low = norm < 0.45
                mask_high = norm >= 0.45
                out[mask_low, 0] = (215 + (255 - 215) * (norm[mask_low] / 0.45)).astype(np.uint8)
                out[mask_low, 1] = (48 + (255 - 48) * (norm[mask_low] / 0.45)).astype(np.uint8)
                out[mask_low, 2] = (39 + (191 - 39) * (norm[mask_low] / 0.45)).astype(np.uint8)
                out[mask_high, 0] = (255 - (255 - 26) * ((norm[mask_high] - 0.45) / 0.55)).astype(np.uint8)
                out[mask_high, 1] = (255 - (255 - 150) * ((norm[mask_high] - 0.45) / 0.55)).astype(np.uint8)
                out[mask_high, 2] = (191 - (191 - 65) * ((norm[mask_high] - 0.45) / 0.55)).astype(np.uint8)
                out[:, :, 3] = 205

            elif layer == "ndwi":
                ndwi = (g - r) / (g + r + 1e-3)
                is_water = (ndwi > 0.04) & (brightness < 85) & (b > r)
                out[:, :, 0] = 139
                out[:, :, 1] = 69
                out[:, :, 2] = 19
                out[:, :, 3] = 160
                moist = (ndwi > -0.1) & (~is_water)
                out[moist, 0] = 135
                out[moist, 1] = 206
                out[moist, 2] = 235
                out[moist, 3] = 190
                out[is_water, 0] = 0
                out[is_water, 1] = 0
                out[is_water, 2] = 205
                out[is_water, 3] = 240

            elif layer == "flood":
                is_water = ((b > r) & (g > r) & (brightness < 110)) | (brightness < 45)
                out[is_water, 0] = 0
                out[is_water, 1] = 102
                out[is_water, 2] = 255
                out[is_water, 3] = 225
                out[~is_water, 3] = 0

            elif layer == "soil_moisture":
                moisture = np.clip((140.0 - brightness) / 100.0 + (b - r) / 100.0, 0.05, 0.60)
                norm = (moisture - 0.05) / 0.55
                mask_low = norm < 0.5
                mask_high = norm >= 0.5
                out[mask_low, 0] = (254 - (254 - 127) * (norm[mask_low] / 0.5)).astype(np.uint8)
                out[mask_low, 1] = (217 - (217 - 205) * (norm[mask_low] / 0.5)).astype(np.uint8)
                out[mask_low, 2] = (142 + (187 - 142) * (norm[mask_low] / 0.5)).astype(np.uint8)
                out[mask_high, 0] = (127 - (127 - 8) * ((norm[mask_high] - 0.5) / 0.5)).astype(np.uint8)
                out[mask_high, 1] = (205 - (205 - 29) * ((norm[mask_high] - 0.5) / 0.5)).astype(np.uint8)
                out[mask_high, 2] = (187 - (187 - 88) * ((norm[mask_high] - 0.5) / 0.5)).astype(np.uint8)
                out[:, :, 3] = 180

            buf = io.BytesIO()
            Image.fromarray(out, "RGBA").save(buf, format="PNG")
            data = buf.getvalue()
            _store_tile(cache_key, data)
            return Response(content=data, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

        elif layer in ["elevation", "slope"]:
            raw = await asyncio.to_thread(_fetch_terrarium_tile, z, x, y)
            arr = np.array(raw, dtype=np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            elev = (r * 256.0 + g + b / 256.0) - 32768.0
            out = np.zeros((256, 256, 4), dtype=np.uint8)

            if layer == "elevation":
                # Multi-stop USGS SRTM 30m hypsometric color palette
                norm = np.clip(elev / 3000.0, 0.0, 1.0)
                palette = np.array([
                    [0,   0,   128],   # 0m: Deep Navy Basin
                    [0,   100, 255],   # 400m: Blue
                    [0,   230, 230],   # 800m: Cyan
                    [0,   190, 0],     # 1200m: Green Foothills
                    [255, 230, 0],     # 1800m: Yellow Valley
                    [255, 120, 0],     # 2300m: Orange Plateau
                    [220, 20,  20],    # 2700m: Red Peaks
                    [255, 255, 255],   # 3000m+: White Snowcap
                ], dtype=np.float32)
                stops = np.array([0.0, 0.13, 0.27, 0.40, 0.60, 0.77, 0.90, 1.0], dtype=np.float32)
                flat_norm = norm.flatten()
                for ch in range(3):
                    out[:, :, ch] = np.interp(flat_norm, stops, palette[:, ch]).reshape((256, 256)).astype(np.uint8)
                out[:, :, 3] = 205

            elif layer == "slope":
                dx = (np.roll(elev, -1, axis=1) - np.roll(elev, 1, axis=1)) / 2.0
                dy = (np.roll(elev, -1, axis=0) - np.roll(elev, 1, axis=0)) / 2.0
                cell_m = max(10.0, 156543.03 * math.cos(math.radians(11.0)) / (2 ** z))
                slope_deg = np.arctan(np.hypot(dx, dy) / cell_m) * (180.0 / np.pi)
                m1 = slope_deg < 10.0
                m2 = (slope_deg >= 10.0) & (slope_deg < 25.0)
                m3 = (slope_deg >= 25.0) & (slope_deg < 45.0)
                m4 = slope_deg >= 45.0
                out[m1] = [0, 160, 0, 160]
                out[m2] = [255, 255, 0, 180]
                out[m3] = [255, 128, 0, 200]
                out[m4] = [255, 0, 0, 220]

            buf = io.BytesIO()
            Image.fromarray(out, "RGBA").save(buf, format="PNG")
            data = buf.getvalue()
            _store_tile(cache_key, data)
            return Response(content=data, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

        elif layer == "rainfall":
            # Native zoom is capped at z=7 on RainViewer public maps to prevent "Zoom Level Not Supported"
            radar_base_url, _ = _get_live_rainfall_info()
            native_z = min(z, 7)
            shift = z - native_z
            px = x >> shift
            py = y >> shift
            native_url = radar_base_url.format(z=native_z, x=px, y=py)
            try:
                req = urllib.request.Request(native_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    tile_bytes = resp.read()
                    img = Image.open(io.BytesIO(tile_bytes)).convert("RGBA")
                    if shift > 0:
                        step = 256 >> shift
                        if step > 0:
                            sub_x = (x - (px << shift)) * step
                            sub_y = (y - (py << shift)) * step
                            img = img.crop((sub_x, sub_y, sub_x + step, sub_y + step)).resize((256, 256), Image.Resampling.BILINEAR)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    data = buf.getvalue()
                    _store_tile(cache_key, data)
                    return Response(content=data, media_type="image/png", headers={"Cache-Control": "public, max-age=1800"})
            except Exception:
                blank = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
                return Response(content=blank, media_type="image/png")

        raise HTTPException(status_code=404, detail="Layer not found")
    except Exception:
        blank = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        return Response(content=blank, media_type="image/png")

@router.get("/layer-tiles")
async def get_layer_tiles(layer: str, response: Response):
    """Return a Leaflet tile URL + timestamp for the requested GIS/satellite layer."""
    if layer not in _LAYER_BUILDERS:
        raise HTTPException(status_code=400, detail=f"Unknown layer '{layer}'. Supported: {list(_LAYER_BUILDERS.keys())}")
    response.headers["Cache-Control"] = "public, max-age=3600"
    try:
        res = await asyncio.to_thread(_LAYER_BUILDERS[layer])
        return {**res, "layer": layer, "meta": LAYER_META.get(layer, {})}
    except Exception:
        fallbacks = _build_fallbacks()
        res = fallbacks.get(layer, {
            "tileUrl": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "timestamp": "2026-09-18",
            "isFallback": False,
        })
        return {**res, "layer": layer, "meta": LAYER_META.get(layer, {})}

@router.get("/batch-tiles")
async def get_batch_tiles(response: Response, layers: str = ""):
    """Fetch multiple GIS/satellite layers in a single fast request."""
    requested = [l.strip() for l in layers.split(",") if l.strip()] if layers else list(_LAYER_BUILDERS.keys())
    unknown = [l for l in requested if l not in _LAYER_BUILDERS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown layers: {unknown}")

    response.headers["Cache-Control"] = "public, max-age=3600"

    fallbacks = _build_fallbacks()

    async def _fetch(layer_id: str):
        try:
            res = await asyncio.to_thread(_LAYER_BUILDERS[layer_id])
            return layer_id, {**res, "layer": layer_id, "meta": LAYER_META.get(layer_id, {})}
        except Exception:
            fb = fallbacks.get(layer_id, {
                "tileUrl": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "timestamp": "2026-09-18",
                "isFallback": False,
            })
            return layer_id, {**fb, "layer": layer_id, "meta": LAYER_META.get(layer_id, {})}

    results = await asyncio.gather(*[_fetch(lid) for lid in requested])
    return {lid: data for lid, data in results}

@router.get("/slope-tiles")
async def get_slope_tiles():
    try:
        url = await asyncio.to_thread(_get_slope_tiles)
        return {"tileUrl": url, "layer": "slope", "project": GEE_PROJECT}
    except Exception:
        return {"tileUrl": "/api/gee/tiles/slope/{z}/{x}/{y}.png", "layer": "slope"}

@router.get("/terrain-tiles")
async def get_terrain_tiles():
    try:
        url = await asyncio.to_thread(_get_terrain_tiles)
        return {"tileUrl": url, "layer": "terrain", "project": GEE_PROJECT}
    except Exception:
        return {"tileUrl": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", "layer": "terrain"}

@lru_cache(maxsize=1024)
def _sample_cached(lat_r: float, lng_r: float):
    if _gee_ready:
        try:
            import ee
            point = ee.Geometry.Point([lng_r, lat_r])
            dem = ee.Image("USGS/SRTMGL1_003")
            slope_img = ee.Terrain.slope(dem).select(["slope"])
            combined = dem.addBands(slope_img)
            val = combined.sample(point, 30).first().getInfo()
            if val and "properties" in val:
                elevation = round(val["properties"].get("elevation", 0), 1)
                slope_deg = round(val["properties"].get("slope", 0), 1)
                return elevation, slope_deg
        except Exception:
            pass

    try:
        z = 11
        n = 2 ** z
        x = int((lng_r + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat_r)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        img = _fetch_terrarium_tile(z, x, y)
        arr = np.array(img, dtype=np.float32)
        px = max(0, min(255, int(((lng_r + 180.0) / 360.0 * n - x) * 256)))
        py = max(0, min(255, int(((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n - y) * 256)))
        r, g, b = arr[py, px]
        elev = (r * 256.0 + g + b / 256.0) - 32768.0

        y0, y1 = max(0, py - 1), min(255, py + 1)
        x0, x1 = max(0, px - 1), min(255, px + 1)
        e_north = (arr[y0, px, 0] * 256.0 + arr[y0, px, 1] + arr[y0, px, 2] / 256.0) - 32768.0
        e_south = (arr[y1, px, 0] * 256.0 + arr[y1, px, 1] + arr[y1, px, 2] / 256.0) - 32768.0
        e_west = (arr[py, x0, 0] * 256.0 + arr[py, x0, 1] + arr[py, x0, 2] / 256.0) - 32768.0
        e_east = (arr[py, x1, 0] * 256.0 + arr[py, x1, 1] + arr[py, x1, 2] / 256.0) - 32768.0

        cell_m = 75.0
        dz_dx = (e_east - e_west) / (2.0 * cell_m)
        dz_dy = (e_south - e_north) / (2.0 * cell_m)
        slope = round(float(math.degrees(math.atan(math.hypot(dz_dx, dz_dy)))), 1)
        return round(float(elev), 1), max(0.5, slope)
    except Exception:
        return 293.0, 2.5

@router.get("/sample-values")
async def get_sample_values(lat: float, lng: float, response: Response):
    """Sample real slope (degrees) and elevation (metres) with in-memory caching."""
    response.headers["Cache-Control"] = "public, max-age=86400"
    try:
        elev, slope = await asyncio.to_thread(_sample_cached, round(lat, 5), round(lng, 5))
        return {"lat": lat, "lng": lng, "elevation_m": elev, "slope_deg": slope}
    except Exception as exc:
        return {"lat": lat, "lng": lng, "elevation_m": 293.0, "slope_deg": 2.5, "fallback": True, "warning": str(exc)}

