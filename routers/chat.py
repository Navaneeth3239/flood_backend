"""
Disaster Intelligence Chat Router
Streams AI responses directly from external LLM API (Qwen2.5-VL) in chunks
with spatial GIS context (paths, location name, coordinates, sensors).
"""
import asyncio
import json
import logging
import os
from typing import List, Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.location_service import bbox_from_radius, bbox_from_polygon
from services.osm_road_service import OSMRoadService
from services.osm_river_service import OSMRiverService
from services.openweather_service import openweather_service
from lib.db import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

def get_qwen_api_url() -> str:
    raw = os.environ.get("LLM_API_URL", "https://qwen.blk2np.qzz.io").strip()
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    return raw.rstrip("/")

QWEN_API_URL = get_qwen_api_url()
road_service = OSMRoadService()
river_service = OSMRiverService()


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    prompt: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    area_name: Optional[str] = None
    polygon: Optional[List[List[float]]] = None
    paths: Optional[List[str]] = None
    radius_km: Optional[float] = 5.0
    history: Optional[List[Dict[str, str]]] = None
    forecast_hour: Optional[int] = Field(default=0, ge=0, le=11)
    rainfall_intensity: Optional[float] = Field(default=None, ge=0, le=500)
    wind_speed: Optional[float] = Field(default=None, ge=0, le=300)
    rain_active: bool = False
    water_sim_active: bool = False
    buildings: Optional[List[Dict[str, Any]]] = None
    risk_zones: Optional[List[Dict[str, Any]]] = None
    sensors: Optional[List[Dict[str, Any]]] = None
    mesh_nodes: Optional[List[Dict[str, Any]]] = None

    def get_query(self) -> str:
        return (self.query or self.message or self.prompt or "").strip()


class RiskAnalyzeRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    area_name: Optional[str] = None
    radius_km: Optional[float] = 5.0


async def _extract_spatial_context(
    lat: Optional[float],
    lng: Optional[float],
    area_name: Optional[str],
    polygon: Optional[List[List[float]]],
    provided_paths: Optional[List[str]],
    radius_km: float = 5.0,
) -> Dict[str, Any]:
    """Resolves roads/paths, waterways, and sensor telemetry for the model context."""
    effective_lat = lat if lat is not None else 10.6608
    effective_lng = lng if lng is not None else 77.0048
    effective_name = area_name or f"Zone ({effective_lat:.4f}°N, {effective_lng:.4f}°E)"

    if polygon and len(polygon) >= 3:
        bbox = bbox_from_polygon(polygon)
    else:
        bbox = bbox_from_radius(effective_lat, effective_lng, radius_km)

    north, south, east, west = bbox["north"], bbox["south"], bbox["east"], bbox["west"]

    async def load_paths() -> List[str]:
        if provided_paths:
            return provided_paths[:10]
        try:
            _, road_geojson = await asyncio.wait_for(
                road_service.get_road_network(north, south, east, west, polygon=polygon),
                timeout=0.25,
            )
            named_roads = set()
            for feat in road_geojson.get("features", []):
                props = feat.get("properties", {})
                name = props.get("name")
                rtype = props.get("road_type", "road")
                length = props.get("length_m")
                if name:
                    named_roads.add(f"{name} ({rtype}{f', {int(length)}m' if length else ''})")
                elif rtype in ("primary", "secondary", "tertiary", "trunk"):
                    named_roads.add(f"{rtype.capitalize()} Route")
            return sorted(named_roads)[:10]
        except Exception:
            return []

    async def load_rivers() -> List[str]:
        try:
            _, river_geojson = await asyncio.wait_for(
                river_service.get_river_network(north, south, east, west, polygon=polygon),
                timeout=0.25,
            )
            named_rivers = set()
            for feat in river_geojson.get("features", []):
                props = feat.get("properties", {})
                name = props.get("name")
                wtype = props.get("waterway_type", "waterway")
                if name:
                    named_rivers.add(f"{name} ({wtype})")
                elif wtype in ("river", "stream", "canal"):
                    named_rivers.add(f"Local {wtype.capitalize()}")
            return sorted(named_rivers)[:6]
        except Exception:
            return []

    async def load_telemetry() -> str:
        try:
            zones_list = await asyncio.wait_for(db.zones.find({}, {"_id": 0}).to_list(100), timeout=0.1)
            matching_zone = next(
                (
                    zone for zone in zones_list
                    if zone.get("name") == effective_name
                    or (lat is not None and abs((zone.get("latitude") or 0) - lat) < 0.1)
                ),
                None,
            )
            if matching_zone:
                return (
                    f"Rainfall: {float(matching_zone.get('rainfall_mm') or 0):.1f} mm/h; "
                    f"Water Level: {float(matching_zone.get('water_level_m') or 0):.2f} m; "
                    f"Soil Moisture: {float(matching_zone.get('soil_moisture_pct') or 0):.0f}%; "
                    f"Hazard Type: {matching_zone.get('hazard_type', 'flood')}"
                )
        except Exception:
            pass
    async def load_weather() -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(
                openweather_service.get_current_weather(effective_lat, effective_lng),
                timeout=0.35,
            )
        except Exception:
            return {}

    extracted_paths, extracted_rivers, telemetry_summary, live_weather = await asyncio.gather(
        load_paths(), load_rivers(), load_telemetry(), load_weather()
    )

    weather_desc = ""
    if live_weather:
        weather_desc = (
            f"Weather: {live_weather.get('condition', 'Overcast')}, {live_weather.get('temperature_c', 26.0)}°C, "
            f"Rain Rate: {live_weather.get('rainfall_rate_mmh', 0.0)} mm/h, Wind: {live_weather.get('wind_speed_kmh', 12.0)} km/h {live_weather.get('wind_direction', '')} "
            f"({live_weather.get('data_source', 'OpenWeatherMap')})"
        )

    full_telemetry = f"{telemetry_summary} | {weather_desc}" if weather_desc else telemetry_summary

    return {
        "area_name": effective_name,
        "latitude": effective_lat,
        "longitude": effective_lng,
        "paths": extracted_paths or ["Main Access Road", "High Ridge Evacuation Path"],
        "rivers": extracted_rivers or ["Local River Channel", "Valley Stream"],
        "telemetry": full_telemetry,
        "weather": live_weather,
    }


def _build_system_prompt(spatial: Dict[str, Any], req: ChatRequest) -> str:
    # Limit to top 5 paths and top 3 waterways to guarantee < 1s prompt prefill on Apple Silicon MPS
    paths_list = "\n".join(f"  • {p}" for p in spatial["paths"][:5])
    rivers_list = "\n".join(f"  • {r}" for r in spatial["rivers"][:3])

    return f"""You are the AI Disaster Intelligence & Early Warning Specialist for {spatial['area_name']} ({spatial['latitude']:.4f}°N, {spatial['longitude']:.4f}°E).
Monitored Geo-Context:
- Primary Evacuation Routes:
{paths_list}
- Key Drainage & River Channels:
{rivers_list}
- Real-time Telemetry & State:
  Rain: {'active' if req.rain_active else 'inactive'} ({req.rainfall_intensity or 0} mm/h), Water Simulation: {'active' if req.water_sim_active else 'inactive'}
  {spatial['telemetry']}

Directives:
1. Provide a direct, actionable emergency plan, hazard analysis, or evacuation guidance for {spatial['area_name']}.
2. Reference the above roads and waterways where relevant.
3. Use clear markdown headings and bullet points.
"""


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    Streams AI responses directly from API in chunks token-by-token.
    """
    spatial = await _extract_spatial_context(
        lat=req.latitude,
        lng=req.longitude,
        area_name=req.area_name,
        polygon=req.polygon,
        provided_paths=req.paths,
        radius_km=req.radius_km or 5.0,
    )

    query_text = req.get_query() or "What is the flood risk and evacuation plan?"
    system_prompt = _build_system_prompt(spatial, req)

    user_prompt = query_text
    if req.history and len(req.history) > 0:
        history_snippet = "\n".join(
            f"{h.get('role', 'user').capitalize()}: {h.get('text', '')[:300]}"
            for h in req.history[-3:]
            if h.get('text')
        )
        user_prompt = f"Previous conversation context:\n{history_snippet}\n\nUser Question: {user_prompt}"

    async def event_generator():
        # Flush SSE connection headers immediately to client/browser
        yield ": open\n\n"

        # Generous timeout for comprehensive planning responses (3 minutes)
        client_timeout = httpx.Timeout(180.0, connect=15.0, read=180.0, write=30.0)
        form_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "max_tokens": "600",
        }
        current_api_url = get_qwen_api_url()
        plan_url = f"{current_api_url}/plan"
        text_url = f"{current_api_url}/text"

        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                streamed = False
                try:
                    # Stream tokens in real-time as chunks from /plan (https://qwen.blk2np.qzz.io/docs#/default/plan_plan_post)
                    async with client.stream("POST", plan_url, data=form_data) as response:
                        if response.status_code == 200:
                            async for chunk in response.aiter_text():
                                if chunk:
                                    streamed = True
                                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                        else:
                            logger.warning(f"/plan returned {response.status_code}, trying /text fallback")
                except Exception as plan_err:
                    logger.warning(f"Failed streaming /plan: {plan_err}, trying /text")

                if not streamed:
                    async with client.stream("POST", text_url, data=form_data) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_text():
                            if chunk:
                                yield f"data: {json.dumps({'token': chunk})}\n\n"
        except Exception as exc:
            logger.error(f"Error streaming directly from API: {exc}")
            err_payload = json.dumps({"token": f"\n\n*[API Streaming Error: {str(exc)}]*"})
            yield f"data: {err_payload}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post("/risk/analyze")
async def chat_risk_analyze(req: RiskAnalyzeRequest):
    """
    Returns quick risk assessment metadata for the chat badge and UI chips.
    """
    spatial = await _extract_spatial_context(
        lat=req.latitude,
        lng=req.longitude,
        area_name=req.area_name,
        polygon=None,
        provided_paths=None,
        radius_km=req.radius_km or 5.0,
    )

    risk_level = "MODERATE"
    confidence = 88
    try:
        zone = await db.zones.find_one({"name": spatial["area_name"]}, {"_id": 0})
        if zone:
            rf = float(zone.get("rainfall_mm") or 0)
            if rf > 60:
                risk_level = "CRITICAL"
                confidence = 94
            elif rf > 30:
                risk_level = "HIGH"
                confidence = 91
            elif rf > 10:
                risk_level = "MODERATE"
                confidence = 86
            else:
                risk_level = "LOW"
                confidence = 90
    except Exception:
        pass

    actions = [
        f"Monitor water levels along {spatial['rivers'][0] if spatial['rivers'] else 'local waterways'}",
        f"Keep {spatial['paths'][0] if spatial['paths'] else 'main access path'} clear for emergency vehicles",
        "Maintain LoRaWAN field telemetry uplinks at 5-minute sampling intervals",
    ]

    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "area_name": spatial["area_name"],
        "latitude": spatial["latitude"],
        "longitude": spatial["longitude"],
        "paths": spatial["paths"],
        "recommended_actions": actions,
    }
