"""
Weather Router for NEXGI GIS and Disaster Intelligence
Provides live real-world weather and multi-day forecasts via OpenWeatherMap.
"""
import os
from typing import Optional
from fastapi import APIRouter, Query
from services.openweather_service import openweather_service, OPENWEATHER_API_KEY

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/current")
async def get_current_weather(
    lat: float = Query(default=10.6608, description="Latitude"),
    lon: float = Query(default=77.0048, description="Longitude"),
):
    """
    Returns current weather data for the specified coordinates using OpenWeatherMap.
    """
    return await openweather_service.get_current_weather(lat, lon)


@router.get("/forecast")
async def get_weather_forecast(
    lat: float = Query(default=10.6608, description="Latitude"),
    lon: float = Query(default=77.0048, description="Longitude"),
    days: int = Query(default=7, ge=1, le=7, description="Forecast days"),
):
    """
    Returns multi-day weather forecast (1d to 7d) for Digital Twin and GIS overlays.
    """
    return await openweather_service.get_forecast(lat, lon, days)


@router.get("/status")
async def get_weather_status():
    """
    Returns the OpenWeatherMap configuration and connection status.
    """
    key = os.environ.get("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY)
    masked_key = f"{key[:4]}...{key[-4:]}" if key and len(key) >= 8 else "not set"
    return {
        "provider": "OpenWeatherMap",
        "api_key_configured": bool(key),
        "api_key_masked": masked_key,
        "endpoint": "https://api.openweathermap.org/data/2.5",
        "status": "active",
    }
