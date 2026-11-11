from __future__ import annotations

"""
Satellite imagery and Earth Engine analysis endpoints for disaster monitoring.
Integrates Sentinel-1 SAR, Sentinel-2 optical, DEM, flood and landslide detection.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime, date
import os
from lib.auth import current_user
from models.schemas import User

router = APIRouter(prefix="/satellite", tags=["satellite"])

# Earth Engine Project ID from environment
EE_PROJECT_ID = os.getenv("EE_PROJECT_ID", "formal-purpose-466115-i8")

_ee_initialized = False
ee = None

def _ensure_ee():
    """Load Earth Engine only for satellite requests, never during API startup."""
    global _ee_initialized, ee
    if ee is None:
        try:
            import ee as earth_engine
            ee = earth_engine
        except Exception as exc:
            raise HTTPException(503, f"Earth Engine client is unavailable: {exc}") from exc
    if not _ee_initialized:
        try:
            ee.Initialize(project=EE_PROJECT_ID)
            _ee_initialized = True
        except Exception as e:
            raise HTTPException(503, f"Earth Engine could not initialize: {e}") from e


# ==========================================
# REQUEST/RESPONSE MODELS
# ==========================================

class AOIRequest(BaseModel):
    """Area of Interest definition"""
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    area_size_km: Optional[float] = Field(5.0, gt=0, le=100, description="Radius in km or size")
    polygon: Optional[List[List[float]]] = Field(None, description="Custom polygon coordinates [[lat,lng],...]")


class SatelliteLayerRequest(BaseModel):
    """Request for satellite imagery layer"""
    aoi: AOIRequest
    layer_type: Literal[
        "sentinel1_sar",
        "sentinel2_true_color",
        "dem_terrain",
        "flood_detection",
        "landslide_risk",
        "change_detection",
        "ndvi_vegetation"
    ]
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD format")
    polarization: Optional[Literal["VV", "VH", "VV+VH"]] = Field("VV", description="SAR polarization")
    orbit: Optional[Literal["ASCENDING", "DESCENDING"]] = Field("DESCENDING")


class BeforeAfterRequest(BaseModel):
    """Request for before/after comparison"""
    aoi: AOIRequest
    monitoring_type: Literal["flood", "landslide", "combined"]
    before_date: str = Field(..., description="YYYY-MM-DD")
    after_date: str = Field(..., description="YYYY-MM-DD")


class FloodAnalysisRequest(BaseModel):
    """Flood monitoring analysis request"""
    aoi: AOIRequest
    baseline_start: str = Field(..., description="Baseline period start YYYY-MM-DD")
    baseline_end: str = Field(..., description="Baseline period end YYYY-MM-DD")
    event_start: str = Field(..., description="Event period start YYYY-MM-DD")
    event_end: str = Field(..., description="Event period end YYYY-MM-DD")
    threshold: Optional[float] = Field(-18, description="dB threshold for water detection")


class LandslideRiskRequest(BaseModel):
    """Landslide risk assessment request"""
    aoi: AOIRequest
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    rainfall_threshold: Optional[float] = Field(50.0, description="mm threshold")


class TileResponse(BaseModel):
    """Earth Engine tile layer response"""
    tile_url: str
    map_id: str
    token: str
    acquisition_date: Optional[str] = None
    source: str
    bounds: List[List[float]]
    metadata: dict = {}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_aoi_geometry(aoi: AOIRequest) -> ee.Geometry:
    """Create Earth Engine geometry from AOI request"""
    _ensure_ee()
    if aoi.polygon and len(aoi.polygon) >= 3:
        # Custom polygon
        coords = [[lng, lat] for lat, lng in aoi.polygon]
        return ee.Geometry.Polygon([coords])
    else:
        # Rectangle from center point and size
        size_degrees = aoi.area_size_km / 111.0  # rough km to degrees
        return ee.Geometry.Rectangle([
            aoi.lng - size_degrees/2,
            aoi.lat - size_degrees/2,
            aoi.lng + size_degrees/2,
            aoi.lat + size_degrees/2
        ])


def get_sentinel1_image(aoi: ee.Geometry, start_date: str, end_date: str, 
                        polarization: str = "VV", orbit: str = "DESCENDING") -> ee.Image:
    """Get latest Sentinel-1 SAR image"""
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarization))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
    )
    
    return collection.sort("system:time_start", False).first().clip(aoi)


def get_sentinel2_image(aoi: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    """Get latest cloud-free Sentinel-2 optical image"""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    
    return collection.sort("system:time_start", False).first().clip(aoi)


def detect_flood_extent(before_img: ee.Image, after_img: ee.Image, threshold: float = -18) -> ee.Image:
    """Detect flood extent using SAR backscatter change"""
    # Use VV or VH polarization
    before_vv = before_img.select("VV")
    after_vv = after_img.select("VV")
    
    # Calculate difference
    diff = after_vv.subtract(before_vv)
    
    # Water appears dark in SAR, so after < before
    # Also check absolute low backscatter in after image
    flood_mask = after_vv.lt(threshold).And(diff.lt(-3))
    
    return flood_mask.selfMask()


def calculate_landslide_risk(aoi: ee.Geometry, dem: ee.Image, ndvi_change: ee.Image, 
                              slope: ee.Image) -> ee.Image:
    """Calculate landslide risk based on slope, vegetation, and terrain"""
    
    # Normalize slope (0-90 degrees to 0-1)
    slope_norm = slope.divide(90)
    
    # High slope = high risk
    slope_risk = slope_norm.multiply(0.5)
    
    # Negative NDVI change (vegetation loss) = high risk
    ndvi_risk = ndvi_change.multiply(-1).multiply(0.3)
    
    # Elevation variability = moderate risk factor
    elevation_risk = dem.divide(5000).multiply(0.2)
    
    # Combined risk score
    risk = slope_risk.add(ndvi_risk).add(elevation_risk).clamp(0, 1)
    
    return risk


def classify_risk_zones(risk: ee.Image) -> ee.Image:
    """Classify risk into Low/Moderate/High/Critical zones"""
    # Risk thresholds: 0-0.25 = Low, 0.25-0.5 = Moderate, 0.5-0.75 = High, 0.75-1 = Critical
    classified = (
        risk.where(risk.lt(0.25), 1)  # Low = 1
        .where(risk.gte(0.25).And(risk.lt(0.5)), 2)  # Moderate = 2
        .where(risk.gte(0.5).And(risk.lt(0.75)), 3)  # High = 3
        .where(risk.gte(0.75), 4)  # Critical = 4
    )
    return classified


# ==========================================
# API ENDPOINTS
# ==========================================

@router.post("/layer", response_model=TileResponse)
async def get_satellite_layer(
    req: SatelliteLayerRequest,
    user: dict = Depends(current_user)
):
    """
    Get satellite imagery layer as Earth Engine tile URL.
    Supports Sentinel-1 SAR, Sentinel-2 optical, DEM, and analysis layers.
    """
    try:
        aoi = create_aoi_geometry(req.aoi)
        
        # Default date range (last 30 days if not specified)
        end_date = req.end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = req.start_date or (datetime.now().replace(day=1)).strftime("%Y-%m-%d")
        
        metadata = {
            "layer_type": req.layer_type,
            "date_range": f"{start_date} to {end_date}"
        }
        
        # Get appropriate imagery based on layer type
        if req.layer_type == "sentinel1_sar":
            image = get_sentinel1_image(aoi, start_date, end_date, req.polarization, req.orbit)
            vis_params = {"min": -25, "max": 0, "palette": ["black", "white"]}
            metadata["polarization"] = req.polarization
            metadata["orbit"] = req.orbit
            
        elif req.layer_type == "sentinel2_true_color":
            image = get_sentinel2_image(aoi, start_date, end_date)
            image = image.select(["B4", "B3", "B2"])  # RGB
            vis_params = {"min": 0, "max": 3000, "gamma": 1.4}
            
        elif req.layer_type == "dem_terrain":
            dem = ee.Image("USGS/SRTMGL1_003")
            image = dem.clip(aoi)
            vis_params = {"min": 0, "max": 3000, "palette": ["blue", "green", "yellow", "red"]}
            
        elif req.layer_type == "ndvi_vegetation":
            s2 = get_sentinel2_image(aoi, start_date, end_date)
            nir = s2.select("B8")
            red = s2.select("B4")
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            image = ndvi.clip(aoi)
            vis_params = {"min": -0.2, "max": 0.8, "palette": ["brown", "yellow", "green", "darkgreen"]}
            
        else:
            raise HTTPException(400, f"Layer type {req.layer_type} not yet implemented for standalone view")
        
        # Get tile URL
        map_id_dict = image.getMapId(vis_params)
        
        # Get acquisition date if available
        acquisition_date = None
        try:
            if req.layer_type in ["sentinel1_sar", "sentinel2_true_color"]:
                timestamp = image.get("system:time_start").getInfo()
                if timestamp:
                    acquisition_date = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
        except:
            pass
        
        bounds = aoi.bounds().getInfo()["coordinates"][0]
        
        return TileResponse(
            tile_url=map_id_dict["tile_fetcher"].url_format,
            map_id=map_id_dict["mapid"],
            token=map_id_dict["token"],
            acquisition_date=acquisition_date,
            source=f"Earth Engine - {req.layer_type}",
            bounds=bounds,
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(500, f"Earth Engine error: {str(e)}")


@router.post("/flood-analysis", response_model=dict)
async def analyze_flood(
    req: FloodAnalysisRequest,
    user: dict = Depends(current_user)
):
    """
    Perform flood detection analysis using Sentinel-1 SAR before/after comparison.
    Returns flood extent map and statistics.
    """
    try:
        aoi = create_aoi_geometry(req.aoi)
        
        # Get baseline and event images
        baseline = get_sentinel1_image(aoi, req.baseline_start, req.baseline_end)
        event = get_sentinel1_image(aoi, req.event_start, req.event_end)
        
        # Detect flood extent
        flood_mask = detect_flood_extent(baseline, event, req.threshold)
        
        # Calculate statistics
        flood_area_pixels = flood_mask.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9
        )
        
        # Get tile URLs for visualization
        baseline_map = baseline.select("VV").getMapId({"min": -25, "max": 0})
        event_map = event.select("VV").getMapId({"min": -25, "max": 0})
        flood_map = flood_mask.getMapId({"palette": ["blue"]})
        
        # Get dates
        baseline_date = datetime.fromtimestamp(
            baseline.get("system:time_start").getInfo() / 1000
        ).strftime("%Y-%m-%d")
        event_date = datetime.fromtimestamp(
            event.get("system:time_start").getInfo() / 1000
        ).strftime("%Y-%m-%d")
        
        return {
            "status": "success",
            "baseline_date": baseline_date,
            "event_date": event_date,
            "flood_detected": True,
            "baseline_tile_url": baseline_map["tile_fetcher"].url_format,
            "event_tile_url": event_map["tile_fetcher"].url_format,
            "flood_extent_tile_url": flood_map["tile_fetcher"].url_format,
            "flood_area_pixels": flood_area_pixels.getInfo(),
            "threshold_db": req.threshold,
            "bounds": aoi.bounds().getInfo()["coordinates"][0]
        }
        
    except Exception as e:
        raise HTTPException(500, f"Flood analysis error: {str(e)}")


@router.post("/landslide-risk", response_model=dict)
async def assess_landslide_risk(
    req: LandslideRiskRequest,
    user: dict = Depends(current_user)
):
    """
    Assess landslide risk using DEM slope, vegetation change, and terrain analysis.
    Returns risk classification map and statistics.
    """
    try:
        aoi = create_aoi_geometry(req.aoi)
        
        # Get DEM and calculate slope
        dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
        slope = ee.Terrain.slope(dem)
        
        # Get NDVI change (vegetation loss indicator)
        s2_before = get_sentinel2_image(
            aoi, 
            (datetime.strptime(req.start_date, "%Y-%m-%d").replace(month=1)).strftime("%Y-%m-%d"),
            req.start_date
        )
        s2_after = get_sentinel2_image(aoi, req.start_date, req.end_date)
        
        def calc_ndvi(img):
            nir = img.select("B8")
            red = img.select("B4")
            return nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        
        ndvi_before = calc_ndvi(s2_before)
        ndvi_after = calc_ndvi(s2_after)
        ndvi_change = ndvi_after.subtract(ndvi_before)
        
        # Calculate risk
        risk = calculate_landslide_risk(aoi, dem, ndvi_change, slope)
        risk_classified = classify_risk_zones(risk)
        
        # Get tile URLs
        dem_map = dem.getMapId({"min": 0, "max": 3000, "palette": ["blue", "green", "yellow", "red"]})
        slope_map = slope.getMapId({"min": 0, "max": 45, "palette": ["green", "yellow", "orange", "red"]})
        risk_map = risk_classified.getMapId({
            "min": 1,
            "max": 4,
            "palette": ["green", "yellow", "orange", "red"]
        })
        
        # Calculate area statistics for each risk class
        risk_stats = risk_classified.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        
        return {
            "status": "success",
            "start_date": req.start_date,
            "end_date": req.end_date,
            "dem_tile_url": dem_map["tile_fetcher"].url_format,
            "slope_tile_url": slope_map["tile_fetcher"].url_format,
            "risk_tile_url": risk_map["tile_fetcher"].url_format,
            "risk_classification": {
                "1_low": "green",
                "2_moderate": "yellow",
                "3_high": "orange",
                "4_critical": "red"
            },
            "risk_statistics": risk_stats.getInfo(),
            "bounds": aoi.bounds().getInfo()["coordinates"][0]
        }
        
    except Exception as e:
        raise HTTPException(500, f"Landslide risk analysis error: {str(e)}")


@router.post("/before-after", response_model=dict)
async def before_after_comparison(
    req: BeforeAfterRequest,
    user: dict = Depends(current_user)
):
    """
    Generate before/after comparison for flood or landslide monitoring.
    Returns both imagery layers and change detection overlay.
    """
    try:
        aoi = create_aoi_geometry(req.aoi)
        
        if req.monitoring_type in ["flood", "combined"]:
            # SAR-based flood detection
            before_sar = get_sentinel1_image(aoi, req.before_date, req.before_date, "VV")
            after_sar = get_sentinel1_image(aoi, req.after_date, req.after_date, "VV")
            
            flood_mask = detect_flood_extent(before_sar, after_sar)
            
            before_map = before_sar.select("VV").getMapId({"min": -25, "max": 0})
            after_map = after_sar.select("VV").getMapId({"min": -25, "max": 0})
            change_map = flood_mask.getMapId({"palette": ["blue"]})
            
            return {
                "status": "success",
                "monitoring_type": req.monitoring_type,
                "before_date": req.before_date,
                "after_date": req.after_date,
                "before_tile_url": before_map["tile_fetcher"].url_format,
                "after_tile_url": after_map["tile_fetcher"].url_format,
                "change_tile_url": change_map["tile_fetcher"].url_format,
                "change_type": "flood_extent",
                "bounds": aoi.bounds().getInfo()["coordinates"][0]
            }
        else:
            # Optical-based landslide/change detection
            before_optical = get_sentinel2_image(aoi, req.before_date, req.before_date)
            after_optical = get_sentinel2_image(aoi, req.after_date, req.after_date)
            
            before_rgb = before_optical.select(["B4", "B3", "B2"])
            after_rgb = after_optical.select(["B4", "B3", "B2"])
            
            before_map = before_rgb.getMapId({"min": 0, "max": 3000, "gamma": 1.4})
            after_map = after_rgb.getMapId({"min": 0, "max": 3000, "gamma": 1.4})
            
            return {
                "status": "success",
                "monitoring_type": req.monitoring_type,
                "before_date": req.before_date,
                "after_date": req.after_date,
                "before_tile_url": before_map["tile_fetcher"].url_format,
                "after_tile_url": after_map["tile_fetcher"].url_format,
                "change_type": "visual_comparison",
                "bounds": aoi.bounds().getInfo()["coordinates"][0]
            }
            
    except Exception as e:
        raise HTTPException(500, f"Before/after comparison error: {str(e)}")


@router.get("/status")
async def ee_status():
    """Check Earth Engine API status and project configuration"""
    try:
        # Test basic EE operation
        test = ee.Number(1).getInfo()
        return {
            "status": "operational",
            "project_id": EE_PROJECT_ID,
            "api_version": ee.__version__,
            "test_result": test
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "project_id": EE_PROJECT_ID
        }
