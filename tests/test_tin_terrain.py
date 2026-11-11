"""Unit and integration tests for TIN Terrain service and FastAPI endpoint."""

import io
import math
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from unittest.mock import patch, AsyncMock
from fastapi import HTTPException

from services.tin_service import (
    compute_cache_key,
    process_dem_to_tin,
    generate_or_get_tin,
    calculate_terrain_attributes,
)
from routers.digital_twin import get_tin_terrain


def _create_mock_geotiff(
    rows: int = 50,
    cols: int = 50,
    min_elev: float = 200.0,
    max_elev: float = 300.0,
    with_nodata: bool = True,
) -> bytes:
    """Helper to synthesize a valid GeoTIFF DEM buffer in memory."""
    lons = np.linspace(76.99, 77.01, cols)
    lats = np.linspace(10.66, 10.64, rows)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    elev_grid = min_elev + (max_elev - min_elev) * (
        0.5 + 0.3 * np.sin(lon_grid * 150) + 0.2 * np.cos(lat_grid * 150)
    ).astype(np.float32)

    if with_nodata:
        elev_grid[0, 0] = -9999.0
        elev_grid[5, 5] = np.nan

    transform = from_bounds(76.99, 10.64, 77.01, 10.66, cols, rows)
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elev_grid, 1)

    buf.seek(0)
    return buf.getvalue()


def test_compute_cache_key():
    k1 = compute_cache_key(10.66, 10.64, 77.01, 76.99, "COP30", 25000)
    k2 = compute_cache_key(10.66, 10.64, 77.01, 76.99, "COP30", 25000)
    k3 = compute_cache_key(10.67, 10.64, 77.01, 76.99, "COP30", 25000)
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 24


def test_process_dem_to_tin_structure():
    mock_tif = _create_mock_geotiff(rows=40, cols=40, with_nodata=True)
    res = process_dem_to_tin(mock_tif, max_vertices=25000)

    assert res["dem_source"] == "Copernicus DEM GLO-30"
    assert "vertex_count" in res
    assert "triangle_count" in res
    assert "min_elevation" in res
    assert "max_elevation" in res
    assert "vertices" in res
    assert "triangles" in res
    assert "terrain_features" in res

    assert res["vertex_count"] > 1000
    assert res["triangle_count"] > 2000
    assert len(res["vertices"]) == res["vertex_count"]
    assert len(res["triangles"]) == res["triangle_count"]

    # Verify vertex coordinate format [lon, lat, elev]
    v0 = res["vertices"][0]
    assert len(v0) == 3
    assert 76.98 <= v0[0] <= 77.02
    assert 10.63 <= v0[1] <= 10.67
    assert 150.0 <= v0[2] <= 350.0

    # Verify triangles format [i, j, k]
    t0 = res["triangles"][0]
    assert len(t0) == 3
    assert all(0 <= idx < res["vertex_count"] for idx in t0)

    # Check terrain features for ML/GNN
    tf = res["terrain_features"]
    assert "mean_slope_deg" in tf
    assert "max_slope_deg" in tf
    assert "mean_aspect_deg" in tf
    assert "mean_gradient" in tf
    assert "gnn_node_attributes" in tf
    assert tf["gnn_node_attributes"]["node_count"] == res["vertex_count"]

    # Check GNN Graph model execution
    assert "gnn" in res
    gnn = res["gnn"]
    assert gnn["status"] == "connected"
    assert gnn["graph_count"] == 1
    assert gnn["node_count"] == res["vertex_count"]
    assert gnn["edge_count"] > 0
    assert len(gnn["node_scores"]) == res["vertex_count"]
    assert 0.0 <= gnn["mean_node_risk"] <= 1.0


def test_process_dem_downsampling():
    # Large 200x200 grid = 40,000 cells. With max_vertices=5,000, must downsample
    mock_tif = _create_mock_geotiff(rows=200, cols=200, with_nodata=False)
    res = process_dem_to_tin(mock_tif, max_vertices=5000)

    assert res["vertex_count"] <= 5000
    assert res["triangle_count"] > 0


@pytest.mark.asyncio
async def test_generate_or_get_tin_with_cache(tmp_path):
    mock_tif = _create_mock_geotiff(rows=30, cols=30)
    with patch("services.tin_service.fetch_opentopography_dem", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_tif

        # Call 1: fetch from mock API
        res1 = await generate_or_get_tin(
            north=10.66,
            south=10.64,
            east=77.01,
            west=76.99,
            refresh=True,
        )
        assert mock_fetch.call_count == 1
        assert res1["vertex_count"] > 0

        # Call 2: fetch from cache (refresh=False)
        res2 = await generate_or_get_tin(
            north=10.66,
            south=10.64,
            east=77.01,
            west=76.99,
            refresh=False,
        )
        assert mock_fetch.call_count == 1  # Should NOT have incremented
        assert res2["vertex_count"] == res1["vertex_count"]


@pytest.mark.asyncio
async def test_router_tin_invalid_bounds():
    # South >= North
    with pytest.raises(HTTPException) as exc_info:
        await get_tin_terrain(
            north=10.60,
            south=10.70,
            east=77.01,
            west=76.99,
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_router_tin_success():
    mock_tif = _create_mock_geotiff(rows=25, cols=25)
    with patch("services.tin_service.fetch_opentopography_dem", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_tif
        res = await get_tin_terrain(
            north=10.66,
            south=10.64,
            east=77.01,
            west=76.99,
            refresh=True,
        )
        assert res["dem_source"] == "Copernicus DEM GLO-30"
        assert res["vertex_count"] > 0
        assert res["triangle_count"] > 0
        assert "vertices" in res
        assert "triangles" in res
