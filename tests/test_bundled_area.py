import asyncio
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

from routers import routing_and_rivers as api
from services.bundled_area import DIRECTORY, load_bundled_area


def test_snapshot_is_identical_in_both_deployments_and_contains_real_geometry():
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    entry = manifest[0]
    content = gzip.decompress((DIRECTORY / entry["file"]).read_bytes())
    root = Path(__file__).resolve().parents[2]
    assert content == (root / "frontend/public/prebaked_zones" / entry["file"].removesuffix(".gz")).read_bytes()
    assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    snapshot = json.loads(content)
    assert entry["counts"] == {"roads": 194, "rivers": 349, "buildings": 2647}
    for layer, count in entry["counts"].items():
        features = snapshot[layer]["geojson"]["features"]
        assert len(features) == count
        assert all("fallback" not in str(f.get("id") or f.get("properties", {}).get("id")) for f in features)


def test_api_serves_all_layers_without_database_or_osm(monkeypatch):
    entry = json.loads((DIRECTORY / "manifest.json").read_text())[0]
    def offline(*args):
        raise AssertionError("Must not access database")
    monkeypatch.setattr(api.area_map_store, "load_layers", offline)
    for service, method in [(api.road_service, "get_road_network"), (api.river_service, "get_river_network"), (api.building_service, "get_buildings")]:
        monkeypatch.setattr(service, method, AsyncMock(side_effect=AssertionError("Must not access OSM")))
    payload = api.LocationRequest(area_id=entry["area_id"], polygon=entry["polygon"])
    result = asyncio.run(api.extract_networks(payload))
    assert result["osm_loading"]["complete"]
    assert len(result["buildings"]["geojson"]["features"]) == 2647
    assert asyncio.run(api.extract_buildings(payload))["buildings"]["total_features"] == 2647
    payload.polygon[0][0] += .001
    assert load_bundled_area(payload) is None
    assert load_bundled_area(api.LocationRequest(lat=31, lng=79)) is None
    assert load_bundled_area(api.LocationRequest(area_id="../../default")) is None
