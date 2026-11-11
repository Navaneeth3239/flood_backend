import asyncio
from unittest.mock import AsyncMock

import networkx as nx

from routers import routing_and_rivers
from services.osm_geometry import geometry_intersects_polygon, join_rings
from services.osm_river_service import OSMRiverService
from services.osm_road_service import OSMRoadService


BOUNDARY = [[10, 77], [10, 77.01], [10.01, 77.01], [10.01, 77]]


def test_real_road_download_builds_graph_and_caches_geometry(monkeypatch, tmp_path):
    service = OSMRoadService()
    monkeypatch.setattr(service, "_cache_key", lambda *args: tmp_path / "roads.json")
    provider = AsyncMock(return_value=[
        {"type": "node", "id": 1, "lat": 10.002, "lon": 77.002},
        {"type": "node", "id": 2, "lat": 10.008, "lon": 77.008},
        {"type": "way", "id": 3, "nodes": [1, 2], "tags": {"highway": "primary"}},
    ])
    monkeypatch.setattr(service, "fetch_road_elements_overpass", provider)
    graph, layer = asyncio.run(service.get_road_network(10.01, 10, 77.01, 77))
    assert graph.has_edge(1, 2)
    assert layer["features"][0]["properties"]["id"] == "way-3"
    _, cached = asyncio.run(service.get_road_network(10.01, 10, 77.01, 77))
    assert cached == layer
    provider.assert_awaited_once()


def test_selected_polygon_overrides_place_name_and_viewport(monkeypatch):
    geocode = AsyncMock(side_effect=AssertionError("Do not geocode a selected polygon"))
    monkeypatch.setattr(routing_and_rivers, "geocode_place_name", geocode)
    response = (nx.DiGraph(), {"type": "FeatureCollection", "features": []})
    roads = AsyncMock(return_value=response)
    monkeypatch.setattr(routing_and_rivers.road_service, "get_road_network", roads)
    monkeypatch.setattr(routing_and_rivers.river_service, "get_river_network", AsyncMock(return_value=response))
    payload = routing_and_rivers.LocationRequest(polygon=BOUNDARY, place_name="Wrong area", north=20, south=19, east=88, west=87)
    result = asyncio.run(routing_and_rivers.extract_networks(payload))
    assert result["bbox"]["north"] < 10.02
    assert result["bbox"]["west"] > 76.99
    assert result["osm_loading"]["complete"]
    geocode.assert_not_awaited()


def test_upstream_failure_is_not_reported_as_complete(monkeypatch):
    monkeypatch.setattr(routing_and_rivers.road_service, "get_road_network", AsyncMock(side_effect=RuntimeError("Offline")))
    monkeypatch.setattr(routing_and_rivers.river_service, "get_river_network", AsyncMock(return_value=(nx.DiGraph(), {"type": "FeatureCollection", "features": []})))
    result = asyncio.run(routing_and_rivers.extract_networks(routing_and_rivers.LocationRequest(polygon=BOUNDARY)))
    assert not result["osm_loading"]["complete"]
    assert result["osm_loading"]["failed_layers"] == ["roads"]


def test_crossing_paths_and_enclosing_buildings_intersect_selection():
    assert geometry_intersects_polygon({"type": "LineString", "coordinates": [[76.99, 10.005], [77.02, 10.005]]}, BOUNDARY)
    assert geometry_intersects_polygon({"type": "Polygon", "coordinates": [[[76, 9], [78, 9], [78, 11], [76, 11], [76, 9]]]}, BOUNDARY)
    assert not geometry_intersects_polygon({"type": "LineString", "coordinates": [[76, 9], [76.1, 9.1]]}, BOUNDARY)


def test_open_member_ways_are_joined_without_fabricating_closures():
    assert len(join_rings([[[0, 0], [1, 0], [1, 1]], [[0, 0], [0, 1], [1, 1]]])) == 1
    assert join_rings([[[0, 0], [1, 0], [1, 1]]]) == []


def test_water_relations_preserve_islands(monkeypatch, tmp_path):
    service = OSMRiverService()
    monkeypatch.setattr(service, "_cache_key", lambda *args: tmp_path / "network.json")
    elements = [
        {"type": "node", "id": index + 1, "lon": point[0], "lat": point[1]}
        for index, point in enumerate([[77, 10], [77.01, 10], [77.01, 10.01], [77, 10.01], [77.004, 10.004], [77.006, 10.004], [77.006, 10.006], [77.004, 10.006]])
    ] + [
        {"type": "way", "id": 20, "nodes": [1, 2, 3]},
        {"type": "way", "id": 21, "nodes": [1, 4, 3]},
        {"type": "way", "id": 22, "nodes": [5, 6, 7, 8, 5]},
        {"type": "relation", "id": 30, "tags": {"type": "multipolygon", "natural": "water"}, "members": [
            {"type": "way", "ref": 20, "role": "outer"}, {"type": "way", "ref": 21, "role": "outer"}, {"type": "way", "ref": 22, "role": "inner"},
        ]},
    ]
    monkeypatch.setattr(service, "fetch_waterway_elements_overpass", AsyncMock(return_value=elements))
    _, collection = asyncio.run(service.get_river_network(10.01, 10, 77.01, 77))
    assert len(collection["features"]) == 1
    assert len(collection["features"][0]["geometry"]["coordinates"]) == 2
