import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import networkx as nx
import torch
from services.location_service import haversine_distance_m, bbox_from_radius, bbox_from_polygon
from services.osm_road_service import OSMRoadService
from services.osm_river_service import OSMRiverService
from services.graph_builder import UnifiedGraphBuilder
from services.risk_model import GNNTransformerFloodModel, FloodRiskEngine
from services.routing_service import EvacuationRoutingService


def test_location_service():
    lat, lng = 9.9252, 78.1198
    # 1.0 km radius bbox
    bbox = bbox_from_radius(lat, lng, 1.0)
    assert bbox["north"] > lat
    assert bbox["south"] < lat
    assert bbox["east"] > lng
    assert bbox["west"] < lng
    
    dist = haversine_distance_m(lat, lng, lat + 0.01, lng)
    assert 1000 < dist < 1200

    # Polygon bbox [[lat, lng], ...]
    poly = [[9.92, 78.11], [9.92, 78.13], [9.93, 78.13], [9.93, 78.11]]
    p_bbox = bbox_from_polygon(poly)
    assert p_bbox["north"] >= 9.93
    assert p_bbox["south"] <= 9.92


def test_osm_road_service_offline():
    service = OSMRoadService()
    elements = [
        {"type": "node", "id": 1, "lat": 9.92, "lon": 78.11},
        {"type": "node", "id": 2, "lat": 9.93, "lon": 78.12},
        {"type": "node", "id": 3, "lat": 9.94, "lon": 78.13},
        {
            "type": "way",
            "id": 101,
            "nodes": [1, 2],
            "tags": {"highway": "primary", "name": "Main Highway", "maxspeed": "60"}
        },
        {
            "type": "way",
            "id": 102,
            "nodes": [2, 3],
            "tags": {"highway": "secondary", "name": "Valley Road", "oneway": "yes"}
        }
    ]
    graph = service.build_graph_from_osm_elements(elements)
    assert isinstance(graph, nx.DiGraph)
    assert len(graph.nodes) == 3
    # Node 1 -> 2 is two-way (2 edges), Node 2 -> 3 is one-way (1 edge)
    assert graph.has_edge(1, 2)
    assert graph.has_edge(2, 1)
    assert graph.has_edge(2, 3)
    assert not graph.has_edge(3, 2)

    # GeoJSON
    geojson = service.graph_to_geojson(graph)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0


def test_osm_river_service_offline():
    service = OSMRiverService()
    elements = [
        {"type": "node", "id": 10, "lat": 9.95, "lon": 78.10},
        {"type": "node", "id": 11, "lat": 9.94, "lon": 78.11},
        {"type": "node", "id": 12, "lat": 9.93, "lon": 78.12},
        {
            "type": "way",
            "id": 201,
            "nodes": [10, 11, 12],
            "tags": {"waterway": "river", "name": "Vaigai River"}
        }
    ]
    graph = service.build_graph_from_osm_elements(elements)
    assert isinstance(graph, nx.DiGraph)
    assert len(graph.nodes) == 3
    assert graph.has_edge(10, 11)
    assert graph.has_edge(11, 12)

    geojson = service.graph_to_geojson(graph)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2

    # Distance to river
    dist = service.find_distance_to_nearest_river(graph, 9.94, 78.11)
    assert dist == 0.0


def test_gnn_transformer_model():
    model = GNNTransformerFloodModel(node_features=8, hidden_dim=32)
    num_nodes = 5
    x = torch.randn(num_nodes, 8)
    adj = torch.eye(num_nodes)

    prob, severity = model(x, adj)
    assert prob.shape == (num_nodes, 1)
    assert severity.shape == (num_nodes, 4)
    assert (prob >= 0.0).all() and (prob <= 1.0).all()


def test_unified_graph_and_risk_routing():
    # Setup road graph
    road_graph = nx.DiGraph()
    road_graph.add_node(1, lat=9.920, lng=78.110, elevation=120.0, highway_type="primary", is_junction=True)
    road_graph.add_node(2, lat=9.925, lng=78.115, elevation=118.0, highway_type="primary", is_junction=False)
    road_graph.add_node(3, lat=9.930, lng=78.120, elevation=125.0, highway_type="primary", is_junction=True)
    road_graph.add_edge(1, 2, length=600.0, highway="primary", speed_kmh=50, flood_risk=0.0)
    road_graph.add_edge(2, 3, length=600.0, highway="primary", speed_kmh=50, flood_risk=0.0)

    # River graph
    river_graph = nx.DiGraph()
    river_graph.add_node(10, lat=9.924, lng=78.116, elevation=115.0, waterway_type="river")
    river_graph.add_node(11, lat=9.928, lng=78.122, elevation=114.0, waterway_type="river")
    river_graph.add_edge(10, 11, length=800.0, waterway="river")

    sensors = [
        {
            "id": "master_1",
            "name": "River Gateway",
            "type": "master",
            "lat": 9.924,
            "lng": 78.115,
            "elevation": 116.0,
            "readings": {"water_level_m": 4.5, "rainfall_mm": 65.0, "soil_moisture_pct": 82.0}
        }
    ]

    shelters = [
        {
            "id": "shelter_1",
            "name": "Highland Relief Center",
            "lat": 9.930,
            "lng": 78.120,
            "elevation": 135.0,
            "capacity": 500
        }
    ]

    builder = UnifiedGraphBuilder()
    unified_res = builder.build_unified_graph(road_graph, river_graph, sensors, shelters)
    assert unified_res["summary"]["total_nodes"] > 0
    unified_graph = unified_res["graph"]

    # Risk Engine
    engine = FloodRiskEngine()
    risk_results = engine.predict_graph_risk(unified_graph)
    assert "node_predictions" in risk_results
    assert "high_risk_zones" in risk_results
    assert len(risk_results["node_predictions"]) == len(unified_graph.nodes)

    # Evacuation Routing (Safe high-ground exit or custom destination)
    router = EvacuationRoutingService()
    route_res = router.calculate_safest_route(
        road_graph=road_graph,
        user_lat=9.920,
        user_lng=78.110,
        dest_lat=9.930,
        dest_lng=78.120,
        destination_name="Highland Relief Junction",
        flood_risk_predictions=risk_results
    )
    assert route_res["status"] == "success"
    assert len(route_res["coordinates"]) >= 2
    assert route_res["destination_name"] == "Highland Relief Junction"

    # Route with automatic high-ground exit selection (no destination supplied)
    auto_route = router.calculate_safest_route(
        road_graph=road_graph,
        user_lat=9.920,
        user_lng=78.110,
        flood_risk_predictions=risk_results
    )
    assert auto_route["status"] == "success"
    assert len(auto_route["coordinates"]) >= 2
    assert "Safest High-Ground" in auto_route["destination_name"]
    print("All unit tests passed successfully!")


if __name__ == "__main__":
    test_location_service()
    test_osm_road_service_offline()
    test_osm_river_service_offline()
    test_gnn_transformer_model()
    test_unified_graph_and_risk_routing()
    print("✓ All 5 tests completed and PASSED!")
