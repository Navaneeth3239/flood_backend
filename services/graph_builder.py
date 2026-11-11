"""Unified Spatial Graph Builder: Fuses River, Road, IoT Sensor, and Shelter Networks into a Multimodal Spatial Graph for GNNs."""
import math
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from services.location_service import haversine_distance_m


class SpatialGridIndex:
    """Fast 2D Spatial Grid Hash Index for O(1) average-time nearest-neighbor queries."""
    def __init__(self, cell_size_deg: float = 0.005):  # ~500m grid cells
        self.cell_size = cell_size_deg
        self.grid: Dict[Tuple[int, int], List[Tuple[Any, float, float]]] = {}
        self.all_items: List[Tuple[Any, float, float]] = []

    def insert(self, item_id: Any, lat: float, lng: float):
        if lat is None or lng is None:
            return
        cell_x = int(math.floor(lat / self.cell_size))
        cell_y = int(math.floor(lng / self.cell_size))
        entry = (item_id, lat, lng)
        self.grid.setdefault((cell_x, cell_y), []).append(entry)
        self.all_items.append(entry)

    def find_nearest(self, q_lat: float, q_lng: float, max_dist_m: float = 5000.0) -> Tuple[Optional[Any], float]:
        if not self.all_items or q_lat is None or q_lng is None:
            return None, float("inf")

        q_cx = int(math.floor(q_lat / self.cell_size))
        q_cy = int(math.floor(q_lng / self.cell_size))

        cell_radius = max(1, int(math.ceil((max_dist_m / 111132.0) / self.cell_size)))
        best_item = None
        min_dist = float("inf")

        for r in range(0, min(cell_radius + 1, 10)):
            found_in_ring = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    cell_items = self.grid.get((q_cx + dx, q_cy + dy))
                    if cell_items:
                        for item_id, lat, lng in cell_items:
                            # Fast Manhattan degree threshold filter before haversine
                            deg_bound = min_dist / 111132.0
                            if abs(lat - q_lat) < deg_bound and abs(lng - q_lng) < deg_bound:
                                d = haversine_distance_m(q_lat, q_lng, lat, lng)
                                if d < min_dist:
                                    min_dist = d
                                    best_item = item_id
                                    found_in_ring = True
            if found_in_ring and min_dist <= (r * self.cell_size * 111132.0):
                break

        # Fallback to general minimum if within reasonable bounds
        if best_item is None and min_dist > max_dist_m:
            for item_id, lat, lng in self.all_items:
                d = haversine_distance_m(q_lat, q_lng, lat, lng)
                if d < min_dist:
                    min_dist = d
                    best_item = item_id

        return best_item, min_dist


class UnifiedGraphBuilder:
    def __init__(self):
        pass

    def build_unified_graph(
        self,
        road_graph: nx.DiGraph,
        river_graph: nx.DiGraph,
        sensors: Optional[List[Dict[str, Any]]] = None,
        shelters: Optional[List[Dict[str, Any]]] = None,
        default_rainfall_mm: float = 35.0,
        default_elevation_m: float = 240.0,
    ) -> Dict[str, Any]:
        """
        Constructs a unified heterogeneous spatial graph with:
        - River nodes & edges
        - Road nodes & edges
        - IoT Sensor nodes & proximity edges to rivers
        - Optional safe destination nodes
        Enriches all nodes with multi-modal feature vectors using O(1) Spatial Grid Indexing.
        """
        unified = nx.DiGraph()
        sensors = sensors or []
        shelters = shelters or []

        # Build 2D Spatial Grid Index for River Nodes: O(M)
        river_spatial_index = SpatialGridIndex(cell_size_deg=0.005)

        # 1. Add River Nodes & Edges
        for node, data in river_graph.nodes(data=True):
            node_id = f"river_{node}"
            lat = data.get("lat", 0.0)
            lng = data.get("lng", 0.0)
            elevation = data.get("elevation", default_elevation_m)
            water_level = data.get("water_level", 2.5)

            river_spatial_index.insert(node_id, lat, lng)

            unified.add_node(
                node_id,
                domain="river",
                type="river_node",
                lat=lat,
                lng=lng,
                elevation=elevation,
                slope=data.get("slope", 1.8),
                rainfall=default_rainfall_mm,
                water_level=water_level,
                soil_moisture=85.0,
                distance_to_river=0.0,
                flood_risk=min(1.0, 0.45 + (water_level / 10.0)),
                landslide_risk=0.15,
                osm_id=node,
            )

        for u, v, data in river_graph.edges(data=True):
            u_id = f"river_{u}"
            v_id = f"river_{v}"
            if u_id in unified and v_id in unified:
                unified.add_edge(
                    u_id,
                    v_id,
                    relation="flows_to",
                    length=data.get("length", 100.0),
                    waterway=data.get("waterway", "stream"),
                    capacity_m3s=data.get("width_m", 5.0) * 1.5,
                )

        # Build 2D Spatial Grid Index for Road Nodes: O(N)
        road_spatial_index = SpatialGridIndex(cell_size_deg=0.005)

        # 2. Add Road Nodes & Edges with O(1) average nearest river lookup
        for node, data in road_graph.nodes(data=True):
            node_id = f"road_{node}"
            lat = data.get("lat", 0.0)
            lng = data.get("lng", 0.0)
            elevation = data.get("elevation", default_elevation_m + 3.0)

            road_spatial_index.insert(node_id, lat, lng)

            # High-performance spatial query via grid hash
            _, dist_to_river = river_spatial_index.find_nearest(lat, lng, max_dist_m=5000.0)
            dist_to_river = round(min(dist_to_river if dist_to_river != float("inf") else 5000.0, 5000.0), 1)

            # Baseline flood risk correlates inversely with distance to river and elevation
            base_risk = max(0.05, min(0.95, 0.7 - (dist_to_river / 600.0) * 0.5))

            unified.add_node(
                node_id,
                domain="road",
                type="road_intersection",
                lat=lat,
                lng=lng,
                elevation=elevation,
                slope=data.get("slope", 2.5),
                rainfall=default_rainfall_mm,
                water_level=max(0.0, 0.5 - (dist_to_river / 400.0)),
                soil_moisture=max(30.0, 75.0 - (dist_to_river / 300.0)),
                distance_to_river=dist_to_river,
                flood_risk=round(base_risk, 3),
                landslide_risk=round(min(1.0, 0.1 + (data.get("slope", 2.5) / 30.0)), 3),
                osm_id=node,
            )

        for u, v, data in road_graph.edges(data=True):
            u_id = f"road_{u}"
            v_id = f"road_{v}"
            if u_id in unified and v_id in unified:
                unified.add_edge(
                    u_id,
                    v_id,
                    relation="road_connects",
                    length=data.get("length", 100.0),
                    highway=data.get("highway", "residential"),
                    speed_kmh=data.get("speed_kmh", 30),
                    accessibility=data.get("accessibility", "open"),
                )

        # 3. Add IoT Sensor Nodes & Link to Nearest River via spatial index
        for idx, sensor in enumerate(sensors):
            sensor_id = f"sensor_{sensor.get('id', idx)}"
            s_lat = float(sensor.get("lat", 0.0))
            s_lng = float(sensor.get("lng", 0.0))
            s_type = sensor.get("type", "slave")

            unified.add_node(
                sensor_id,
                domain="sensor",
                type="iot_station",
                sensor_type=s_type,
                name=sensor.get("name", f"Station {idx + 1}"),
                lat=s_lat,
                lng=s_lng,
                elevation=default_elevation_m,
                slope=1.0,
                rainfall=float(sensor.get("rainfall_mm", default_rainfall_mm)),
                water_level=float(sensor.get("water_level_m", 2.1)),
                soil_moisture=float(sensor.get("soil_moisture_pct", 68.0)),
                distance_to_river=0.0,
                flood_risk=0.4,
                landslide_risk=0.1,
                battery=sensor.get("battery", 90),
                signalDbm=sensor.get("signalDbm", -72),
            )

            # Connect sensor to nearest river node within 1.5 km
            nearest_river_node, min_r_dist = river_spatial_index.find_nearest(s_lat, s_lng, max_dist_m=1500.0)

            if nearest_river_node and min_r_dist < 1500.0:
                unified.add_edge(
                    sensor_id,
                    nearest_river_node,
                    relation="monitors_waterway",
                    distance=round(min_r_dist, 1),
                )

        # 4. Add Emergency Shelters & Link to Nearest Road Intersections via road spatial index
        for idx, shelter in enumerate(shelters):
            shelter_id = f"shelter_{shelter.get('id', idx)}"
            sh_lat = float(shelter.get("lat", 0.0))
            sh_lng = float(shelter.get("lng", 0.0))

            unified.add_node(
                shelter_id,
                domain="shelter",
                type="evacuation_facility",
                name=shelter.get("name", f"Safe Shelter {idx + 1}"),
                capacity=shelter.get("capacity", 250),
                lat=sh_lat,
                lng=sh_lng,
                elevation=default_elevation_m + 15.0,  # Typically on high safe ground
                slope=1.0,
                rainfall=default_rainfall_mm,
                water_level=0.0,
                soil_moisture=25.0,
                distance_to_river=850.0,
                flood_risk=0.02,  # Designated safe
                landslide_risk=0.01,
            )

            # Connect shelter to nearest road intersection via spatial index
            nearest_road_node, min_rd_dist = road_spatial_index.find_nearest(sh_lat, sh_lng, max_dist_m=2000.0)

            if nearest_road_node and min_rd_dist < 2000.0:
                # Bidirectional connector between shelter and road network
                unified.add_edge(
                    nearest_road_node,
                    shelter_id,
                    relation="access_to_shelter",
                    length=round(min_rd_dist, 1),
                )
                unified.add_edge(
                    shelter_id,
                    nearest_road_node,
                    relation="evacuee_dispatch",
                    length=round(min_rd_dist, 1),
                )


        # 5. Extract Feature Matrix and Summary
        nodes_list = []
        for n, d in unified.nodes(data=True):
            feature_vec = [
                d.get("lat", 0.0),
                d.get("lng", 0.0),
                d.get("elevation", 0.0),
                d.get("slope", 0.0),
                d.get("rainfall", 0.0),
                d.get("water_level", 0.0),
                d.get("soil_moisture", 0.0),
                d.get("distance_to_river", 0.0),
            ]
            nodes_list.append({
                "id": n,
                "domain": d.get("domain", "road"),
                "name": d.get("name", n),
                "lat": d.get("lat"),
                "lng": d.get("lng"),
                "features": feature_vec,
                "flood_risk": d.get("flood_risk", 0.0),
                "landslide_risk": d.get("landslide_risk", 0.0),
            })

        edges_list = [
            {
                "source": u,
                "target": v,
                "relation": d.get("relation", "connects"),
                "length": d.get("length", 1.0),
            }
            for u, v, d in unified.edges(data=True)
        ]

        return {
            "graph": unified,
            "summary": {
                "total_nodes": unified.number_of_nodes(),
                "total_edges": unified.number_of_edges(),
                "river_nodes": sum(1 for _, d in unified.nodes(data=True) if d.get("domain") == "river"),
                "road_nodes": sum(1 for _, d in unified.nodes(data=True) if d.get("domain") == "road"),
                "sensor_nodes": sum(1 for _, d in unified.nodes(data=True) if d.get("domain") == "sensor"),
                "shelter_nodes": sum(1 for _, d in unified.nodes(data=True) if d.get("domain") == "shelter"),
            },
            "nodes": nodes_list,
            "edges": edges_list,
        }

    def build_road_only_graph(
        self,
        road_graph: nx.DiGraph,
        default_rainfall_mm: float = 35.0,
        default_elevation_m: float = 240.0,
    ) -> Dict[str, Any]:
        """
        Builds a lightweight road-only graph (no rivers, no sensors, no shelters).
        Much faster than build_unified_graph — used for quick evacuation routing.
        Road flood risk is estimated heuristically without river proximity data.
        """
        unified = nx.DiGraph()

        for node, data in road_graph.nodes(data=True):
            node_id = f"road_{node}"
            lat = data.get("lat", 0.0)
            lng = data.get("lng", 0.0)
            elevation = data.get("elevation", default_elevation_m + 3.0)
            # Heuristic baseline flood risk (moderate, no river data)
            base_risk = 0.25

            unified.add_node(
                node_id,
                domain="road",
                type="road_intersection",
                lat=lat,
                lng=lng,
                elevation=elevation,
                slope=data.get("slope", 2.5),
                rainfall=default_rainfall_mm,
                water_level=0.5,
                soil_moisture=60.0,
                distance_to_river=300.0,
                flood_risk=round(base_risk, 3),
                landslide_risk=0.1,
                osm_id=node,
            )

        for u, v, data in road_graph.edges(data=True):
            u_id = f"road_{u}"
            v_id = f"road_{v}"
            if u_id in unified and v_id in unified:
                unified.add_edge(
                    u_id,
                    v_id,
                    relation="road_connects",
                    length=data.get("length", 100.0),
                    highway=data.get("highway", "residential"),
                    speed_kmh=data.get("speed_kmh", 30),
                    accessibility=data.get("accessibility", "open"),
                )

        nodes_list = [
            {
                "id": n,
                "domain": d.get("domain", "road"),
                "lat": d.get("lat"),
                "lng": d.get("lng"),
                "features": [
                    d.get("lat", 0.0), d.get("lng", 0.0), d.get("elevation", 0.0),
                    d.get("slope", 0.0), d.get("rainfall", 0.0), d.get("water_level", 0.0),
                    d.get("soil_moisture", 0.0), d.get("distance_to_river", 0.0),
                ],
                "flood_risk": d.get("flood_risk", 0.0),
                "landslide_risk": d.get("landslide_risk", 0.0),
            }
            for n, d in unified.nodes(data=True)
        ]

        return {
            "graph": unified,
            "summary": {
                "total_nodes": unified.number_of_nodes(),
                "total_edges": unified.number_of_edges(),
                "road_nodes": unified.number_of_nodes(),
            },
            "nodes": nodes_list,
        }

