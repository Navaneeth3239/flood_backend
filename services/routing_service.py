"""Evacuation Routing Service: Risk-Weighted A*/Dijkstra Pathfinding on Real OSM Road Networks."""
import math
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from services.location_service import haversine_distance_m


class EvacuationRoutingService:
    def __init__(self):
        pass

    def calculate_safest_route(
        self,
        road_graph: nx.DiGraph,
        user_lat: float,
        user_lng: float,
        target_shelter: Optional[Dict[str, Any]] = None,
        dest_lat: Optional[float] = None,
        dest_lng: Optional[float] = None,
        destination_name: str = "Safe Relief Destination",
        flood_risk_predictions: Optional[Dict[str, Any]] = None,
        avoid_critical: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculates the safest evacuation path on the real road network.
        Dynamically adjusts edge costs based on predicted flood risk:
        - Safe roads (< 0.3): normal length cost
        - Moderate risk (0.3 - 0.7): penalized weight
        - Flooded / Critical (>= 0.7): blocked / prohibited
        """
        if road_graph.number_of_nodes() == 0:
            return {
                "status": "error",
                "message": "Road network is empty. Please extract road network first.",
            }

        # 1. Find nearest road graph node to user origin
        origin_node = self._find_nearest_node(road_graph, user_lat, user_lng)

        # 2. Determine destination coordinates
        if dest_lat is not None and dest_lng is not None:
            target_lat = float(dest_lat)
            target_lng = float(dest_lng)
        elif target_shelter and target_shelter.get("lat") and target_shelter.get("lng"):
            target_lat = float(target_shelter["lat"])
            target_lng = float(target_shelter["lng"])
            destination_name = target_shelter.get("name", destination_name)
        else:
            # Automatically find highest elevation node with lowest flood risk in road network
            best_node = None
            best_score = -999999.0
            predictions = flood_risk_predictions.get("node_predictions", {}) if flood_risk_predictions else {}
            for n, d in road_graph.nodes(data=True):
                if road_graph.number_of_nodes() > 1 and n == origin_node:
                    continue
                elev = d.get("elevation", 100.0)
                n_key = f"road_{n}" if not str(n).startswith("road_") else n
                risk = predictions.get(n_key, {}).get("flood_probability", 0.1)
                score = elev - (risk * 200.0)
                if score > best_score:
                    best_score = score
                    best_node = n
            if best_node is not None:
                target_lat = road_graph.nodes[best_node]["lat"]
                target_lng = road_graph.nodes[best_node]["lng"]
                destination_name = "Safest High-Ground Junction"
            else:
                target_lat = user_lat
                target_lng = user_lng

        dest_node = self._find_nearest_node(road_graph, target_lat, target_lng)

        if not origin_node or not dest_node:
            return {
                "status": "error",
                "message": "Could not map origin or destination to nearby road intersections.",
            }

        # 2. Build cost-weighted graph
        weighted_G = nx.DiGraph()
        for n, data in road_graph.nodes(data=True):
            weighted_G.add_node(n, **data)

        predictions = flood_risk_predictions.get("node_predictions", {}) if flood_risk_predictions else {}

        blocked_edges_count = 0
        moderate_edges_count = 0

        for u, v, data in road_graph.edges(data=True):
            length = float(data.get("length", 50.0))
            hw_type = data.get("highway", "residential")
            speed_kmh = float(data.get("speed_kmh", 30))

            # Retrieve node flood risks
            u_key = f"road_{u}" if not str(u).startswith("road_") else u
            v_key = f"road_{v}" if not str(v).startswith("road_") else v

            u_risk = predictions.get(u_key, {}).get("flood_probability", 0.05)
            v_risk = predictions.get(v_key, {}).get("flood_probability", 0.05)
            edge_risk = max(u_risk, v_risk)

            # Dynamic Cost Function
            if edge_risk >= 0.70 and avoid_critical:
                # Prohibited: Flooded/blocked
                cost = float("inf")
                blocked_edges_count += 1
                accessibility = "flooded_blocked"
            elif edge_risk >= 0.30:
                # Moderate risk: heavy penalty
                cost = length * (1.0 + 8.0 * edge_risk)
                moderate_edges_count += 1
                accessibility = "moderate_risk"
            else:
                # Safe road
                cost = length * 1.0
                accessibility = "safe"

            if cost < float("inf"):
                weighted_G.add_edge(
                    u,
                    v,
                    weight=cost,
                    length=length,
                    speed_kmh=speed_kmh,
                    edge_risk=edge_risk,
                    accessibility=accessibility,
                    name=data.get("name", "Unnamed Road"),
                    highway=hw_type,
                    coordinates=data.get("coordinates", []),
                )

        # 3. Pathfinding via Dijkstra on weighted graph
        path_nodes = None
        try:
            path_nodes = nx.shortest_path(weighted_G, source=origin_node, target=dest_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback: if all paths are blocked by floodwaters, attempt least-risk path without strict blockage
            print("Strict safe path not found, attempting least-risk emergency path...")
            emergency_G = nx.DiGraph()
            for u, v, data in road_graph.edges(data=True):
                u_key = f"road_{u}" if not str(u).startswith("road_") else u
                v_key = f"road_{v}" if not str(v).startswith("road_") else v
                r = max(predictions.get(u_key, {}).get("flood_probability", 0.05),
                        predictions.get(v_key, {}).get("flood_probability", 0.05))
                emergency_G.add_edge(u, v, weight=float(data.get("length", 50)) * (1.0 + 10.0 * r))

            try:
                path_nodes = nx.shortest_path(emergency_G, source=origin_node, target=dest_node, weight="weight")
            except Exception:
                path_nodes = None

        if not path_nodes:
            return {
                "status": "no_path",
                "message": "No accessible road route to shelter found due to severe flooding.",
                "origin": [user_lat, user_lng],
                "destination": [target_lat, target_lng],
            }

        # 4. Construct detailed step-by-step route coordinates & metrics
        route_coords: List[List[float]] = []
        # Add actual user starting point
        route_coords.append([user_lat, user_lng])

        total_distance_m = 0.0
        total_time_sec = 0.0
        max_route_risk = 0.0
        segments = []

        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]

            edge_data = road_graph.get_edge_data(u, v) or {}
            seg_len = float(edge_data.get("length", haversine_distance_m(
                road_graph.nodes[u]["lat"], road_graph.nodes[u]["lng"],
                road_graph.nodes[v]["lat"], road_graph.nodes[v]["lng"]
            )))
            spd = float(edge_data.get("speed_kmh", 30))

            u_key = f"road_{u}" if not str(u).startswith("road_") else u
            v_key = f"road_{v}" if not str(v).startswith("road_") else v
            r = max(predictions.get(u_key, {}).get("flood_probability", 0.05),
                    predictions.get(v_key, {}).get("flood_probability", 0.05))

            max_route_risk = max(max_route_risk, r)
            total_distance_m += seg_len
            total_time_sec += seg_len / (spd * (1000.0 / 3600.0))

            u_coord = [road_graph.nodes[u]["lat"], road_graph.nodes[u]["lng"]]
            v_coord = [road_graph.nodes[v]["lat"], road_graph.nodes[v]["lng"]]

            if not route_coords or route_coords[-1] != u_coord:
                route_coords.append(u_coord)
            route_coords.append(v_coord)

            segments.append({
                "from_node": u,
                "to_node": v,
                "road_name": edge_data.get("name", "Connecting Road"),
                "road_type": edge_data.get("highway", "residential"),
                "length_m": round(seg_len, 1),
                "speed_kmh": spd,
                "flood_risk": round(r, 2),
            })

        # Final step into the shelter
        if route_coords[-1] != [target_lat, target_lng]:
            route_coords.append([target_lat, target_lng])

        route_status = "SAFE" if max_route_risk < 0.35 else ("CAUTION" if max_route_risk < 0.70 else "HAZARDOUS")

        shelter_id = target_shelter.get("id", "dest-1") if target_shelter else "dest-safe-point"
        shelter_name = target_shelter.get("name", destination_name) if target_shelter else destination_name
        shelter_cap = target_shelter.get("capacity", 0) if target_shelter else 0

        return {
            "status": "success",
            "route_status": route_status,
            "destination_name": destination_name,
            "origin": {"lat": user_lat, "lng": user_lng},
            "destination": {
                "id": shelter_id,
                "name": destination_name,
                "lat": target_lat,
                "lng": target_lng,
            },
            "shelter": {
                "id": shelter_id,
                "name": shelter_name,
                "capacity": shelter_cap,
                "lat": target_lat,
                "lng": target_lng,
            },
            "total_distance_km": round(total_distance_m / 1000.0, 2),
            "total_distance_m": round(total_distance_m, 1),
            "estimated_time_minutes": max(1, round(total_time_sec / 60.0)),
            "max_flood_risk_encountered": round(max_route_risk, 3),
            "path_nodes_count": len(path_nodes),
            "coordinates": route_coords,
            "segments": segments,
            "avoided_blocked_edges": blocked_edges_count,
        }

    def _find_nearest_node(self, G: nx.DiGraph, lat: float, lng: float) -> Optional[Any]:
        best = None
        min_d = float("inf")
        # Fast Manhattan coordinate delta bounding before computing full spherical haversine
        for node, data in G.nodes(data=True):
            n_lat = data.get("lat")
            n_lng = data.get("lng")
            if n_lat is not None and n_lng is not None:
                deg_bound = min_d / 111132.0
                if abs(n_lat - lat) < deg_bound and abs(n_lng - lng) < deg_bound:
                    d = haversine_distance_m(lat, lng, n_lat, n_lng)
                    if d < min_d:
                        min_d = d
                        best = node
        return best
