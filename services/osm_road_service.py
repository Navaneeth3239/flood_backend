"""OSM Road Service: Extracts real road/path network from OpenStreetMap.
GeoJSON output uses full OSM way coordinates for continuous polylines.
"""
import asyncio
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from services.location_service import bbox_from_radius, haversine_distance_m, point_in_polygon
from services.osm_tile_loader import osm_tile_loader
from services.osm_geometry import geometry_intersects_polygon

import os

if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    CACHE_DIR = Path("/tmp") / "cache" / "roads"
else:
    CACHE_DIR = Path(__file__).parent.parent / "cache" / "roads"
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Speed limits by highway type (km/h)
HIGHWAY_SPEEDS = {
    "motorway": 100, "trunk": 80, "primary": 60, "secondary": 50,
    "tertiary": 40, "residential": 30, "unclassified": 30,
    "service": 20, "track": 15, "path": 10, "living_street": 15, "road": 30,
    "footway": 5, "cycleway": 15,
}

# Road width for display (pixels)
ROAD_WIDTHS = {
    "motorway": 8.0, "trunk": 7.0, "primary": 6.0, "secondary": 5.0,
    "tertiary": 4.5, "residential": 3.5, "unclassified": 3.0,
    "service": 2.5, "track": 2.0, "path": 1.5, "footway": 1.5,
    "cycleway": 2.0, "living_street": 3.0, "road": 3.0,
}


def _extract_points(geometry: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Extracts (lat, lng) tuples from LineString, MultiLineString, Polygon, MultiPolygon."""
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates", [])
    points: List[Tuple[float, float]] = []
    if not coords:
        return points
    if gtype == "Point" and len(coords) >= 2:
        points.append((float(coords[1]), float(coords[0])))
    elif gtype == "LineString":
        for p in coords:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append((float(p[1]), float(p[0])))
    elif gtype in ("MultiLineString", "Polygon"):
        for ring in coords:
            if isinstance(ring, (list, tuple)):
                for p in ring:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        points.append((float(p[1]), float(p[0])))
    elif gtype == "MultiPolygon":
        for poly in coords:
            if isinstance(poly, (list, tuple)):
                for ring in poly:
                    if isinstance(ring, (list, tuple)):
                        for p in ring:
                            if isinstance(p, (list, tuple)) and len(p) >= 2:
                                points.append((float(p[1]), float(p[0])))
    return points


class OSMRoadService:
    def __init__(self):
        pass

    def _cache_key(self, north: float, south: float, east: float, west: float) -> Path:
        key_str = f"road_v3_{north:.6f}_{south:.6f}_{east:.6f}_{west:.6f}"
        hash_val = hashlib.md5(key_str.encode()).hexdigest()
        return CACHE_DIR / f"{hash_val}.json"

    def _load_cached_network(self, cache_file: Path, polygon: Optional[List[List[float]]] = None) -> Optional[Tuple[nx.DiGraph, Dict[str, Any]]]:
        """Loads a cached full network and scopes its visible features to a polygon."""
        if not cache_file.exists() or time.time() - cache_file.stat().st_mtime > 86400:
            return None
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            geojson = cached.get("geojson")
            if not geojson or not isinstance(geojson, dict):
                return None

            features = geojson.get("features", [])
            if polygon and len(polygon) >= 3:
                scoped_features = []
                for feat in features:
                    pts = _extract_points(feat.get("geometry", {}))
                    if geometry_intersects_polygon(feat.get("geometry", {}), polygon):
                        scoped_features.append(feat)
                # If polygon filtering matched features, scope to it; otherwise retain all features
                features = scoped_features
                geojson = {
                    "type": "FeatureCollection",
                    "features": features,
                    "metadata": {"total_nodes": len(cached.get("nodes", {})), "total_edges": len(features)},
                }
            return self._reconstruct_graph(cached), geojson
        except Exception as e:
            print(f"Road cache read error: {e}")
            return None

    async def fetch_road_elements_overpass(
        self, north: float, south: float, east: float, west: float
    ) -> List[Dict[str, Any]]:
        """Loads complete roads/paths through the shared tile cache and queue."""
        elements, _ = await osm_tile_loader.load(
            "roads",
            north,
            south,
            east,
            west,
            'way["highway"]{bbox};relation["highway"]{bbox}',
        )
        return elements

    async def get_road_network(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        polygon: Optional[List[List[float]]] = None,
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """Downloads road network for the bbox, builds DiGraph + full-way GeoJSON."""
        cache_file = self._cache_key(north, south, east, west)
        cached = self._load_cached_network(cache_file, polygon)
        if cached:
            return cached

        # Check broad 5km area cache around center to instantly serve requests
        center_lat = (sum(point[0] for point in polygon) / len(polygon)) if (polygon and len(polygon) >= 3) else (north + south) / 2.0
        center_lng = (sum(point[1] for point in polygon) / len(polygon)) if (polygon and len(polygon) >= 3) else (east + west) / 2.0
        broad_bbox = bbox_from_radius(center_lat, center_lng, 5.0)
        broad_cache = self._load_cached_network(
            self._cache_key(broad_bbox["north"], broad_bbox["south"], broad_bbox["east"], broad_bbox["west"]),
            polygon,
        )
        if broad_cache and broad_bbox["north"] >= north and broad_bbox["south"] <= south and broad_bbox["east"] >= east and broad_bbox["west"] <= west:
            return broad_cache

        elements = await self.fetch_road_elements_overpass(north, south, east, west)

        # Build nodes dict
        nodes_dict: Dict[int, Tuple[float, float]] = {}
        for el in elements:
            if el.get("type") == "node":
                nodes_dict[el["id"]] = (float(el["lat"]), float(el["lon"]))

        # Generate full-way GeoJSON — all nodes per way → continuous polylines
        features = []
        ways_for_graph: List[Dict[str, Any]] = []

        for el in elements:
            if el.get("type") != "way":
                continue
            tags = el.get("tags", {})
            hw = tags.get("highway")
            if not hw:
                continue

            node_refs = el.get("nodes", [])
            # Collect all coordinates in way order
            coords = []
            for nid in node_refs:
                if nid in nodes_dict:
                    lat, lon = nodes_dict[nid]
                    coords.append([lon, lat])  # GeoJSON: [lng, lat]

            if len(coords) < 2:
                continue

            name = tags.get("name", "")
            speed = HIGHWAY_SPEEDS.get(hw, 30)
            width = ROAD_WIDTHS.get(hw, 2.5)
            is_major = hw in ("motorway", "trunk", "primary", "secondary")

            features.append({
                "type": "Feature",
                "properties": {
                    "id": f"way-{el['id']}",
                    "name": name,
                    "road_type": hw,
                    "length_m": round(len(coords) * 10, 1),  # Approx
                    "speed_kmh": speed,
                    "width_px": width,
                    "is_major": is_major,
                    "accessibility": "open",
                    "flood_risk": 0.0,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
            })
            ways_for_graph.append({"el": el, "nodes_dict": nodes_dict, "hw": hw, "name": name, "speed": speed})

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"total_nodes": len(nodes_dict), "total_edges": len(features)},
        }

        # Build graph for routing (still needed for evacuation routing)
        G = self._build_graph(elements, nodes_dict, north, south, east, west, polygon)
        self._save_cache(cache_file, G, geojson)
        return G, geojson

    def generate_fallback_roads(self, north: float, south: float, east: float, west: float, polygon: Optional[List[List[float]]] = None) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        G = nx.DiGraph()
        features = []
        lat_s = max(0.01, north - south)
        lng_s = max(0.01, east - west)

        grid_coords = {}
        node_id = 1000
        for r in range(4):
            for c in range(4):
                nlat = round(south + (0.15 + r * 0.23) * lat_s, 6)
                nlng = round(west + (0.15 + c * 0.23) * lng_s, 6)
                nid = node_id
                node_id += 1
                grid_coords[(r, c)] = (nid, nlat, nlng)
                G.add_node(nid, lat=nlat, lng=nlng, type="road_intersection")

        edge_idx = 1
        for r in range(4):
            for c in range(4):
                u_id, u_lat, u_lng = grid_coords[(r, c)]
                if c < 3:
                    v_id, v_lat, v_lng = grid_coords[(r, c + 1)]
                    hw = "primary" if r in (1, 2) else "secondary"
                    dist = round(haversine_distance_m(u_lat, u_lng, v_lat, v_lng), 1)
                    G.add_edge(u_id, v_id, highway=hw, length=dist, speed_kmh=50, flood_risk=0.1, name=f"Highway Corridor {r+1}", coordinates=[[u_lat, u_lng], [v_lat, v_lng]])
                    G.add_edge(v_id, u_id, highway=hw, length=dist, speed_kmh=50, flood_risk=0.1, name=f"Highway Corridor {r+1}", coordinates=[[v_lat, v_lng], [u_lat, u_lng]])
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "id": f"fallback-road-{edge_idx}",
                            "name": f"Highway Corridor {r+1}",
                            "road_type": hw,
                            "length_m": dist,
                            "speed_kmh": 50,
                            "width_px": 5.0 if hw == "primary" else 3.5,
                            "is_major": hw == "primary",
                            "accessibility": "open",
                            "flood_risk": 0.1,
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[u_lng, u_lat], [v_lng, v_lat]]
                        }
                    })
                    edge_idx += 1
                if r < 3:
                    v_id, v_lat, v_lng = grid_coords[(r + 1, c)]
                    hw = "primary" if c in (1, 2) else "tertiary"
                    dist = round(haversine_distance_m(u_lat, u_lng, v_lat, v_lng), 1)
                    G.add_edge(u_id, v_id, highway=hw, length=dist, speed_kmh=40, flood_risk=0.15, name=f"Sector Link {c+1}", coordinates=[[u_lat, u_lng], [v_lat, v_lng]])
                    G.add_edge(v_id, u_id, highway=hw, length=dist, speed_kmh=40, flood_risk=0.15, name=f"Sector Link {c+1}", coordinates=[[v_lat, v_lng], [u_lat, u_lng]])
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "id": f"fallback-road-{edge_idx}",
                            "name": f"Sector Link {c+1}",
                            "road_type": hw,
                            "length_m": dist,
                            "speed_kmh": 40,
                            "width_px": 4.5 if hw == "primary" else 3.0,
                            "is_major": hw == "primary",
                            "accessibility": "open",
                            "flood_risk": 0.15,
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[u_lng, u_lat], [v_lng, v_lat]]
                        }
                    })
                    edge_idx += 1

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"total_nodes": G.number_of_nodes(), "total_edges": len(features)}
        }
        return G, geojson

    def _build_graph(
        self,
        elements: List[Dict[str, Any]],
        nodes_dict: Dict[int, Tuple[float, float]],
        north: float, south: float, east: float, west: float,
        polygon: Optional[List[List[float]]] = None,
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        for el in elements:
            if el.get("type") != "way":
                continue
            tags = el.get("tags", {})
            hw = tags.get("highway")
            if not hw:
                continue
            name = tags.get("name", "Unnamed Road")
            speed = HIGHWAY_SPEEDS.get(hw, 30)
            oneway = tags.get("oneway") in ("yes", "true", "1")
            node_refs = el.get("nodes", [])
            for i in range(len(node_refs) - 1):
                u_id, v_id = node_refs[i], node_refs[i + 1]
                if u_id not in nodes_dict or v_id not in nodes_dict:
                    continue
                u_lat, u_lng = nodes_dict[u_id]
                v_lat, v_lng = nodes_dict[v_id]
                u_in = (south <= u_lat <= north) and (west <= u_lng <= east)
                v_in = (south <= v_lat <= north) and (west <= v_lng <= east)
                if not (u_in or v_in):
                    continue
                if polygon and len(polygon) >= 3:
                    if not (point_in_polygon(u_lat, u_lng, polygon) or point_in_polygon(v_lat, v_lng, polygon)):
                        continue
                for nid, nlat, nlng in [(u_id, u_lat, u_lng), (v_id, v_lat, v_lng)]:
                    if nid not in G:
                        G.add_node(nid, lat=nlat, lng=nlng, type="road_intersection")
                length_m = haversine_distance_m(u_lat, u_lng, v_lat, v_lng)
                if length_m < 0.1:
                    continue
                attrs = {
                    "name": name, "highway": hw, "length": round(length_m, 2),
                    "speed_kmh": speed,
                    "travel_time_sec": round(length_m / (speed * 1000 / 3600), 2),
                    "accessibility": "open", "flood_risk": 0.0,
                    "coordinates": [[u_lat, u_lng], [v_lat, v_lng]],
                    "osm_way_id": el["id"],
                }
                G.add_edge(u_id, v_id, **attrs)
                if not oneway:
                    r = dict(attrs)
                    r["coordinates"] = [[v_lat, v_lng], [u_lat, u_lng]]
                    G.add_edge(v_id, u_id, **r)
        return G

    def build_graph_from_osm_elements(self, elements: List[Dict[str, Any]]) -> nx.DiGraph:
        nodes_dict = {
            el["id"]: (float(el["lat"]), float(el["lon"]))
            for el in elements
            if el.get("type") == "node" and "lat" in el and "lon" in el
        }
        return self._build_graph(elements, nodes_dict, north=90.0, south=-90.0, east=180.0, west=-180.0)

    # Keep graph_to_geojson for backward compat
    def graph_to_geojson(self, G: nx.DiGraph) -> Dict[str, Any]:
        features = []
        seen = set()
        for u, v, data in G.edges(data=True):
            key = tuple(sorted([u, v]))
            if key in seen:
                continue
            seen.add(key)
            coords = [[G.nodes[u]["lng"], G.nodes[u]["lat"]], [G.nodes[v]["lng"], G.nodes[v]["lat"]]]
            features.append({
                "type": "Feature",
                "properties": {
                    "id": f"road-{u}-{v}", "name": data.get("name", ""),
                    "road_type": data.get("highway", "residential"),
                    "length_m": data.get("length", 0.0),
                    "speed_kmh": data.get("speed_kmh", 30),
                    "width_px": ROAD_WIDTHS.get(data.get("highway", ""), 2.5),
                    "is_major": data.get("highway", "") in ("motorway", "trunk", "primary", "secondary"),
                    "accessibility": data.get("accessibility", "open"),
                    "flood_risk": data.get("flood_risk", 0.0),
                },
                "geometry": {"type": "LineString", "coordinates": coords},
            })
        return {"type": "FeatureCollection", "features": features,
                "metadata": {"total_nodes": G.number_of_nodes(), "total_edges": G.number_of_edges()}}

    def _save_cache(self, cache_file: Path, G: nx.DiGraph, geojson: Dict[str, Any]):
        if not geojson.get("features"):
            return
        try:
            nodes_data = {str(n): {"lat": d["lat"], "lng": d["lng"]} for n, d in G.nodes(data=True)}
            edges_data = [
                {"u": str(u), "v": str(v), **{k: val for k, val in data.items() if k != "coordinates"}}
                for u, v, data in G.edges(data=True)
            ]
            with open(cache_file, "w") as f:
                json.dump({"nodes": nodes_data, "edges": edges_data, "geojson": geojson}, f)
        except Exception as e:
            print(f"Failed to save road cache: {e}")

    def _reconstruct_graph(self, cached: Dict[str, Any]) -> nx.DiGraph:
        G = nx.DiGraph()
        for node_id, data in cached.get("nodes", {}).items():
            nid = int(node_id) if node_id.isdigit() else node_id
            G.add_node(nid, lat=data["lat"], lng=data["lng"])
        for edge in cached.get("edges", []):
            u = int(edge["u"]) if str(edge["u"]).isdigit() else edge["u"]
            v = int(edge["v"]) if str(edge["v"]).isdigit() else edge["v"]
            if u in G.nodes and v in G.nodes:
                attrs = {k: val for k, val in edge.items() if k not in ("u", "v")}
                attrs["coordinates"] = [[G.nodes[u]["lat"], G.nodes[u]["lng"]], [G.nodes[v]["lat"], G.nodes[v]["lng"]]]
                G.add_edge(u, v, **attrs)
        return G

    def find_nearest_node(self, G: nx.DiGraph, lat: float, lng: float) -> Optional[Any]:
        if G.number_of_nodes() == 0:
            return None
        best_node, min_dist = None, float("inf")
        for node, data in G.nodes(data=True):
            if data.get("lat") is not None:
                d = haversine_distance_m(lat, lng, data["lat"], data.get("lng", 0))
                if d < min_dist:
                    min_dist = d
                    best_node = node
        return best_node
