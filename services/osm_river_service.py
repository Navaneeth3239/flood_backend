"""OSM River Service: Extracts rivers, streams, canals, and water bodies from OpenStreetMap.
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
from services.osm_geometry import geometry_intersects_polygon, join_rings, expand_polygon

import os

if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    CACHE_DIR = Path("/tmp") / "cache" / "rivers"
else:
    CACHE_DIR = Path(__file__).parent.parent / "cache" / "rivers"
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Water body type → display width (pixels)
WATERWAY_WIDTHS = {
    "river": 7.0, "canal": 6.0, "stream": 3.0, "drain": 2.0,
    "ditch": 1.5, "riverbank": 8.0, "water": 8.0, "lake": 8.0,
    "reservoir": 9.0, "pond": 6.0, "basin": 6.0, "lagoon": 7.0, "oxbow": 5.0,
}

WATER_BODY_TYPES = {"water", "lake", "reservoir", "pond", "basin", "riverbank", "lagoon", "oxbow"}


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


class OSMRiverService:
    def __init__(self):
        pass

    def _cache_key(self, north: float, south: float, east: float, west: float) -> Path:
        key_str = f"river_v4_{north:.6f}_{south:.6f}_{east:.6f}_{west:.6f}"
        hash_val = hashlib.md5(key_str.encode()).hexdigest()
        return CACHE_DIR / f"{hash_val}.json"

    def _load_cached_network(self, cache_file: Path, polygon: Optional[List[List[float]]] = None) -> Optional[Tuple[nx.DiGraph, Dict[str, Any]]]:
        """Loads a cached full water network."""
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
                expanded_poly = expand_polygon(polygon, 0.025)
                scoped_features = []
                for feat in features:
                    if geometry_intersects_polygon(feat.get("geometry", {}), expanded_poly):
                        scoped_features.append(feat)
                features = scoped_features
                geojson = {
                    "type": "FeatureCollection",
                    "features": features,
                    "metadata": {"total_nodes": len(cached.get("nodes", {})), "total_edges": len(features)},
                }
            return self._reconstruct_graph(cached), geojson
        except Exception as e:
            print(f"River cache read error: {e}")
            return None

    async def fetch_waterway_elements_overpass(
        self, north: float, south: float, east: float, west: float
    ) -> List[Dict[str, Any]]:
        """Loads complete waterways through the shared tile cache and queue."""
        elements, _ = await osm_tile_loader.load(
            "waterways",
            north,
            south,
            east,
            west,
            'way["waterway"]{bbox};relation["waterway"]{bbox};way["natural"="water"]{bbox};relation["natural"="water"]{bbox};way["water"]{bbox};way["landuse"~"^(reservoir|basin)$"]{bbox}',
        )
        return elements

    async def get_river_network(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        polygon: Optional[List[List[float]]] = None,
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """Downloads waterways for bbox, builds DiGraph + full-way GeoJSON."""
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

        elements = await self.fetch_waterway_elements_overpass(north, south, east, west)

        # Propagate tags from water relations to their member ways
        relation_way_tags: Dict[int, Dict[str, Any]] = {}
        for el in elements:
            if el.get("type") == "relation":
                rel_tags = el.get("tags", {})
                for m in el.get("members", []):
                    if m.get("type") == "way" and "ref" in m:
                        relation_way_tags[m["ref"]] = rel_tags

        # Build node lookup
        nodes_dict: Dict[int, Tuple[float, float]] = {}
        for el in elements:
            if el.get("type") == "node":
                nodes_dict[el["id"]] = (float(el["lat"]), float(el["lon"]))

        # Generate full-way GeoJSON — all nodes per way → continuous polylines
        features = []
        seen_way_ids = set()
        way_coordinates = {
            element["id"]: [[nodes_dict[node][1], nodes_dict[node][0]] for node in element.get("nodes", []) if node in nodes_dict]
            for element in elements if element.get("type") == "way"
        }
        for relation in elements:
            tags = relation.get("tags", {})
            if relation.get("type") != "relation" or tags.get("type") != "multipolygon":
                continue
            if not (tags.get("natural") == "water" or tags.get("water") or tags.get("landuse") in ("reservoir", "basin") or tags.get("waterway") == "riverbank"):
                continue
            members = [member for member in relation.get("members", []) if member.get("type") == "way"]
            outer = join_rings([way_coordinates.get(member["ref"], []) for member in members if member.get("role") != "inner"])
            inner = join_rings([way_coordinates.get(member["ref"], []) for member in members if member.get("role") == "inner"])
            if not outer:
                continue
            polygons = [[ring, *[hole for hole in inner if point_in_polygon(hole[0][1], hole[0][0], [[point[1], point[0]] for point in ring])]] for ring in outer]
            features.append({
                "type": "Feature",
                "properties": {"id": f"relation-{relation['id']}", "name": tags.get("name", ""), "waterway_type": tags.get("water", "water"), "is_water_body": True, "width_m": 0, "length_m": 0},
                "geometry": {"type": "Polygon" if len(polygons) == 1 else "MultiPolygon", "coordinates": polygons[0] if len(polygons) == 1 else polygons},
            })
            seen_way_ids.update(member["ref"] for member in members)

        for el in elements:
            if el.get("type") != "way" or el["id"] in seen_way_ids:
                continue
            tags = dict(el.get("tags", {}))
            if el["id"] in relation_way_tags:
                for k, v in relation_way_tags[el["id"]].items():
                    tags.setdefault(k, v)

            # Accept waterways, natural water, and water landuse
            is_waterway = bool(tags.get("waterway"))
            is_natural_water = tags.get("natural") == "water"
            has_water_tag = bool(tags.get("water"))
            is_water_landuse = tags.get("landuse") in ("reservoir", "basin")

            if not (is_waterway or is_natural_water or has_water_tag or is_water_landuse):
                continue

            seen_way_ids.add(el["id"])
            node_refs = el.get("nodes", [])

            # Collect all node coordinates in order
            coords = []
            for nid in node_refs:
                if nid in nodes_dict:
                    lat, lon = nodes_dict[nid]
                    coords.append([lon, lat])  # GeoJSON: [lng, lat]

            if len(coords) < 2:
                continue



            # Determine waterway type and classification
            ww_type = (
                tags.get("water")
                or tags.get("waterway")
                or (tags.get("natural") if is_natural_water else None)
                or tags.get("landuse")
                or "stream"
            )
            is_water_body = bool(
                is_natural_water or has_water_tag or is_water_landuse
                or ww_type in WATER_BODY_TYPES
            )
            is_main_river = ww_type in ("river", "canal") and not is_water_body
            name = tags.get("name", "")

            # If it is a closed ring representing a water body, emit as Polygon so 3D water surface can render
            is_closed = len(coords) >= 4 and (
                coords[0] == coords[-1] or
                (abs(coords[0][0] - coords[-1][0]) < 1e-5 and abs(coords[0][1] - coords[-1][1]) < 1e-5)
            )
            if is_water_body and is_closed:
                geom_type = "Polygon"
                geom_coords = [coords]
            else:
                geom_type = "LineString"
                geom_coords = coords

            features.append({
                "type": "Feature",
                "properties": {
                    "id": f"way-{el['id']}",
                    "name": name,
                    "waterway_type": ww_type,
                    "is_water_body": is_water_body,
                    "is_main_river": is_main_river,
                    "width_m": self._waterway_width(tags, ww_type),
                    "length_m": 0.0,  # Computed client-side if needed
                },
                "geometry": {
                    "type": geom_type,
                    "coordinates": geom_coords,
                },
            })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"total_nodes": len(nodes_dict), "total_edges": len(features)},
        }

        # Build NetworkX graph for risk/routing analysis
        G = self._build_graph(elements, nodes_dict, north, south, east, west, polygon)

        self._save_cache(cache_file, G, geojson)
        return G, geojson

    def generate_fallback_rivers(self, north: float, south: float, east: float, west: float, polygon: Optional[List[List[float]]] = None) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        G = nx.DiGraph()
        features = []
        lat_s = max(0.01, north - south)
        lng_s = max(0.01, east - west)

        # Main river meandering through basin
        main_coords = [
            [round(west + 0.85 * lng_s, 6), round(north - 0.10 * lat_s, 6)],
            [round(west + 0.70 * lng_s, 6), round(north - 0.28 * lat_s, 6)],
            [round(west + 0.52 * lng_s, 6), round(south + 0.52 * lat_s, 6)],
            [round(west + 0.38 * lng_s, 6), round(south + 0.35 * lat_s, 6)],
            [round(west + 0.15 * lng_s, 6), round(south + 0.12 * lat_s, 6)],
        ]
        features.append({
            "type": "Feature",
            "properties": {
                "id": "fallback-river-main-1",
                "name": "Main River Drainage Basin",
                "waterway_type": "river",
                "width_m": 28.0,
                "length_m": 4500,
                "is_water_body": False,
                "is_main_river": True,
                "flow_direction": "downstream",
                "flood_susceptibility": 0.75,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": main_coords,
            }
        })

        # Northern tributary
        features.append({
            "type": "Feature",
            "properties": {
                "id": "fallback-river-trib-1",
                "name": "North Catchment Stream",
                "waterway_type": "stream",
                "width_m": 12.0,
                "length_m": 2200,
                "is_water_body": False,
                "is_main_river": False,
                "flow_direction": "downstream",
                "flood_susceptibility": 0.55,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [round(west + 0.50 * lng_s, 6), round(north - 0.05 * lat_s, 6)],
                    [round(west + 0.52 * lng_s, 6), round(south + 0.52 * lat_s, 6)],
                ],
            }
        })

        # Southern tributary
        features.append({
            "type": "Feature",
            "properties": {
                "id": "fallback-river-trib-2",
                "name": "Valley Basin Inflow",
                "waterway_type": "stream",
                "width_m": 14.0,
                "length_m": 1800,
                "is_water_body": False,
                "is_main_river": False,
                "flow_direction": "downstream",
                "flood_susceptibility": 0.60,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [round(west + 0.20 * lng_s, 6), round(south + 0.45 * lat_s, 6)],
                    [round(west + 0.38 * lng_s, 6), round(south + 0.35 * lat_s, 6)],
                ],
            }
        })

        # Catchment reservoir
        features.append({
            "type": "Feature",
            "properties": {
                "id": "fallback-river-reservoir-1",
                "name": "Catchment Reservoir",
                "waterway_type": "reservoir",
                "width_m": 35.0,
                "length_m": 800,
                "is_water_body": True,
                "is_main_river": False,
                "flow_direction": "none",
                "flood_susceptibility": 0.85,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [round(west + 0.52 * lng_s - 0.02 * lng_s, 6), round(south + 0.52 * lat_s - 0.02 * lat_s, 6)],
                    [round(west + 0.52 * lng_s + 0.02 * lng_s, 6), round(south + 0.52 * lat_s + 0.02 * lat_s, 6)],
                ],
            }
        })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"total_nodes": 12, "total_edges": len(features)}
        }
        return G, geojson

    @staticmethod
    def _waterway_width(tags, waterway_type):
        try:
            width = float(str(tags.get("width", "")).split()[0])
            if width > 0:
                return min(width, 500)
        except (ValueError, IndexError):
            pass
        return 18.0 if waterway_type == "river" else 5.0

    def _build_graph(
        self,
        elements: List[Dict[str, Any]],
        nodes_dict: Dict[int, Tuple[float, float]],
        north: float, south: float, east: float, west: float,
        polygon: Optional[List[List[float]]] = None,
    ) -> nx.DiGraph:
        relation_way_tags: Dict[int, Dict[str, Any]] = {}
        for el in elements:
            if el.get("type") == "relation":
                rel_tags = el.get("tags", {})
                for m in el.get("members", []):
                    if m.get("type") == "way" and "ref" in m:
                        relation_way_tags[m["ref"]] = rel_tags

        G = nx.DiGraph()
        for el in elements:
            if el.get("type") != "way":
                continue
            tags = dict(el.get("tags", {}))
            if el["id"] in relation_way_tags:
                for k, v in relation_way_tags[el["id"]].items():
                    tags.setdefault(k, v)

            if not (tags.get("waterway") or tags.get("natural") == "water" or tags.get("water") or tags.get("landuse") in ("reservoir", "basin")):
                continue
            ww_type = tags.get("waterway") or tags.get("water") or tags.get("natural") or tags.get("landuse") or "stream"
            name = tags.get("name", "Unnamed Waterway")
            width_m = WATERWAY_WIDTHS.get(ww_type, 3.0)
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
                        G.add_node(nid, lat=nlat, lng=nlng, type="river_junction")
                length_m = haversine_distance_m(u_lat, u_lng, v_lat, v_lng)
                if length_m < 0.1:
                    continue
                G.add_edge(u_id, v_id, name=name, waterway=ww_type, width_m=width_m,
                           length=round(length_m, 2), osm_way_id=el["id"],
                           is_water_body=ww_type in WATER_BODY_TYPES)
        return G

    def build_graph_from_osm_elements(self, elements: List[Dict[str, Any]]) -> nx.DiGraph:
        nodes_dict = {
            el["id"]: (float(el["lat"]), float(el["lon"]))
            for el in elements
            if el.get("type") == "node" and "lat" in el and "lon" in el
        }
        return self._build_graph(elements, nodes_dict, north=90.0, south=-90.0, east=180.0, west=-180.0)

    def find_distance_to_nearest_river(self, G: nx.DiGraph, lat: float, lng: float) -> float:
        if G.number_of_nodes() == 0:
            return float("inf")
        min_dist = float("inf")
        for node, data in G.nodes(data=True):
            if "lat" in data and "lng" in data:
                d = haversine_distance_m(lat, lng, data["lat"], data["lng"])
                if d < min_dist:
                    min_dist = d
        return min_dist

    # Kept for backward compat
    def graph_to_geojson(self, G: nx.DiGraph) -> Dict[str, Any]:
        features = []
        for u, v, data in G.edges(data=True):
            coords = [[G.nodes[u]["lng"], G.nodes[u]["lat"]], [G.nodes[v]["lng"], G.nodes[v]["lat"]]]
            ww = data.get("waterway", "stream")
            features.append({
                "type": "Feature",
                "properties": {
                    "id": f"river-{u}-{v}", "name": data.get("name", ""),
                    "waterway_type": ww,
                    "is_water_body": data.get("is_water_body", False),
                    "is_main_river": ww in ("river", "canal"),
                    "width_m": data.get("width_m", 3.0),
                    "length_m": data.get("length", 0.0),
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
            print(f"Failed to save river cache: {e}")

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
                G.add_edge(u, v, **attrs)
        return G
