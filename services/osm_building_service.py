"""Complete, tiled OpenStreetMap building footprints for the Digital Twin."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.location_service import point_in_polygon
from services.osm_tile_loader import osm_tile_loader
from services.osm_geometry import geometry_intersects_polygon, join_rings


def _numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def _height(tags: Dict[str, Any]) -> tuple[float, str]:
    height = _numeric(tags.get("height") or tags.get("building:height"))
    if height:
        return height, "height_tag"
    levels = _numeric(tags.get("building:levels") or tags.get("levels"))
    if levels:
        return levels * 3.0, "levels_tag"
    return 6.0, "default_6m"


def _closed(coords: list[list[float]]) -> list[list[float]]:
    if len(coords) >= 3 and coords[0] != coords[-1]:
        return [*coords, coords[0]]
    return coords


def _join_rings(ways: list[list[list[float]]]) -> list[list[list[float]]]:
    """Join relation member ways by shared endpoints into closed boundary rings."""
    pending = [_closed(way) for way in ways if len(way) >= 2]
    rings: list[list[list[float]]] = []
    while pending:
        chain = pending.pop()
        changed = True
        while changed and chain[0] != chain[-1]:
            changed = False
            for index, candidate in enumerate(pending):
                if chain[-1] == candidate[0]:
                    chain.extend(candidate[1:])
                elif chain[-1] == candidate[-1]:
                    chain.extend(list(reversed(candidate[:-1])))
                elif chain[0] == candidate[-1]:
                    chain = candidate[:-1] + chain
                elif chain[0] == candidate[0]:
                    chain = list(reversed(candidate[1:])) + chain
                else:
                    continue
                pending.pop(index)
                changed = True
                break
        closed = _closed(chain)
        if len(closed) >= 4 and closed[0] == closed[-1]:
            rings.append(closed)
    return rings


class OSMBuildingService:
    async def get_buildings(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        polygon: Optional[List[List[float]]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        elements, load_status = await osm_tile_loader.load(
            "buildings",
            north,
            south,
            east,
            west,
            'way["building"]{bbox};relation["building"]{bbox}',
        )
        nodes = {
            item["id"]: [float(item["lon"]), float(item["lat"])]
            for item in elements
            if item.get("type") == "node" and "lat" in item and "lon" in item
        }
        way_coords: dict[int, list[list[float]]] = {}
        way_tags: dict[int, dict[str, Any]] = {}
        for item in elements:
            if item.get("type") != "way":
                continue
            coords = [nodes[node_id] for node_id in item.get("nodes", []) if node_id in nodes]
            if len(coords) >= 2:
                way_coords[item["id"]] = coords
                way_tags[item["id"]] = dict(item.get("tags") or {})

        features: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()

        def include(geometry: dict[str, Any], tags: dict[str, Any], osm_type: str, osm_id: int) -> None:
            rings = geometry.get("coordinates", [])
            flat = (
                [point for ring in rings for point in ring]
                if geometry.get("type") == "Polygon"
                else [point for polygon_rings in rings for ring in polygon_rings for point in ring]
            )
            if not geometry_intersects_polygon(geometry, polygon):
                return
            height_m, source = _height(tags)
            features.append({
                "type": "Feature",
                "properties": {
                    "id": f"{osm_type}-{osm_id}",
                    "osm_type": osm_type,
                    "osm_id": osm_id,
                    "building": tags.get("building", "yes"),
                    "name": tags.get("name", ""),
                    "height_m": height_m,
                    "height_source": source,
                },
                "geometry": geometry,
            })

        for way_id, coords in way_coords.items():
            tags = way_tags[way_id]
            if not tags.get("building") or ("way", way_id) in seen or len(coords) < 4 or coords[0] != coords[-1]:
                continue
            seen.add(("way", way_id))
            include({"type": "Polygon", "coordinates": [coords]}, tags, "way", way_id)

        # Relations are typically multipolygon buildings. Their member ways are
        # already present through the recursive Overpass query above.
        for relation in elements:
            if relation.get("type") != "relation" or not (relation.get("tags") or {}).get("building"):
                continue
            relation_id = relation.get("id")
            if not isinstance(relation_id, int) or ("relation", relation_id) in seen:
                continue
            members = relation.get("members") or []
            outer_ways = [way_coords[member["ref"]] for member in members if member.get("type") == "way" and member.get("role", "outer") in ("", "outer") and member.get("ref") in way_coords]
            inner_ways = [way_coords[member["ref"]] for member in members if member.get("type") == "way" and member.get("role") == "inner" and member.get("ref") in way_coords]
            outers, inners = join_rings(outer_ways), join_rings(inner_ways)
            if not outers:
                continue
            seen.add(("relation", relation_id))
            # Keep each outer as a polygon; holes are attached to the first
            # outer when available (the common OSM building relation shape).
            polygons = [[outer, *[hole for hole in inners if point_in_polygon(hole[0][1], hole[0][0], [[point[1], point[0]] for point in outer])]] for outer in outers]
            geometry = {"type": "Polygon", "coordinates": polygons[0]} if len(polygons) == 1 else {"type": "MultiPolygon", "coordinates": polygons}
            include(geometry, dict(relation.get("tags") or {}), "relation", relation_id)

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"total_buildings": len(features)},
        }, load_status
