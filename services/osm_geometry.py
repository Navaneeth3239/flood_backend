from services.location_service import point_in_polygon


def join_rings(ways):
    pending = [list(way) for way in ways if len(way) >= 2]
    rings = []
    while pending:
        chain = pending.pop()
        while chain[0] != chain[-1]:
            for index, candidate in enumerate(pending):
                if chain[-1] == candidate[0]:
                    chain.extend(candidate[1:])
                elif chain[-1] == candidate[-1]:
                    chain.extend(candidate[-2::-1])
                elif chain[0] == candidate[-1]:
                    chain = candidate[:-1] + chain
                elif chain[0] == candidate[0]:
                    chain = candidate[:0:-1] + chain
                else:
                    continue
                pending.pop(index)
                break
            else:
                break
        if len(chain) >= 4 and chain[0] == chain[-1]:
            rings.append(chain)
    return rings


def expand_polygon(polygon, buffer: float = 0.025):
    """Expands a [lat, lng] polygon outward from its centroid by buffer degrees (~2.5km for 0.025)."""
    if not polygon or len(polygon) < 3:
        return polygon
    c_lat = sum(p[0] for p in polygon) / len(polygon)
    c_lng = sum(p[1] for p in polygon) / len(polygon)
    expanded = []
    for lat, lng in polygon:
        d_lat = lat - c_lat
        d_lng = lng - c_lng
        dist = (d_lat**2 + d_lng**2)**0.5
        if dist > 1e-6:
            expanded.append([lat + (d_lat / dist) * buffer, lng + (d_lng / dist) * buffer])
        else:
            expanded.append([lat, lng])
    return expanded


def geometry_intersects_polygon(geometry, polygon):
    if not polygon or len(polygon) < 3:
        return True
    selection = [[longitude, latitude] for latitude, longitude in polygon]
    coordinates = geometry.get("coordinates", [])
    kind = geometry.get("type")
    parts = [coordinates] if kind in ("LineString", "Polygon") else coordinates
    if kind in ("LineString", "MultiLineString"):
        parts = [[line] for line in parts]

    def cross(start, end, point):
        return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])

    def intersects(start, end, first, second):
        if max(start[0], end[0]) < min(first[0], second[0]) or max(first[0], second[0]) < min(start[0], end[0]):
            return False
        if max(start[1], end[1]) < min(first[1], second[1]) or max(first[1], second[1]) < min(start[1], end[1]):
            return False
        return cross(start, end, first) * cross(start, end, second) <= 0 and cross(first, second, start) * cross(first, second, end) <= 0

    for rings in parts:
        if not rings or not rings[0]:
            continue
        for ring in rings:
            if any(point_in_polygon(latitude, longitude, polygon) for longitude, latitude in ring):
                return True
            for index in range(1, len(ring)):
                if any(intersects(ring[index - 1], ring[index], point, selection[(edge + 1) % len(selection)]) for edge, point in enumerate(selection)):
                    return True
        if kind in ("Polygon", "MultiPolygon"):
            for longitude, latitude in selection:
                if point_in_polygon(latitude, longitude, [[point[1], point[0]] for point in rings[0]]) and not any(
                    point_in_polygon(latitude, longitude, [[point[1], point[0]] for point in hole]) for hole in rings[1:]
                ):
                    return True
    return False
