"""Read exported geometry without requiring Supabase or a map provider."""
import gzip
import hashlib
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "prebaked_zones"


def load_bundled_area(payload):
    try:
        entries = json.loads((DIRECTORY / "manifest.json").read_text())
        area_id = (payload.area_id or payload.area_key or "").removeprefix("dt-area-")
        for entry in entries:
            # 1. Match by ID
            if area_id and area_id == entry["area_id"]:
                content = (DIRECTORY / entry["file"]).read_bytes()
                if entry["file"].endswith(".gz"):
                    content = gzip.decompress(content)
                return json.loads(content)

            # 2. Match by polygon (exact match or >40% bbox overlap)
            if payload.polygon and len(payload.polygon) >= 3:
                expected = entry.get("polygon", [])
                if len(expected) >= 3:
                    if len(payload.polygon) == len(expected) and all(
                        len(point) == 2 and all(abs(v - expected[i][j]) < 1e-4 for j, v in enumerate(point))
                        for i, point in enumerate(payload.polygon)
                    ):
                        content = (DIRECTORY / entry["file"]).read_bytes()
                        if entry["file"].endswith(".gz"):
                            content = gzip.decompress(content)
                        return json.loads(content)

                    # Bounding box overlap match
                    p_lats = [p[0] for p in payload.polygon]
                    p_lngs = [p[1] for p in payload.polygon]
                    e_lats = [p[0] for p in expected]
                    e_lngs = [p[1] for p in expected]
                    p_min_lat, p_max_lat = min(p_lats), max(p_lats)
                    p_min_lng, p_max_lng = min(p_lngs), max(p_lngs)
                    e_min_lat, e_max_lat = min(e_lats), max(e_lats)
                    e_min_lng, e_max_lng = min(e_lngs), max(e_lngs)
                    lat_overlap = max(0.0, min(p_max_lat, e_max_lat) - max(p_min_lat, e_min_lat))
                    lng_overlap = max(0.0, min(p_max_lng, e_max_lng) - max(p_min_lng, e_min_lng))
                    overlap_area = lat_overlap * lng_overlap
                    p_area = (p_max_lat - p_min_lat) * (p_max_lng - p_min_lng)
                    if p_area > 0 and (overlap_area / p_area) > 0.4:
                        content = (DIRECTORY / entry["file"]).read_bytes()
                        if entry["file"].endswith(".gz"):
                            content = gzip.decompress(content)
                        return json.loads(content)
    except (OSError, EOFError, ValueError, KeyError, TypeError):
        pass
    return None
