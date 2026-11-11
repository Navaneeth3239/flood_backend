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

            # 2. Match by polygon
            if payload.polygon and len(payload.polygon) >= 3:
                expected = entry["polygon"]
                if len(payload.polygon) == len(expected) and all(
                    len(point) == 2 and all(abs(v - expected[i][j]) < 1e-4 for j, v in enumerate(point))
                    for i, point in enumerate(payload.polygon)
                ):
                    content = (DIRECTORY / entry["file"]).read_bytes()
                    if entry["file"].endswith(".gz"):
                        content = gzip.decompress(content)
                    return json.loads(content)
    except (OSError, EOFError, ValueError, KeyError, TypeError):
        pass
    return None
