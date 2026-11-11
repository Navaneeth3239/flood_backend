"""Complete area layers in Supabase, bound to an exact saved area and boundary.

Independent layer upserts never clear another layer. The foreign key prevents a
late provider response from recreating data after the user deletes its area.
"""
import hashlib
import json
from datetime import datetime, timezone
from uuid import UUID


def area_id_for(payload):
    value = payload.area_id or payload.area_key
    if not value:
        return None
    try:
        return str(UUID(value.removeprefix('dt-area-')))
    except ValueError:
        return None


def boundary_key(polygon, bbox):
    geometry = polygon or [bbox[k] for k in ('north', 'south', 'east', 'west')]
    return hashlib.sha256(json.dumps(geometry, separators=(',', ':')).encode()).hexdigest()


def load_layers(area_id, key):
    if not area_id:
        return {}
    try:
        from lib.db import get_sync_db
        sdb = get_sync_db()
        rows = list(sdb.area_map_layers.find({"area_id": area_id, "boundary_key": key}, {"_id": 0, "layer": 1, "geojson": 1}))
        return {row['layer']: row['geojson'] for row in rows if 'layer' in row and 'geojson' in row}
    except Exception:
        return {}


def save_layer(area_id, key, layer, geojson):
    if not area_id:
        return
    try:
        from lib.db import get_sync_db
        sdb = get_sync_db()
        sdb.area_map_layers.update_one(
            {"area_id": area_id, "boundary_key": key, "layer": layer},
            {"$set": {
                'area_id': area_id, 'boundary_key': key, 'layer': layer,
                'geojson': geojson, 'updated_at': datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True
        )
    except Exception:
        pass
