"""Merge mapped buildings with uncapped, real satellite footprints."""
import gzip
import hashlib
import json
import csv
import io
import urllib.request
import os
import threading
try:
    from shapely.geometry import shape, Polygon, box
    from shapely.strtree import STRtree
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False
from services.ms_building_service import bbox_to_quadkeys, TILES_DIR, CACHE_DIR, DATASET_LINKS_URL, load_quadkey_index

BUILDING_VERSION = 3
_index_lock = threading.Lock()
_parts_index = None


def tile_parts(key):
    # Quadkeys at borders contain multiple country/partition files. Never let
    # the last CSV row overwrite earlier rows for the same quadkey.
    global _parts_index
    with _index_lock:
        if _parts_index is None:
            cache = CACHE_DIR / 'dataset_links_parts_v2.json'
            if cache.exists():
                _parts_index = json.loads(cache.read_text())
            else:
                with urllib.request.urlopen(DATASET_LINKS_URL, timeout=30) as response:
                    rows = csv.DictReader(io.StringIO(response.read().decode('utf-8')))
                    index = {}
                    for row in rows:
                        index.setdefault(row['QuadKey'], []).append({'url':row['Url'], 'loc':row['Location']})
                cache.write_text(json.dumps(index))
                _parts_index = index
    parts = _parts_index.get(key, [])
    legacy = load_quadkey_index().get(key, {})
    paths = []
    for part in parts:
        legacy_path = TILES_DIR / f'{key}.csv.gz'
        if part['url'] == legacy.get('url') and legacy_path.exists():
            paths.append(legacy_path)
            continue
        digest = hashlib.sha256(part['url'].encode()).hexdigest()[:16]
        path = TILES_DIR / f'{key}-{digest}.csv.gz'
        if not path.exists():
            temporary = path.with_suffix(f'.{threading.get_ident()}.tmp')
            try:
                with urllib.request.urlopen(part['url'], timeout=60) as response:
                    with temporary.open('wb') as target:
                        while chunk := response.read(1024 * 1024):
                            target.write(chunk)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        paths.append(path)
    return paths


def enrich_buildings(mapped, bbox, polygon):
    if not _HAS_SHAPELY:
        return mapped
    area = Polygon([(p[1], p[0]) for p in polygon]) if polygon else box(bbox['west'], bbox['south'], bbox['east'], bbox['north'])
    if not area.is_valid:
        area = area.buffer(0)
    features = list(mapped.get('features', []))
    osm_shapes = [shape(f['geometry']) for f in features]
    osm_tree = STRtree(osm_shapes)
    seen = {hashlib.sha256(json.dumps(f['geometry'], sort_keys=True).encode()).hexdigest() for f in features}
    for key in bbox_to_quadkeys(bbox['south'], bbox['west'], bbox['north'], bbox['east']):
        for path in tile_parts(key):
            with gzip.open(path, 'rt', encoding='utf-8') as source:
                for line in source:
                    feature = json.loads(line)
                    geometry = feature.get('geometry') or {}
                    if geometry.get('type') not in ('Polygon', 'MultiPolygon'):
                        continue
                    # Fast bounds rejection before constructing a geometry object.
                    parts = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
                    points = [p for part in parts for p in part[0]]
                    if not points or max(p[0] for p in points) < bbox['west'] or min(p[0] for p in points) > bbox['east'] or max(p[1] for p in points) < bbox['south'] or min(p[1] for p in points) > bbox['north']:
                        continue
                    footprint = shape(geometry)
                    if not footprint.is_valid:
                        footprint = footprint.buffer(0)
                    if footprint.is_empty or not footprint.intersects(area):
                        continue
                    digest = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
                    if digest in seen:
                        continue
                    # Keep OSM's surveyed footprint when both sources map one house.
                    duplicate = False
                    for index in osm_tree.query(footprint):
                        other = osm_shapes[index]
                        if other.is_valid and min(other.area, footprint.area) > 0 and footprint.intersection(other).area / min(other.area, footprint.area) >= 0.5:
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    seen.add(digest)
                    properties = feature.get('properties') or {}
                    height = properties.get('height')
                    try:
                        height = float(height)
                    except (ValueError, TypeError):
                        height = 6.0
                    if height <= 0:
                        height = 6.0
                    features.append({'type':'Feature', 'id':f'ms-{digest[:20]}', 'geometry':geometry,
                        'properties':{**properties, 'id':f'ms-{digest[:20]}', 'height_m':height, 'source':'Microsoft Global ML Building Footprints'}})
    return {'type':'FeatureCollection', 'features':features, 'metadata':{'building_version':BUILDING_VERSION, 'total_buildings':len(features), 'sources':['OpenStreetMap','Microsoft Global ML Building Footprints']}}
