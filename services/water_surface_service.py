"""Bounded, cached OSM loading for 3D water, independent of road extraction."""
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

import os
from services.osm_tile_loader import osm_tile_loader

if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    CACHE = Path('/tmp') / 'cache' / 'water_surfaces'
else:
    CACHE = Path(__file__).resolve().parent.parent / 'cache' / 'water_surfaces'
try:
    CACHE.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
TTL = 86400
ENDPOINTS = ('https://overpass.private.coffee/api/interpreter',
             'https://overpass-api.de/api/interpreter')


async def load_water_elements(south, west, north, east):
    key = hashlib.sha256(f'v1:{south:.5f},{west:.5f},{north:.5f},{east:.5f}'.encode()).hexdigest()
    path = CACHE / f'{key}.json'
    try:
        if time.time() - path.stat().st_mtime < TTL:
            return json.loads(path.read_text())
    except (OSError, ValueError):
        pass

    # Reuse raw OSM geometry already fetched by the map, preserving relations
    # and holes rather than attempting to fill flattened river polylines.
    tiles = osm_tile_loader.tiles_for_bbox(north, south, east, west)
    cached = [osm_tile_loader._read_cache('waterways', tile) for tile in tiles]
    if all(part is not None for part in cached):
        elements = {(el['type'], el['id']): el for part in cached for el in part}
        return {'elements': list(elements.values())}

    box = f'{south},{west},{north},{east}'
    query = f'''[out:json][timeout:10];(
      way["natural"="water"]({box});relation["natural"="water"]({box});
      way["landuse"~"^(reservoir|basin)$"]({box});relation["landuse"~"^(reservoir|basin)$"]({box});
      way["waterway"~"^(river|stream|canal|riverbank)$"]({box});relation["waterway"="riverbank"]({box});
    );out body;>;out skel qt;'''
    async with httpx.AsyncClient(timeout=12) as client:
        async def request(endpoint):
            response = await client.post(endpoint, data={'data': query})
            response.raise_for_status()
            payload = response.json()
            if payload.get('remark') or not isinstance(payload.get('elements'), list):
                raise ValueError('OSM returned incomplete data')
            return payload
        tasks = [asyncio.create_task(request(endpoint)) for endpoint in ENDPOINTS]
        try:
            for future in asyncio.as_completed(tasks):
                try:
                    payload = await future
                except (httpx.HTTPError, ValueError):
                    continue
                try:
                    CACHE.mkdir(parents=True, exist_ok=True)
                    # Atomic replacement prevents a concurrent reader seeing partial JSON.
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', dir=CACHE, delete=False) as f:
                        json.dump(payload, f)
                        temporary = Path(f.name)
                    temporary.replace(path)
                except OSError:
                    pass
                return payload
            raise RuntimeError('OSM servers are unavailable. Please retry.')
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
