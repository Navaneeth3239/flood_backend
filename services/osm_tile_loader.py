"""Reliable, tile-cached Overpass loading shared by the road and water services."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import httpx


import os

if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    CACHE_ROOT = Path("/tmp") / "cache" / "osm_tiles"
else:
    CACHE_ROOT = Path(__file__).parent.parent / "cache" / "osm_tiles"
try:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

CACHE_TTL_SECONDS = 24 * 60 * 60
MAX_CONCURRENCY = 6
# Keep a new-area request responsive. Endpoint failover is still used, but a
# dead Overpass mirror must not hold the Digital Twin risk panel for minutes.
REQUEST_TIMEOUT_SECONDS = 8.0
MAX_ATTEMPTS = 2

ENDPOINTS = (
    # Reliable Overpass mirrors
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)


class OSMTileLoadError(RuntimeError):
    def __init__(self, failures: list[str]):
        super().__init__("; ".join(failures))
        self.failures = failures


@dataclass(frozen=True)
class Tile:
    north: float
    south: float
    east: float
    west: float

    @property
    def key(self) -> str:
        return f"{self.north:.5f}:{self.south:.5f}:{self.east:.5f}:{self.west:.5f}"

    def as_dict(self) -> dict[str, float]:
        return {"north": self.north, "south": self.south, "east": self.east, "west": self.west}


class OSMTileLoader:
    """Downloads bounded OSM queries with cache, retries and endpoint failover."""

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self._unhealthy_until: dict[str, float] = {}
        self._inflight: dict[str, asyncio.Future] = {}
        self._map_tasks: dict[str, asyncio.Task] = {}
        self._map_semaphore = asyncio.Semaphore(2)

    @staticmethod
    def tiles_for_bbox(north: float, south: float, east: float, west: float) -> list[Tile]:
        lat_span, lng_span = abs(north - south), abs(east - west)
        target_span = max(0.15, max(lat_span, lng_span) / 2.0)
        rows = min(2, max(1, math.ceil(lat_span / target_span)))
        cols = min(2, max(1, math.ceil(lng_span / target_span)))
        lat_step = lat_span / rows
        lng_step = lng_span / cols
        return [
            Tile(
                north=min(north, south + (row + 1) * lat_step),
                south=south + row * lat_step,
                east=min(east, west + (col + 1) * lng_step),
                west=west + col * lng_step,
            )
            for row in range(rows)
            for col in range(cols)
        ]

    @staticmethod
    def _cache_path(dataset: str, tile: Tile) -> Path:
        digest = hashlib.sha256(f"v2:{dataset}:{tile.key}".encode()).hexdigest()
        p = CACHE_ROOT / dataset
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p / f"{digest}.json"

    def _read_cache(self, dataset: str, tile: Tile) -> list[dict[str, Any]] | None:
        path = self._cache_path(dataset, tile)
        try:
            if not path.exists() or time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
                return None
            data = json.loads(path.read_text())
            elements = data.get("elements")
            return elements if isinstance(elements, list) else None
        except (OSError, ValueError, TypeError):
            return None

    def _write_cache(self, dataset: str, tile: Tile, elements: list[dict[str, Any]]) -> None:
        path = self._cache_path(dataset, tile)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"elements": elements, "cached_at": time.time()}))
        except OSError:
            pass

    def _endpoint_order(self, attempt: int) -> Iterable[str]:
        now = time.monotonic()
        healthy = [url for url in ENDPOINTS if self._unhealthy_until.get(url, 0) <= now]
        candidates = healthy or list(ENDPOINTS)
        offset = attempt % len(candidates)
        return candidates[offset:] + candidates[:offset]

    async def _request_tile(self, dataset: str, tile: Tile, query: str) -> tuple[list[dict[str, Any]], bool]:
        cached = self._read_cache(dataset, tile)
        if cached is not None:
            return cached, True

        key = f"{dataset}:{tile.key}"
        if key in self._inflight:
            # Another coroutine is already fetching this exact tile — await its result
            return await asyncio.shield(self._inflight[key])

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        future.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
        self._inflight[key] = future

        try:
            res = await self._fetch_tile_from_endpoints(dataset, tile, query)
            if not future.done():
                future.set_result(res)
            return res
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)

    async def _request_map_tile(self, tile: Tile) -> list[dict[str, Any]]:
        """Small-area fallback using the official OSM map API, shared by layers.

        https://wiki.openstreetmap.org/wiki/API_v0.6#Retrieving_map_data_by_bounding_box
        Keep requests small and cached; larger queries remain on Overpass.
        """
        cached = self._read_cache("map", tile)
        if cached is not None:
            return cached
        task = self._map_tasks.get(tile.key)
        if task is None:
            task = asyncio.create_task(self._fetch_map_tile(tile))
            self._map_tasks[tile.key] = task
            def done(completed):
                self._map_tasks.pop(tile.key, None)
                if not completed.cancelled():
                    completed.exception()
            task.add_done_callback(done)
        return await asyncio.shield(task)

    async def _fetch_map_tile(self, tile: Tile) -> list[dict[str, Any]]:
        async with self._map_semaphore:
            async with httpx.AsyncClient(timeout=httpx.Timeout(35, connect=5), headers={"User-Agent": "FlashFloodDigitalTwin/1.0", "Accept": "application/json"}) as client:
                response = await client.get("https://api.openstreetmap.org/api/0.6/map.json", params={
                    "bbox": f"{tile.west},{tile.south},{tile.east},{tile.north}",
                })
                response.raise_for_status()
                elements = response.json().get("elements")
                if not isinstance(elements, list):
                    raise ValueError("OSM map response has no elements")
                unique = {(el["type"], el["id"]): el for el in elements}
                # The map API returns full ways but may omit relation members
                # outside the bbox. Fetch relevant full relations before caching.
                for el in elements:
                    tags = el.get("tags") or {}
                    relevant = tags.get("building") or tags.get("highway") or tags.get("waterway") or tags.get("natural") == "water" or tags.get("water") or tags.get("landuse") in {"reservoir", "basin"}
                    if el.get("type") != "relation" or not relevant:
                        continue
                    if any((m["type"], m["ref"]) not in unique for m in el.get("members", [])):
                        full = await client.get(f"https://api.openstreetmap.org/api/0.6/relation/{el['id']}/full.json")
                        full.raise_for_status()
                        members = full.json().get("elements")
                        if not isinstance(members, list):
                            raise ValueError("Incomplete OSM relation")
                        unique.update({(m["type"], m["id"]): m for m in members})
                result = list(unique.values())
                if any(any(("node", node) not in unique for node in el.get("nodes", [])) for el in result if el["type"] == "way"):
                    raise ValueError("Incomplete OSM way geometry")
                self._write_cache("map", tile, result)
                return result

    async def _fetch_tile_from_endpoints(self, dataset: str, tile: Tile, query: str) -> tuple[list[dict[str, Any]], bool]:
        headers = {
            "User-Agent": "Mozilla/5.0 FlashFloodDigitalTwin/1.0 (contact: admin@ein.gov.in)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
        errors: list[str] = []
        map_attempted = False

        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                await asyncio.sleep(2 ** (attempt - 1))
            endpoints = list(self._endpoint_order(attempt))
            for idx, endpoint in enumerate(endpoints):
                try:
                    async with self._semaphore:
                        async with httpx.AsyncClient(timeout=httpx.Timeout(70.0 if dataset == "buildings" else REQUEST_TIMEOUT_SECONDS, connect=5.0), headers=headers) as client:
                            response = await client.post(endpoint, data={"data": query})
                    if response.status_code in (403, 406, 429, 500, 502, 503, 504):
                        self._unhealthy_until[endpoint] = time.monotonic() + 300
                        errors.append(f"{tile.key} {response.status_code} {endpoint}")
                        raise httpx.HTTPStatusError(f"Overpass returned status {response.status_code}", request=response.request, response=response)
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("remark"):
                        raise ValueError(f"Incomplete Overpass response: {payload['remark']}")
                    elements = payload.get("elements")
                    if not isinstance(elements, list):
                        raise ValueError("response did not contain an elements list")
                    # Authoritative OSM response received: cache and return immediately
                    self._write_cache(dataset, tile, elements)
                    return elements, False
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                    self._unhealthy_until[endpoint] = time.monotonic() + 30
                    errors.append(f"{tile.key} {endpoint}: {exc}")
                    if not map_attempted and max(tile.north - tile.south, tile.east - tile.west) <= 0.20:
                        map_attempted = True
                        try:
                            elements = await self._request_map_tile(tile)
                            self._write_cache(dataset, tile, elements)
                            return elements, False
                        except (httpx.HTTPError, ValueError) as fallback_error:
                            errors.append(f"{tile.key} OSM map fallback: {fallback_error}")

        raise OSMTileLoadError(errors or [f"{tile.key} failed without a response"])

    async def load(
        self,
        dataset: str,
        north: float,
        south: float,
        east: float,
        west: float,
        selectors: str,
        on_progress: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Load every tile or raise with exact failed tile bounds.

        ``on_progress`` is deliberately optional: streaming endpoints can publish
        tile completion immediately, while ordinary API calls still get a fully
        validated, merged result.
        """
        tiles = self.tiles_for_bbox(north, south, east, west)

        def make_query(tile: Tile) -> str:
            bbox = f"({tile.south},{tile.west},{tile.north},{tile.east})"
            # Each statement inside an Overpass union must end in a semicolon.
            # Keeping it here avoids a subtle parse error on the final selector.
            selected = selectors.format(bbox=bbox).rstrip(";") + ";"
            query_timeout = 60 if dataset == "buildings" else 20
            return f"[out:json][timeout:{query_timeout}];({selected});out body;>;out skel qt;"

        async def load_one(tile: Tile) -> tuple[Tile, list[dict[str, Any]] | None, bool, Exception | None]:
            try:
                elements, from_cache = await self._request_tile(dataset, tile, make_query(tile))
                return tile, elements, from_cache, None
            except Exception as exc:
                return tile, None, False, exc

        tasks = [asyncio.create_task(load_one(tile)) for tile in tiles]
        results: list[list[dict[str, Any]]] = []
        failures: list[dict[str, Any]] = []
        completed = 0
        cached_tiles = 0
        try:
            for task in asyncio.as_completed(tasks):
                tile, elements, from_cache, error = await task
                if error is None and elements is not None:
                    results.append(elements)
                    cached_tiles += int(from_cache)
                    state = "success"
                else:
                    failures.append({"tile": tile.as_dict(), "error": str(error)})
                    state = "failed"
                completed += 1
                if on_progress:
                    update = {
                        "dataset": dataset,
                        "completed_tiles": completed,
                        "total_tiles": len(tiles),
                        "cached_tiles": cached_tiles,
                        "state": state,
                        "tile": tile.as_dict(),
                    }
                    outcome = on_progress(update)
                    if asyncio.iscoroutine(outcome):
                        await outcome
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # Never cache or report an incomplete network as a successful empty map.

        if failures:
            raise OSMTileLoadError([failure["error"] for failure in failures])

        unique: dict[tuple[str, int], dict[str, Any]] = {}
        for elements in results:
            for element in elements:
                element_type, osm_id = element.get("type"), element.get("id")
                if element_type in {"node", "way", "relation"} and isinstance(osm_id, int):
                    key = (element_type, osm_id)
                    previous = unique.get(key, {})
                    unique[key] = {**previous, **element, "tags": {**previous.get("tags", {}), **element.get("tags", {})}}
        loaded_count = len(tiles) - len(failures)
        return list(unique.values()), {
            "dataset": dataset,
            "total_tiles": len(tiles),
            "loaded_tiles": loaded_count,
            "cached_tiles": cached_tiles,
            "failed_tiles": failures,
            "complete": len(failures) == 0,
        }


osm_tile_loader = OSMTileLoader()
