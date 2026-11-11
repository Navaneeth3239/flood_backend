"""Small async Qdrant REST client used by the disaster-intelligence chat.

The API key is read only from the backend environment.  Nothing in this module
is imported by the browser bundle, so the Qdrant credentials never reach users.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


VECTOR_SIZE = 384
DEFAULT_COLLECTION = "flashflood_knowledge"
DEFAULT_KNOWLEDGE = (
    "FlashFloods Digital Twin knowledge. The application combines Cesium 3D terrain, "
    "OpenStreetMap roads and waterways, Microsoft building footprints, a shallow-water "
    "hydrodynamic simulation, Open-Meteo hourly precipitation, and a GNN-Transformer "
    "surface hazard heatmap. The chatbot should explain that the heatmap is an experimental "
    "relative hazard index unless a trained checkpoint is configured. Water flows from high "
    "terrain to low terrain, supports source rise, speed, pause, resume, reset, and displays "
    "spread area and maximum depth. The forecast has twelve hourly frames and the map shares "
    "one selected forecast time between precipitation and heatmap controls. Chat responses "
    "must be concise, location-aware, and recommend elevated evacuation routes while avoiding "
    "low ground, rivers, and blocked roads."
)


def _embedding(text: str) -> list[float]:
    """Create a deterministic, dependency-free lexical embedding.

    This keeps chat startup light.  A future model can replace this function
    without changing the Qdrant collection contract.
    """
    values = [0.0] * VECTOR_SIZE
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % VECTOR_SIZE
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign * (1.0 + min(len(token), 12) / 24.0)
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values] if norm else values


class QdrantKnowledge:
    def __init__(self, url: str | None = None, api_key: str | None = None, collection: str | None = None):
        import os

        self.url = (url or os.getenv("QDRANT_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", "")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
        self._collection_ready = False

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    async def ensure_collection(self) -> None:
        if not self.enabled or self._collection_ready:
            return
        timeout = httpx.Timeout(3.0, connect=1.0)
        async with httpx.AsyncClient(timeout=timeout, headers=self._headers()) as client:
            response = await client.get(f"{self.url}/collections/{self.collection}")
            if response.status_code == 200:
                self._collection_ready = True
                return
            if response.status_code != 404:
                response.raise_for_status()
            created = await client.put(
                f"{self.url}/collections/{self.collection}",
                json={"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}},
            )
            created.raise_for_status()
            self._collection_ready = True

    async def upsert(self, text: str, metadata: dict[str, Any]) -> None:
        if not self.enabled or not text.strip():
            return
        await self.ensure_collection()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"flashflood:{text}"))
        payload = {"text": text, **metadata}
        timeout = httpx.Timeout(4.0, connect=1.0)
        async with httpx.AsyncClient(timeout=timeout, headers=self._headers()) as client:
            response = await client.put(
                f"{self.url}/collections/{self.collection}/points?wait=true",
                json={"points": [{"id": point_id, "vector": _embedding(text), "payload": payload}]},
            )
            response.raise_for_status()

    async def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        if not self.enabled or not query.strip():
            return []
        await self.ensure_collection()
        timeout = httpx.Timeout(4.0, connect=1.0)
        async with httpx.AsyncClient(timeout=timeout, headers=self._headers()) as client:
            response = await client.post(
                f"{self.url}/collections/{self.collection}/points/query",
                json={"query": _embedding(query), "limit": limit, "with_payload": True},
            )
            response.raise_for_status()
            result = response.json().get("result", [])
            if isinstance(result, dict):
                result = result.get("points", result.get("result", []))
            return [item.get("payload", {}) for item in result if item.get("payload")]

    async def seed_defaults(self) -> None:
        await self.upsert(DEFAULT_KNOWLEDGE, {"kind": "application", "area_name": "global"})


knowledge_store = QdrantKnowledge()
