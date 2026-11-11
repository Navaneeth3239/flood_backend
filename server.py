from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Body, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
import uuid
from datetime import datetime
import asyncio


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
from lib.db import db


# Startup runs before the yield, shutdown after it. Add your own setup/teardown here.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # In Vercel serverless environment, background daemon threads and async tasks
    # are not supported and would freeze lambda execution.
    if os.environ.get("VERCEL"):
        yield
        return

    async def _seed_chat_knowledge():
        try:
            from services.qdrant_service import knowledge_store
            await knowledge_store.seed_defaults()
        except Exception as exc:
            logger.warning("Qdrant knowledge seed skipped (non-fatal): %s", exc)

    asyncio.create_task(_seed_chat_knowledge())
    # Pre-warm GEE tile cache in a background daemon thread on startup.
    # Daemon thread ensures Uvicorn reload / shutdown is never blocked.
    def _warm_gee_target():
        try:
            from routers.gee import _get_elevation_tiles, _ensure_gee
            _ensure_gee()
            _get_elevation_tiles()
            logger.info("GEE STM 30 / elevation tile cache pre-warmed successfully.")
        except Exception as exc:
            logger.warning(f"GEE cache pre-warm failed (non-fatal): {exc}")

    import threading
    threading.Thread(target=_warm_gee_target, daemon=True).start()
    yield


# Create the main app without a prefix
app = FastAPI(title="NEXGI API", lifespan=lifespan)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.model_dump())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Feature routers — mounted onto api_router so everything stays under /api
from routers.admin import router as admin_router  # noqa: E402
from routers.alerts import router as alerts_router  # noqa: E402
from routers.auth import router as auth_router  # noqa: E402
from routers.intelligence import router as intelligence_router  # noqa: E402
from routers.network import router as network_router  # noqa: E402
from routers.satellite import router as satellite_router  # noqa: E402

from routers.gee import router as gee_router  # noqa: E402
from routers.digital_twin import router as digital_twin_router  # noqa: E402
from routers.routing_and_rivers import router as routing_and_rivers_router  # noqa: E402
from routers.buildings import router as buildings_router  # noqa: E402
from routers.chat import router as chat_router  # noqa: E402
from routers.external_sensors import router as external_sensors_router  # noqa: E402
from routers.simulation_pg import router as simulation_pg_router  # noqa: E402
from routers.weather import router as weather_router  # noqa: E402
from routers.telephony import router as telephony_router  # noqa: E402
from routers.towers import router as towers_router  # noqa: E402

api_router.include_router(auth_router)
api_router.include_router(network_router)
api_router.include_router(alerts_router)
api_router.include_router(intelligence_router)
api_router.include_router(admin_router)
api_router.include_router(satellite_router)
api_router.include_router(gee_router)
api_router.include_router(digital_twin_router)
api_router.include_router(routing_and_rivers_router)
api_router.include_router(buildings_router)
api_router.include_router(chat_router)
api_router.include_router(external_sensors_router)
api_router.include_router(simulation_pg_router)
api_router.include_router(weather_router)
api_router.include_router(telephony_router)
api_router.include_router(towers_router)



# Include the router in the main app (with /api prefix)
app.include_router(api_router)

# Also mount feature routers directly on app (without prefix)
# This guarantees that if Vercel strips /api or sends /auth/..., it matches 100%
app.include_router(auth_router)
app.include_router(network_router)
app.include_router(alerts_router)
app.include_router(intelligence_router)
app.include_router(admin_router)
app.include_router(satellite_router)
app.include_router(gee_router)
app.include_router(digital_twin_router)
app.include_router(routing_and_rivers_router)
app.include_router(buildings_router)
app.include_router(chat_router)
app.include_router(external_sensors_router)
app.include_router(simulation_pg_router)
app.include_router(weather_router)
app.include_router(telephony_router)
app.include_router(towers_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Monitored Areas endpoint ────────────────────────────────────────────────
@api_router.get("/areas")
@app.get("/areas")
async def list_custom_areas():
    """Returns custom monitored areas from MongoDB, with fallback default areas."""
    try:
        from lib.db import db
        docs = await db.custom_areas.find({}, {"_id": 0}).to_list(100)
        if docs:
            return docs
    except Exception as e:
        logger.warning(f"Failed to fetch areas from MongoDB: {e}")
    # Return default monitored areas if empty or query failed
    return [
        {
            "id": "b3e2c3ce-e6b0-4c37-8f85-5595d23f5d40",
            "name": "Monitored Area 1",
            "district": "Western Himalayas",
            "area_type": "River Basin",
            "risk_category": "High",
            "priority": "Critical (Real-time)",
            "description": "High-risk flash flood and landslide monitored area",
            "lat": 31.039,
            "lng": 78.8938,
            "shape": "Polygon:[[30.948169143221595,78.71978759765625],[31.130844260159883,78.7218475341797],[31.129374846459353,79.06070709228517],[30.947580251654042,79.07272338867189]]",
            "created_at": "2026-09-27T13:31:51.998915+00:00"
        },
        {
            "id": "0004ccb1-ebc6-4c8c-90f6-06ebe52a2e42",
            "name": "Monitored Area 2",
            "district": "Western Himalayas",
            "area_type": "River Basin",
            "risk_category": "Medium",
            "priority": "Normal (Hourly)",
            "description": "Extended catchment monitored boundary",
            "lat": 31.0649,
            "lng": 78.8176,
            "shape": "Polygon:[[31.218673801364655,78.6126708984375],[30.9128292266562,78.61129760742188],[30.9187201197222,79.02877807617189],[31.209277877460135,79.01779174804689]]",
            "created_at": "2026-09-18T10:53:33.348793+00:00"
        }
    ]


@api_router.post("/areas")
@app.post("/areas")
async def create_custom_area(payload: dict = Body(...)):
    """Creates a custom monitored area in MongoDB."""
    try:
        from lib.db import db
        import uuid
        data = dict(payload)
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid.uuid4())
        await db.custom_areas.insert_one(data)
        doc = await db.custom_areas.find_one({"id": data["id"]}, {"_id": 0})
        return doc or data
    except Exception as e:
        logger.error(f"Error creating area: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/areas/{area_id}")
@app.delete("/areas/{area_id}")
async def delete_custom_area(area_id: str):
    """Deletes a custom monitored area from MongoDB."""
    try:
        from lib.db import db
        await db.custom_areas.delete_one({"id": area_id})
        try:
            await db.area_map_layers.delete_many({"area_id": area_id})
        except Exception:
            pass
        return {"status": "success", "deleted": area_id}
    except Exception as e:
        logger.error(f"Error deleting area {area_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health endpoints ───────────────────────────────────────────────────────────
@api_router.get("/health")
async def api_health_check():
    """Lightweight liveness probe under /api/health."""
    return {"status": "online", "system": "NEXGI"}


@app.get("/health")
async def root_health_check():
    """Lightweight liveness probe under /health."""
    return {"status": "online", "system": "NEXGI"}


# ── Local development server ───────────────────────────────────────────────────
# This block is intentionally excluded from Vercel serverless execution.
# Vercel imports `app` directly via api/index.py and manages the server lifecycle.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
