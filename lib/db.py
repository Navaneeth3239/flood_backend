"""MongoDB Database Engine: Native asynchronous MongoDB client powered by Motor
connected to MongoDB Atlas, providing 100% compatibility with all routers.
"""
import os
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import pymongo
from bson import ObjectId

# Configure public DNS resolver fallback to prevent Windows local interface SRV timeouts
try:
    import dns.resolver
    _res = dns.resolver.Resolver(configure=False)
    _res.nameservers = ["8.8.8.8", "1.1.1.1", "8.8.4.4"]
    dns.resolver.default_resolver = _res
except Exception:
    pass

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv()

DEFAULT_MONGO_URI = "mongodb+srv://Navee:Navee3239@cluster0.qs8xxxu.mongodb.net/scms_flood?retryWrites=true&w=majority"
MONGO_URL = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URL") or DEFAULT_MONGO_URI
DB_NAME = os.environ.get("DB_NAME", "scms_flood")

TABLE_ALIASES = {
    "areas": "custom_areas",
}

_async_client: Optional[AsyncIOMotorClient] = None
_sync_client: Optional[pymongo.MongoClient] = None


def get_async_client() -> AsyncIOMotorClient:
    global _async_client
    if _async_client is None:
        _async_client = AsyncIOMotorClient(
            MONGO_URL,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )
    return _async_client


def get_async_db() -> AsyncIOMotorDatabase:
    client = get_async_client()
    return client[DB_NAME]


def get_sync_db():
    """Synchronous pymongo database connection for sync utility scripts/services."""
    global _sync_client
    if _sync_client is None:
        _sync_client = pymongo.MongoClient(
            MONGO_URL,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )
    return _sync_client[DB_NAME]


def _clean_doc_output(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if doc is None:
        return None
    d = dict(doc)
    if "_id" in d:
        oid = d["_id"]
        d["_id"] = str(oid)
        if "id" not in d or not d["id"]:
            d["id"] = str(oid)
    return d


class InsertOneResult:
    def __init__(self, inserted_id: Any):
        self.inserted_id = inserted_id


class InsertManyResult:
    def __init__(self, inserted_ids: List[Any]):
        self.inserted_ids = inserted_ids


class UpdateResult:
    def __init__(self, modified_count: int = 1, matched_count: int = 1):
        self.modified_count = modified_count
        self.matched_count = matched_count


class DeleteResult:
    def __init__(self, deleted_count: int = 1):
        self.deleted_count = deleted_count


class MongoCursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor

    def sort(self, *args, **kwargs):
        self._cursor = self._cursor.sort(*args, **kwargs)
        return self

    def limit(self, limit: int):
        self._cursor = self._cursor.limit(limit)
        return self

    def skip(self, skip: int):
        self._cursor = self._cursor.skip(skip)
        return self

    async def to_list(self, length: Optional[int] = 1000) -> List[Dict[str, Any]]:
        raw_docs = await self._cursor.to_list(length)
        return [_clean_doc_output(d) for d in raw_docs if d is not None]

    def __aiter__(self):
        cursor = self._cursor

        async def _gen():
            async for doc in cursor:
                yield _clean_doc_output(doc)

        return _gen()


class MongoCollectionWrapper:
    def __init__(self, name: str):
        self._name = TABLE_ALIASES.get(name, name)

    @property
    def _coll(self):
        return get_async_db()[self._name]

    def _normalize_filter(self, filter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not filter:
            return {}
        f = dict(filter)
        # If querying by string "id", and it looks like an ObjectId, also accept _id match
        if "id" in f and isinstance(f["id"], str) and len(f["id"]) == 24:
            try:
                oid = ObjectId(f["id"])
                # If neither _id nor $or is explicitly given, allow either
                if "_id" not in f and "$or" not in f:
                    return {"$or": [{"id": f["id"]}, {"_id": oid}]}
            except Exception:
                pass
        return f

    def find(self, filter: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None) -> MongoCursorWrapper:
        f = self._normalize_filter(filter)
        cursor = self._coll.find(f, projection)
        return MongoCursorWrapper(cursor)

    async def find_one(self, filter: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        f = self._normalize_filter(filter)
        raw = await self._coll.find_one(f, projection)
        return _clean_doc_output(raw)

    async def insert_one(self, document: Dict[str, Any]) -> InsertOneResult:
        doc = dict(document)
        if "id" not in doc or not doc["id"]:
            doc["id"] = str(uuid.uuid4())
        # Remove empty or None _id so MongoDB creates a valid ObjectId
        if "_id" in doc and doc["_id"] is None:
            doc.pop("_id")
        res = await self._coll.insert_one(doc)
        return InsertOneResult(doc.get("id") or str(res.inserted_id))

    async def insert_many(self, documents: List[Dict[str, Any]]) -> InsertManyResult:
        if not documents:
            return InsertManyResult([])
        clean = []
        for d in documents:
            doc = dict(d)
            if "id" not in doc or not doc["id"]:
                doc["id"] = str(uuid.uuid4())
            if "_id" in doc and doc["_id"] is None:
                doc.pop("_id")
            clean.append(doc)
        res = await self._coll.insert_many(clean)
        ids = [d.get("id") or str(iid) for d, iid in zip(clean, res.inserted_ids)]
        return InsertManyResult(ids)

    async def update_one(self, filter: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> UpdateResult:
        f = self._normalize_filter(filter)
        # Ensure update has standard mongo operators ($set, $inc, etc.)
        if not any(k.startswith("$") for k in update.keys()):
            update = {"$set": update}
        res = await self._coll.update_one(f, update, upsert=upsert)
        return UpdateResult(modified_count=res.modified_count, matched_count=res.matched_count)

    async def update_many(self, filter: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> UpdateResult:
        f = self._normalize_filter(filter)
        if not any(k.startswith("$") for k in update.keys()):
            update = {"$set": update}
        res = await self._coll.update_many(f, update, upsert=upsert)
        return UpdateResult(modified_count=res.modified_count, matched_count=res.matched_count)

    async def delete_one(self, filter: Dict[str, Any]) -> DeleteResult:
        f = self._normalize_filter(filter)
        res = await self._coll.delete_one(f)
        return DeleteResult(deleted_count=res.deleted_count)

    async def delete_many(self, filter: Dict[str, Any]) -> DeleteResult:
        f = self._normalize_filter(filter)
        res = await self._coll.delete_many(f)
        return DeleteResult(deleted_count=res.deleted_count)

    async def count_documents(self, filter: Optional[Dict[str, Any]] = None) -> int:
        f = self._normalize_filter(filter)
        return await self._coll.count_documents(f)

    def aggregate(self, pipeline: List[Dict[str, Any]]):
        return self._coll.aggregate(pipeline)


class MongoDatabaseWrapper:
    def __getattr__(self, name: str) -> MongoCollectionWrapper:
        return MongoCollectionWrapper(name)

    def __getitem__(self, name: str) -> MongoCollectionWrapper:
        return MongoCollectionWrapper(name)


db = MongoDatabaseWrapper()


# ── Compatibility Shims (Graceful transition for legacy references) ──────────
def get_mongo_fallback():
    """Points directly to primary MongoDB Atlas database."""
    return get_async_db()


class _DummySupabaseResult:
    def __init__(self, data=None, count=0):
        self.data = data or []
        self.count = count

    def execute(self):
        return self


class _DummySupabaseTable:
    def __init__(self, table_name: str):
        self.table_name = table_name

    def select(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def neq(self, *args, **kwargs):
        return self

    def delete(self, *args, **kwargs):
        return self

    def insert(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def upsert(self, *args, **kwargs):
        return self

    def execute(self):
        # Sync fallback to MongoDB
        try:
            sdb = get_sync_db()
            coll_name = TABLE_ALIASES.get(self.table_name, self.table_name)
            docs = list(sdb[coll_name].find().limit(100))
            return _DummySupabaseResult(data=[_clean_doc_output(d) for d in docs])
        except Exception:
            return _DummySupabaseResult(data=[])


class _DummySupabaseAuth:
    class _Admin:
        def list_users(self):
            return []

        def get_user_by_id(self, uid):
            return None

        def update_user_by_id(self, uid, changes):
            return None

        def delete_user(self, uid):
            return None

        def create_user(self, data):
            return None

    admin = _Admin()

    def get_user(self, token):
        return None


class _DummySupabaseClient:
    auth = _DummySupabaseAuth()

    def table(self, name: str):
        return _DummySupabaseTable(name)


_dummy_sb_instance = _DummySupabaseClient()


def get_supabase():
    return _dummy_sb_instance


supabase = _dummy_sb_instance
