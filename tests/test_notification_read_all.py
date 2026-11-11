import asyncio
import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from supabase import ClientOptions, create_client

from lib import db as storage
from lib.auth import current_user
from routers.admin import router


def test_read_all_updates_only_unread_notifications(monkeypatch):
    calls = []

    def database(request):
        calls.append(request)
        assert request.method == "PATCH"
        assert request.url.path == "/rest/v1/notifications"
        assert request.url.params["read"].lower() == "eq.false"
        assert json.loads(request.content) == {"read": True}
        return httpx.Response(200, json=[{"id": "a", "read": True}, {"id": "b", "read": True}])

    with httpx.Client(transport=httpx.MockTransport(database)) as http:
        supabase = create_client("https://example.supabase.co", "test-key", ClientOptions(httpx_client=http))
        monkeypatch.setattr(storage, "get_supabase", lambda: supabase)
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[current_user] = lambda: {"id": "test", "role": "admin"}
        response = TestClient(app).post("/api/notifications/read-all")
        assert response.status_code == 200
        assert len(calls) == 1


def test_bulk_update_reports_zero_when_no_rows_match(monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[]))) as http:
        supabase = create_client("https://example.supabase.co", "test-key", ClientOptions(httpx_client=http))
        monkeypatch.setattr(storage, "get_supabase", lambda: supabase)
        result = asyncio.run(storage.db.notifications.update_many({"read": False}, {"$set": {"read": True}}))
        assert result.modified_count == 0
