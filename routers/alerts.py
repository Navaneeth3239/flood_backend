from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from lib.auth import current_user, require_roles
from lib.db import db
from models.schemas import Alert, AlertAction, AlertCreate, MessageResponse, Notification

router = APIRouter(tags=["alerts"])

_PREFIX = {"critical": "CRT", "high": "HGH", "medium": "MED", "low": "LOW"}


@router.get("/alerts", response_model=List[Alert])
async def list_alerts(
    risk_level: Optional[str] = Query(default=None),
    hazard_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    location: Optional[str] = Query(default=None),
    user: dict = Depends(current_user),
):
    query: dict = {}
    if risk_level:
        query["risk_level"] = risk_level
    if hazard_type:
        query["hazard_type"] = hazard_type
    if status:
        query["status"] = status
    if location:
        query["location"] = {"$regex": location, "$options": "i"}
    docs = await db.alerts.find(query, {"_id": 0}).sort("created_at", -1).to_list(500)
    return [Alert(**d) for d in docs]


@router.get("/alerts/sos")
async def get_sos_requests(
    status: Optional[str] = Query(default=None),
    emergency_type: Optional[str] = Query(default=None),
):
    """Fetch live mobile SOS emergency requests from MongoDB mob_sos_requests collection."""
    try:
        query = {}
        if status:
            query["status"] = {"$regex": status.strip(), "$options": "i"}
        if emergency_type:
            query["emergency_type"] = {"$regex": emergency_type.strip(), "$options": "i"}
        return await db.mob_sos_requests.find(query, {"_id": 0}).sort("created_at", -1).limit(100).to_list(100)
    except Exception as e:
        print(f"[SOS] Error fetching mob_sos_requests: {e}")
        return []


@router.post("/alerts/sos")
async def create_sos_request(payload: dict):
    """Insert a new mobile SOS request into MongoDB mob_sos_requests collection."""
    try:
        import uuid
        from datetime import datetime, timezone
        data = {
            "id": payload.get("id") or str(uuid.uuid4()),
            "user_id": payload.get("user_id"),
            "full_name": payload.get("full_name") or "Citizen",
            "phone_number": payload.get("phone_number") or "",
            "language": payload.get("language") or "en",
            "location_name": payload.get("location_name") or "Current Location",
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "location_accuracy_m": payload.get("location_accuracy_m"),
            "emergency_type": payload.get("emergency_type") or "FLOOD_RESCUE",
            "description": payload.get("description") or "",
            "status": (payload.get("status") or "RECEIVED").upper(),
            "created_at": payload.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "location_source": payload.get("location_source") or "mobile_app",
        }
        await db.mob_sos_requests.insert_one(data)
        return {"status": "success", "data": [data]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create SOS request: {str(e)}")


class SosStatusUpdate(AlertAction):
    status: Optional[str] = None


@router.patch("/alerts/sos/{sos_id}/status")
async def update_sos_status(
    sos_id: str,
    payload: dict,
):
    """Update status of a mobile SOS request in MongoDB."""
    try:
        new_status = (payload.get("status") or "ACKNOWLEDGED").upper()
        res = await db.mob_sos_requests.update_one({"id": sos_id}, {"$set": {"status": new_status}})
        return {"status": "success", "modified_count": res.modified_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update SOS status: {str(e)}")


@router.get("/alerts/{alert_id}", response_model=Alert)
async def get_alert(alert_id: str, user: dict = Depends(current_user)):
    doc = await db.alerts.find_one({"id": alert_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Alert not found")
    return Alert(**doc)


@router.post("/alerts", response_model=Alert, status_code=201)
async def create_alert(
    payload: AlertCreate, user: dict = Depends(require_roles("admin", "gov_officer", "field_officer"))
):
    count = await db.alerts.count_documents({}) + 1
    alert = Alert(
        **payload.model_dump(),
        code=f"EIN-{_PREFIX.get(payload.risk_level, 'GEN')}-{count:04d}",
    )
    await db.alerts.insert_one(alert.model_dump())
    await db.notifications.insert_one(
        Notification(
            kind="critical_alert" if payload.risk_level in ("critical", "high") else "system_update",
            title=f"{payload.risk_level.upper()} — {payload.title}",
            body=f"{payload.location} · {payload.hazard_type.replace('_', ' ')}",
        ).model_dump()
    )
    return alert


@router.post("/alerts/{alert_id}/action", response_model=Alert)
async def act_on_alert(
    alert_id: str,
    payload: AlertAction,
    user: dict = Depends(require_roles("admin", "gov_officer", "field_officer")),
):
    doc = await db.alerts.find_one({"id": alert_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Alert not found")
    changes: dict = {"updated_at": datetime.now(timezone.utc)}
    if payload.action == "acknowledge":
        changes["status"] = "acknowledged"
    elif payload.action == "assign":
        if not payload.assigned_to:
            raise HTTPException(status_code=422, detail="assigned_to is required to assign an alert")
        changes["status"] = "assigned"
        changes["assigned_to"] = payload.assigned_to
    elif payload.action == "reopen":
        changes["status"] = "open"
    else:
        changes["status"] = "resolved"
    await db.alerts.update_one({"id": alert_id}, {"$set": changes})
    updated = await db.alerts.find_one({"id": alert_id}, {"_id": 0})
    return Alert(**updated)


@router.delete("/alerts/{alert_id}", response_model=MessageResponse)
async def delete_alert(alert_id: str, user: dict = Depends(require_roles("admin"))):
    result = await db.alerts.delete_one({"id": alert_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return MessageResponse(message="Alert record removed")

