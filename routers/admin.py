from datetime import datetime, timezone
import time
import json
import math
from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException

from lib.auth import ROLES, current_user, require_roles
from lib.db import db
from models.schemas import (
    MessageResponse,
    MobUser,
    MobUserAlertRequest,
    MobUserBroadcastAlertRequest,
    MobUserEvacuationRequest,
    Notification,
    Report,
    ReportCreate,
    SimulationReportCreate,
    User,
    UserAdminUpdate,
)

router = APIRouter(tags=["admin"])


@router.get("/users", response_model=List[User])
async def list_users(user: dict = Depends(require_roles("admin"))):
    docs = await db.users.find({}, {"password_hash": 0}).to_list(500)
    users_list = []
    for d in docs:
        item = dict(d)
        if "first_name" not in item:
            name = item.get("name", "Officer")
            parts = name.split()
            item["first_name"] = parts[0] if parts else "Officer"
            item["last_name"] = " ".join(parts[1:]) if len(parts) > 1 else ""
        if "last_name" not in item:
            item["last_name"] = ""
        if "role" not in item:
            item["role"] = "admin"
        if "status" not in item:
            item["status"] = "active"
        if "verified" not in item:
            item["verified"] = True
        try:
            users_list.append(User(**item))
        except Exception:
            pass
    return users_list


@router.patch("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str, payload: UserAdminUpdate, user: dict = Depends(require_roles("admin"))
):
    changes = {k: v for k, v in payload.model_dump().items() if v is not None}
    if "role" in changes and changes["role"] not in ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {list(ROLES)}")
    if "status" in changes:
        if changes["status"] not in ("pending", "active", "suspended"):
            raise HTTPException(status_code=422, detail="status must be pending, active or suspended")
        changes["verified"] = changes["status"] == "active"

    # Update local MongoDB
    await db.users.update_one({"id": user_id}, {"$set": changes})
    doc = await db.users.find_one({"id": user_id}, {"password_hash": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")

    item = dict(doc)
    if "first_name" not in item:
        name = item.get("name", "Officer")
        parts = name.split()
        item["first_name"] = parts[0] if parts else "Officer"
        item["last_name"] = " ".join(parts[1:]) if len(parts) > 1 else ""
    if "last_name" not in item:
        item["last_name"] = ""
    return User(**item)


@router.delete("/users/{user_id}", response_model=MessageResponse)
async def delete_user(user_id: str, user: dict = Depends(require_roles("admin"))):
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="You cannot remove your own account.")
    await db.users.delete_one({"id": user_id})
    return MessageResponse(message="User account removed")


@router.get("/notifications", response_model=List[Notification])
async def list_notifications(user: dict = Depends(current_user)):
    docs = await db.notifications.find({}, {"_id": 0}).sort("created_at", -1).to_list(10)
    return [Notification(**d) for d in docs]


@router.post("/notifications/read-all", response_model=MessageResponse)
async def mark_all_read(user: dict = Depends(current_user)):
    await db.notifications.update_many({"read": False}, {"$set": {"read": True}})
    return MessageResponse(message="All notifications marked as read")


@router.get("/reports", response_model=List[Report])
async def list_reports(user: dict = Depends(current_user)):
    docs = await db.reports.find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
    return [Report(**d) for d in docs]


@router.post("/reports", response_model=Report, status_code=201)
async def create_report(
    payload: ReportCreate, user: dict = Depends(require_roles("admin", "gov_officer"))
):
    report = Report(**payload.model_dump(), status="ready", size_kb=180 + len(payload.title) * 7)
    await db.reports.insert_one(report.model_dump(exclude_none=True))
    await db.notifications.insert_one(
        Notification(kind="report_ready", title="Report ready", body=report.title).model_dump()
    )
    return report


@router.post("/reports/simulation", response_model=Report, status_code=201)
async def save_simulation_report(
    payload: SimulationReportCreate, user: dict = Depends(require_roles("admin", "gov_officer"))
):
    # Deterministic user-scoped ID makes retries safe after a lost response.
    report_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"simulation:{user['id']}:{payload.simulation_report.runId}"))
    existing = await db.reports.find_one({"id": report_id})
    if existing:
        return Report(**existing)
    data = payload.simulation_report.model_dump(mode="json")
    try:
        encoded = json.dumps(data, allow_nan=False).encode("utf-8")
    except ValueError:
        raise HTTPException(status_code=422, detail="Simulation values must be finite")
    if len(encoded) > 15_000_000:
        raise HTTPException(status_code=413, detail="Simulation report exceeds 15 MB")
    report = Report(id=report_id, title=payload.title, period=payload.period, zone_name=payload.zone_name,
                    report_type="Flood Simulation", status="ready", size_kb=math.ceil(len(encoded) / 1024),
                    simulation_report=data)
    try:
        await db.reports.insert_one(report.model_dump(mode="json"))
    except Exception:
        # Concurrent retries can race against the primary-key constraint.
        existing = await db.reports.find_one({"id": report_id})
        if existing:
            return Report(**existing)
        raise HTTPException(status_code=503, detail="Could not save simulation report; retry after checking database migration")
    return report


# ---------- Mobile App Citizens (mob_users) ----------
@router.get("/mob-users", response_model=List[MobUser])
async def list_mob_users(user: dict = Depends(current_user)):
    """Fetch registered mobile app citizen users from MongoDB."""
    try:
        docs = await db.mob_users.find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
        return [MobUser(**u) for u in docs]
    except Exception as exc:
        print(f"Error fetching mob_users from MongoDB: {exc}")
        return []


@router.post("/mob-users/{user_id}/alert")
async def send_mob_user_alert(
    user_id: str,
    payload: MobUserAlertRequest,
    admin: dict = Depends(require_roles("admin", "gov_officer")),
):
    """Dispatch emergency alert to a citizen (updates mob_users.preferences, alerts, notifications in MongoDB)."""
    try:
        target_user = await db.mob_users.find_one({"id": user_id})
        if not target_user:
            raise HTTPException(status_code=404, detail="Mobile citizen user not found")

        now_iso = datetime.now(timezone.utc).isoformat()
        prefs = target_user.get("preferences") or {}
        alert_data = {
            "hazard_type": payload.hazard_type,
            "risk_level": payload.risk_level,
            "title": payload.title,
            "detail": payload.detail,
            "channels": payload.channels or ["call", "message", "in_app"],
            "dispatch_mode": payload.dispatch_mode or "manual",
            "sent_at": now_iso,
            "active": True,
        }
        prefs["active_alert"] = alert_data

        if payload.evacuation_point:
            prefs["evacuation_point"] = {
                "shelter_name": payload.evacuation_point.shelter_name,
                "latitude": payload.evacuation_point.latitude,
                "longitude": payload.evacuation_point.longitude,
                "elevation_m": payload.evacuation_point.elevation_m,
                "instructions": payload.evacuation_point.instructions or "Proceed to shelter.",
                "assigned_at": now_iso,
            }

        # 1. Update mob_users in MongoDB
        await db.mob_users.update_one({"id": user_id}, {"$set": {"preferences": prefs, "updated_at": now_iso}})

        # 2. Insert into alerts
        try:
            await db.alerts.insert_one({
                "id": str(uuid.uuid4()),
                "code": f"ALT-{int(time.time())}",
                "zone_id": "ZONE-POLLACHI",
                "location": target_user.get("location_name") or "Pollachi Sector",
                "hazard_type": payload.hazard_type,
                "risk_level": payload.risk_level,
                "title": payload.title,
                "detail": payload.detail,
                "status": "active",
                "assigned_to": target_user.get("full_name") or "Citizen",
            })
        except Exception as alert_err:
            print(f"Notice: alerts insert failed: {alert_err}")

        # 3. Insert into notifications
        await db.notifications.insert_one({
            "kind": "critical_alert",
            "title": f"🚨 Mobile Alert to {target_user.get('full_name') or 'Citizen'}: {payload.title}",
            "body": payload.detail,
            "read": False,
            "created_at": datetime.now(timezone.utc),
        })

        # 4. Insert into mob_alerts
        mob_alert_doc = {
            "id": str(uuid.uuid4()),
            "alert_code": f"MAL-{int(time.time())}-{str(uuid.uuid4())[:4].upper()}",
            "user_id": user_id,
            "recipient_name": target_user.get("full_name") or "Citizen",
            "recipient_phone": target_user.get("phone_number") or "",
            "recipient_location": target_user.get("location_name") or "Monitored Zone",
            "hazard_type": payload.hazard_type,
            "risk_level": payload.risk_level,
            "title": payload.title,
            "detail": payload.detail,
            "channels": payload.channels or ["call", "message", "in_app"],
            "dispatch_mode": payload.dispatch_mode or "manual",
            "monitored_area": payload.monitored_area or "Pollachi Catchment Basin",
            "evacuation_point": payload.evacuation_point.model_dump() if payload.evacuation_point else None,
            "status": "delivered",
            "sent_at": now_iso,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await db.mob_alerts.insert_one(dict(mob_alert_doc))

        # 5. Execute real Twilio Telephony (Voice Call & SMS) if selected in channels
        telephony_res = None
        phone_num = target_user.get("phone_number") or ""
        if phone_num and any(ch.lower() in ["call", "voice", "message", "sms"] for ch in (payload.channels or [])):
            try:
                from services.twilio_service import dispatch_telephony_alert
                telephony_res = await dispatch_telephony_alert(
                    phone_number=phone_num,
                    channels=payload.channels or ["call", "message"],
                    title=payload.title,
                    detail=payload.detail,
                    hazard_type=payload.hazard_type,
                )
            except Exception as tel_err:
                print(f"[Telephony Alert] Error dispatching to {phone_num}: {tel_err}")

        return {
            "status": "ok",
            "message": f"Alert successfully dispatched to {target_user.get('full_name') or 'Citizen'}",
            "alert": alert_data,
            "mob_alert": mob_alert_doc,
            "telephony": telephony_res,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to dispatch alert: {str(e)}")


@router.post("/mob-users/broadcast-alert")
async def broadcast_mob_user_alert(
    payload: MobUserBroadcastAlertRequest,
    admin: dict = Depends(require_roles("admin", "gov_officer")),
):
    """Broadcast emergency alert to citizens via MongoDB."""
    try:
        all_users = await db.mob_users.find({}, {"_id": 0}).to_list(1000)

        # Filter target recipients
        target_users = []
        if payload.target == "monitored_zone":
            selected_area_lower = (payload.monitored_area or "").lower()
            for u in all_users:
                loc = (u.get("location_name") or "").lower()
                lat = u.get("latitude")
                lng = u.get("longitude")
                in_bbox = lat is not None and lng is not None and (10.50 <= lat <= 10.80) and (76.85 <= lng <= 77.15)
                in_zone_text = any(k in loc for k in ["pollachi", "catchment", "basin", "sector", "zone", "aliyar", "sholayar", "valparai", "coimbatore"])
                in_specific = any(term in loc for term in selected_area_lower.split()) if selected_area_lower else True
                if in_bbox or in_zone_text or in_specific or (lat is None and lng is None):
                    target_users.append(u)
        elif payload.target == "selected" and payload.user_ids:
            selected_ids = set(payload.user_ids)
            target_users = [u for u in all_users if u.get("id") in selected_ids]
        else:
            target_users = all_users

        now_iso = datetime.now(timezone.utc).isoformat()
        dispatched_count = 0
        mob_alert_records = []

        for u in target_users:
            u_id = u.get("id")
            prefs = u.get("preferences") or {}
            alert_data = {
                "hazard_type": payload.hazard_type,
                "risk_level": payload.risk_level,
                "title": payload.title,
                "detail": payload.detail,
                "channels": payload.channels,
                "dispatch_mode": payload.dispatch_mode,
                "sent_at": now_iso,
                "active": True,
            }
            prefs["active_alert"] = alert_data

            if payload.evacuation_point:
                prefs["evacuation_point"] = {
                    "shelter_name": payload.evacuation_point.shelter_name,
                    "latitude": payload.evacuation_point.latitude,
                    "longitude": payload.evacuation_point.longitude,
                    "elevation_m": payload.evacuation_point.elevation_m,
                    "instructions": payload.evacuation_point.instructions or "Proceed to designated high-ground shelter.",
                    "assigned_at": now_iso,
                }

            await db.mob_users.update_one({"id": u_id}, {"$set": {"preferences": prefs, "updated_at": now_iso}})
            dispatched_count += 1

            alert_id = str(uuid.uuid4())
            mob_alert_records.append({
                "id": alert_id,
                "alert_code": f"MAL-{int(time.time())}-{alert_id[:4].upper()}",
                "user_id": u_id,
                "recipient_name": u.get("full_name") or "Citizen",
                "recipient_phone": u.get("phone_number") or "",
                "recipient_location": u.get("location_name") or "Monitored Zone",
                "hazard_type": payload.hazard_type,
                "risk_level": payload.risk_level,
                "title": payload.title,
                "detail": payload.detail,
                "channels": payload.channels,
                "dispatch_mode": payload.dispatch_mode,
                "monitored_area": payload.monitored_area or "Pollachi Catchment Basin",
                "evacuation_point": payload.evacuation_point.model_dump() if payload.evacuation_point else None,
                "status": "delivered",
                "sent_at": now_iso,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

        if mob_alert_records:
            await db.mob_alerts.insert_many([dict(r) for r in mob_alert_records])

        await db.notifications.insert_one({
            "kind": "critical_alert",
            "title": f"🚨 Broadcast Alert ({payload.target}): {payload.title}",
            "body": f"Delivered to {dispatched_count} citizens via {', '.join(payload.channels)}. {payload.detail}",
            "read": False,
            "created_at": datetime.now(timezone.utc),
        })

        telephony_recipients = 0
        if any(ch.lower() in ["call", "voice", "message", "sms"] for ch in (payload.channels or [])):
            try:
                from services.twilio_service import dispatch_telephony_alert
                for u in target_users:
                    p_num = u.get("phone_number") or ""
                    if p_num:
                        telephony_recipients += 1
                        asyncio.create_task(
                            dispatch_telephony_alert(
                                phone_number=p_num,
                                channels=payload.channels,
                                title=payload.title,
                                detail=payload.detail,
                                hazard_type=payload.hazard_type,
                            )
                        )
            except Exception as tel_broadcast_err:
                print(f"[Telephony Broadcast] Error scheduling alerts: {tel_broadcast_err}")

        target_label = f"Citizens in {payload.monitored_area}" if payload.target == "monitored_zone" else "All Citizens" if payload.target == "all" else "Selected Citizens"
        return {
            "status": "ok",
            "message": f"Successfully dispatched emergency alert to {dispatched_count} {target_label} via {', '.join(payload.channels)}",
            "dispatched_count": dispatched_count,
            "telephony_broadcast_count": telephony_recipients,
            "target": payload.target,
            "monitored_area": payload.monitored_area,
            "saved_to_db": "mob_alerts",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast alert: {str(e)}")


@router.get("/mob-alerts")
async def list_mob_alerts(user: dict = Depends(current_user)):
    """Fetch recent dispatched citizen alerts from MongoDB mob_alerts collection."""
    try:
        docs = await db.mob_alerts.find({}, {"_id": 0}).sort("sent_at", -1).limit(100).to_list(100)
        return docs
    except Exception as e:
        print(f"Notice: list_mob_alerts error: {e}")
        return []


@router.delete("/mob-alerts/{alert_identifier}")
async def delete_mob_alert(
    alert_identifier: str,
    admin: dict = Depends(require_roles("admin", "gov_officer")),
):
    """Delete a dispatched mobile alert record by id or alert_code from MongoDB."""
    try:
        res = await db.mob_alerts.delete_many({
            "$or": [{"id": alert_identifier}, {"alert_code": alert_identifier}]
        })
        return {
            "status": "ok",
            "message": f"Alert {alert_identifier} deleted successfully",
            "deleted_count": res.deleted_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete alert: {str(e)}")


@router.post("/mob-users/{user_id}/evacuation-point")
async def send_mob_user_evacuation(
    user_id: str,
    payload: MobUserEvacuationRequest,
    admin: dict = Depends(require_roles("admin", "gov_officer")),
):
    """Assign emergency evacuation point to a citizen."""
    try:
        target_user = await db.mob_users.find_one({"id": user_id})
        if not target_user:
            raise HTTPException(status_code=404, detail="Mobile citizen user not found")

        now_iso = datetime.now(timezone.utc).isoformat()
        prefs = target_user.get("preferences") or {}
        evac_data = {
            "shelter_name": payload.shelter_name,
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "elevation_m": payload.elevation_m,
            "instructions": payload.instructions or "Proceed immediately to the designated safe evacuation shelter.",
            "assigned_at": now_iso,
        }
        prefs["evacuation_point"] = evac_data

        await db.mob_users.update_one({"id": user_id}, {"$set": {"preferences": prefs, "updated_at": now_iso}})

        try:
            await db.alerts.insert_one({
                "id": str(uuid.uuid4()),
                "code": f"EVAC-{int(time.time())}",
                "zone_id": "ZONE-POLLACHI",
                "location": target_user.get("location_name") or f"Coords ({payload.latitude}, {payload.longitude})",
                "hazard_type": "Evacuation Order",
                "risk_level": "critical",
                "title": f"Evacuate to {payload.shelter_name}",
                "detail": payload.instructions or "Proceed immediately to shelter.",
                "status": "active",
                "assigned_to": target_user.get("full_name") or "Citizen",
            })
        except Exception as alert_err:
            print(f"Notice: alerts insert failed: {alert_err}")

        await db.notifications.insert_one({
            "kind": "critical_alert",
            "title": f"🏃 Evacuation Assigned to {target_user.get('full_name') or 'Citizen'}: {payload.shelter_name}",
            "body": f"Shelter: {payload.shelter_name} ({payload.latitude}, {payload.longitude}). {payload.instructions}",
            "read": False,
            "created_at": datetime.now(timezone.utc),
        })

        return {
            "status": "ok",
            "message": f"Evacuation point assigned to {target_user.get('full_name') or 'Citizen'}",
            "evacuation_point": evac_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assign evacuation point: {str(e)}")


@router.delete("/mob-users/{user_id}/alert")
async def clear_mob_user_alert(
    user_id: str,
    admin: dict = Depends(require_roles("admin", "gov_officer")),
):
    """Clear active alert for a mobile citizen."""
    try:
        target_user = await db.mob_users.find_one({"id": user_id})
        if not target_user:
            raise HTTPException(status_code=404, detail="Mobile citizen user not found")

        now_iso = datetime.now(timezone.utc).isoformat()
        prefs = target_user.get("preferences") or {}
        if "active_alert" in prefs:
            del prefs["active_alert"]

        await db.mob_users.update_one({"id": user_id}, {"$set": {"preferences": prefs, "updated_at": now_iso}})
        return {"status": "ok", "message": f"Alert cleared for {target_user.get('full_name') or 'Citizen'}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear alert: {str(e)}")

