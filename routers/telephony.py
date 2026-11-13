import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from lib.auth import current_user
from services.twilio_service import (
    get_telephony_status,
    make_voice_call,
    send_sms,
    normalize_phone_number,
    dispatch_telephony_alert,
)

logger = logging.getLogger("telephony_router")
router = APIRouter(prefix="/telephony", tags=["telephony"])


class TelephonyTestRequest(BaseModel):
    phone_number: str = Field(default="+919865013239", description="Recipient phone number")
    message: Optional[str] = Field(
        default="Emergency Flood Warning! Immediate Evacuation Alert! Flood in area, evacuate now! Move to higher ground immediately.",
        description="Evacuation warning text",
    )
    make_call: bool = Field(default=True, description="Whether to place a voice call")
    send_message: bool = Field(default=True, description="Whether to send an SMS text")
    url: Optional[str] = Field(default=None, description="Optional custom Twilio voice webhook URL")


class EmergencyWarningRequest(BaseModel):
    phone_number: Optional[str] = Field(default="+919865013239", description="Recipient phone number")
    area_name: Optional[str] = Field(default="Munnar – Devikulam, Kerala", description="Target monitoring area name")
    warning_level: Optional[str] = Field(default="CRITICAL", description="Alert severity level (WARNING, SEVERE, CRITICAL)")
    channels: Optional[List[str]] = Field(default=["call", "sms"], description="Channels: call, sms, or both")
    message: Optional[str] = Field(
        default=None,
        description="Optional custom evacuation warning message",
    )
    url: Optional[str] = Field(default=None, description="Optional Twilio voice webhook URL")


@router.get("/status")
async def telephony_status(user: dict = Depends(current_user)):
    """Check Twilio integration status, account health, and verified numbers."""
    return await get_telephony_status()


@router.post("/test-alert")
async def test_telephony_alert(
    payload: TelephonyTestRequest,
    user: dict = Depends(current_user),
):
    """Trigger a live emergency call and/or text message to test evacuation alerts."""
    phone = normalize_phone_number(payload.phone_number or "+919865013239")
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number provided")

    results = {"phone": phone, "call": None, "sms": None}

    msg_text = payload.message or "Flood in area evacuate now. Seek high ground immediately."

    if payload.send_message:
        sms_res = await send_sms(phone, f"🚨 EMERGENCY ALERT: {msg_text} (NEXGI Warning)")
        results["sms"] = sms_res

    if payload.make_call:
        voice_res = await make_voice_call(phone, speech_text=f"Emergency warning! {msg_text} Evacuate now.", url=payload.url)
        results["call"] = voice_res

    return {
        "status": "completed",
        "message": f"Telephony alert executed for {phone}",
        "results": results,
    }


@router.post("/emergency-warning")
async def trigger_emergency_warning(
    payload: EmergencyWarningRequest,
    user: dict = Depends(current_user),
):
    """Trigger immediate emergency call and/or SMS warning for disaster evacuation."""
    phone = normalize_phone_number(payload.phone_number or "+919865013239")
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid recipient phone number")

    area = payload.area_name or "Munnar – Devikulam, Kerala"
    level = (payload.warning_level or "CRITICAL").upper()
    channels = [c.lower() for c in (payload.channels or ["call", "sms"])]

    custom_text = payload.message.strip() if payload.message else None
    sms_body = (
        custom_text
        if custom_text
        else f"🚨 {level} FLOOD WARNING for {area}: Immediate evacuation required! Water levels rising rapidly. Move to designated high ground shelters now. (NEXGI Emergency System)"
    )
    voice_speech = (
        custom_text
        if custom_text
        else f"Emergency Flood Warning! Attention all residents in {area}. Flash flood detected. Evacuate immediately! Move to higher ground now. Do not cross low-lying bridges or waterways."
    )

    results = {"phone": phone, "area": area, "level": level, "channels": channels, "call": None, "sms": None}

    # Dispatch SMS
    if "sms" in channels or "message" in channels:
        sms_res = await send_sms(phone, sms_body)
        results["sms"] = sms_res

    # Dispatch Voice Call
    if "call" in channels or "voice" in channels:
        voice_res = await make_voice_call(phone, speech_text=voice_speech, url=payload.url)
        results["call"] = voice_res

    return {
        "status": "success",
        "message": f"Emergency warning dispatched to {phone}",
        "results": results,
    }


@router.post("/test-call")
async def test_call_only(
    payload: TelephonyTestRequest,
    user: dict = Depends(current_user),
):
    """Place a live voice call saying the evacuation notice."""
    phone = normalize_phone_number(payload.phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number provided")

    res = await make_voice_call(phone, payload.message)
    if not res.get("success"):
        raise HTTPException(status_code=502, detail=res.get("error", "Twilio call failed"))
    return {"status": "success", "call": res}


@router.post("/test-sms")
async def test_sms_only(
    payload: TelephonyTestRequest,
    user: dict = Depends(current_user),
):
    """Send an SMS text message with the evacuation notice."""
    phone = normalize_phone_number(payload.phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number provided")

    res = await send_sms(phone, f"🚨 EMERGENCY: {payload.message} (NEXGI Alert)")
    if not res.get("success"):
        raise HTTPException(status_code=502, detail=res.get("error", "Twilio SMS failed"))
    return {"status": "success", "sms": res}
