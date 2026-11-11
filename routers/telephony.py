import logging
from typing import Optional
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
    phone_number: str = Field(default="+919003899180", description="Recipient phone number")
    message: Optional[str] = Field(
        default="Flood in area evacuate now. Seek high ground immediately.",
        description="Evacuation warning text",
    )
    make_call: bool = Field(default=True, description="Whether to place a voice call")
    send_message: bool = Field(default=True, description="Whether to send an SMS text")


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
    phone = normalize_phone_number(payload.phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number provided")

    results = {"phone": phone, "call": None, "sms": None}

    msg_text = payload.message or "Flood in area evacuate now. Seek high ground immediately."

    if payload.send_message:
        sms_res = await send_sms(phone, f"🚨 EMERGENCY: {msg_text} (NEXGI Warning)")
        results["sms"] = sms_res

    if payload.make_call:
        voice_res = await make_voice_call(phone, f"Emergency warning! {msg_text} Evacuate now.")
        results["call"] = voice_res

    return {
        "status": "completed",
        "message": f"Telephony alert executed for {phone}",
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
