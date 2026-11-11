"""Pydantic v2 models. Each has a hand-written TS mirror in frontend/src/lib/types.ts."""
import uuid
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- users / auth ----------
class RegisterRequest(BaseModel):
    first_name: str = Field(min_length=1, max_length=60)
    last_name: str = Field(min_length=1, max_length=60)
    email: EmailStr
    phone: str = Field(min_length=6, max_length=20)
    organization: str = Field(min_length=2, max_length=140)
    designation: str
    state: str
    district: str
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class User(BaseModel):
    id: str = Field(default_factory=_uid)
    first_name: str
    last_name: str
    email: str
    phone: str = ""
    organization: str = ""
    designation: str = ""
    role: str = "viewer"
    state: str = ""
    district: str = ""
    status: str = "pending"  # pending | active | suspended
    verified: bool = False
    created_at: datetime = Field(default_factory=_now)


class ProfileUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    organization: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)


class UserAdminUpdate(BaseModel):
    role: Optional[str] = None
    status: Optional[str] = None


class MessageResponse(BaseModel):
    message: str


class RegisterResponse(BaseModel):
    message: str
    requires_verification: bool
    user: User


# ---------- network ----------
class Sensor(BaseModel):
    id: str = Field(default_factory=_uid)
    code: str
    name: str
    sensor_type: str  # rainfall | soil_moisture | temperature | water_level | smoke | air_quality | tilt
    zone_id: str
    zone_name: str = ""
    lat: float
    lng: float
    status: str = "online"  # online | offline | maintenance
    battery: int = 100
    signal_dbm: int = -80
    last_value: float = 0.0
    unit: str = ""
    gateway_id: str = ""
    updated_at: datetime = Field(default_factory=_now)


class SensorCreate(BaseModel):
    code: str = Field(min_length=2, max_length=32)
    name: str = Field(min_length=2, max_length=120)
    sensor_type: str
    zone_id: str
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    status: str = "online"
    battery: int = Field(default=100, ge=0, le=100)
    unit: str = ""


class SensorUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    battery: Optional[int] = None
    last_value: Optional[float] = None


class Gateway(BaseModel):
    id: str = Field(default_factory=_uid)
    code: str
    name: str
    zone_name: str = ""
    lat: float
    lng: float
    status: str = "online"
    connected_nodes: int = 0
    uplink_rate: float = 99.0


class Zone(BaseModel):
    id: str = Field(default_factory=_uid)
    name: str
    state: str
    district: str
    lat: float
    lng: float
    radius_km: float = 8.0
    hazard_type: str = "landslide"  # landslide | flood | forest_fire | air_quality
    risk_level: str = "low"  # low | medium | high | critical
    risk_score: int = 0
    rainfall_mm: float = 0.0
    temperature_c: float = 0.0
    humidity_pct: float = 0.0
    water_level_m: float = 0.0
    soil_moisture_pct: float = 0.0
    sensor_count: int = 0


class NetworkStats(BaseModel):
    active_sensors: int
    total_sensors: int
    online_gateways: int
    active_alerts: int
    monitoring_zones: int
    data_streams: int
    national_hazard_level: str


class TelemetryPoint(BaseModel):
    label: str
    rainfall_mm: float
    water_level_m: float
    temperature_c: float
    soil_moisture_pct: float
    packet_rate: float


class TelemetrySeries(BaseModel):
    zone_id: Optional[str] = None
    points: List[TelemetryPoint]


# ---------- alerts ----------
class Alert(BaseModel):
    id: str = Field(default_factory=_uid)
    code: str
    zone_id: Optional[str] = ""
    location: str
    hazard_type: str
    risk_level: str  # critical | high | medium | low
    title: str
    detail: Optional[str] = ""
    status: str = "open"  # open | acknowledged | assigned | resolved
    assigned_to: Optional[str] = ""
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class AlertCreate(BaseModel):
    location: str = Field(min_length=2)
    hazard_type: str
    risk_level: Literal["critical", "high", "medium", "low"]
    title: str = Field(min_length=3)
    detail: str = ""
    zone_id: str = ""


class AlertAction(BaseModel):
    action: Literal["acknowledge", "assign", "resolve", "reopen"]
    assigned_to: str = ""


# ---------- AI risk / digital twin ----------
class RiskFactor(BaseModel):
    name: str
    weight: int
    value: str


class RiskAssessment(BaseModel):
    zone_id: str
    zone_name: str
    risk_score: int
    risk_level: str
    confidence: float
    factors: List[RiskFactor]
    recommendation: str
    data_sources: List[str]
    models_used: List[str]
    analyzed_at: datetime = Field(default_factory=_now)
    disclaimer: str = (
        "AI recommendations support human decision-making and do not replace "
        "authorized government officials."
    )


class SimulationRequest(BaseModel):
    zone_id: str
    scenario: Literal["flood", "landslide", "forest_fire", "evacuation"]
    rainfall_intensity: float = Field(default=60, ge=0, le=200)
    soil_saturation: float = Field(default=50, ge=0, le=100)
    slope_angle: float = Field(default=30, ge=0, le=70)
    wind_speed: float = Field(default=15, ge=0, le=150)


class SimulationStep(BaseModel):
    hour: int
    impact_pct: float
    affected_area_km2: float


class SimulationResult(BaseModel):
    id: str = Field(default_factory=_uid)
    zone_id: str
    zone_name: str
    scenario: str
    severity: str
    peak_impact_pct: float
    affected_area_km2: float
    population_at_risk: int
    evacuation_time_min: int
    steps: List[SimulationStep]
    summary: str
    run_at: datetime = Field(default_factory=_now)


# ---------- notifications & reports ----------
class Notification(BaseModel):
    id: str = Field(default_factory=_uid)
    kind: str  # critical_alert | system_update | sensor_offline | ai_prediction | report_ready
    title: str
    body: str = ""
    read: bool = False
    created_at: datetime = Field(default_factory=_now)


class SimulationReportPayload(BaseModel):
    runId: uuid.UUID
    startedAt: datetime
    endedAt: datetime
    scenario: dict[str, Any]
    settingsHistory: list[dict[str, Any]] = Field(max_length=10000)
    buildings: list[dict[str, Any]] = Field(max_length=100000)
    summary: dict[str, int]
    method: str = Field(max_length=4000)


class SimulationReportCreate(BaseModel):
    title: str = Field(min_length=3, max_length=300)
    period: str = Field(max_length=200)
    zone_name: str = Field(default="", max_length=300)
    simulation_report: SimulationReportPayload


class Report(BaseModel):
    id: str = Field(default_factory=_uid)
    title: str
    report_type: str
    period: str
    zone_name: str = ""
    status: str = "ready"
    size_kb: int = 0
    simulation_report: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=_now)


class ReportCreate(BaseModel):
    title: str = Field(min_length=3)
    report_type: str
    period: str
    zone_name: str = ""


# ---------- mobile app citizens (mob_users) ----------
class MobUser(BaseModel):
    id: str
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    language: Optional[str] = "en"
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_accuracy_m: Optional[int] = None
    location_updated_at: Optional[str] = None
    preferences: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MobUserAlertRequest(BaseModel):
    hazard_type: str = "Flash Flood"
    risk_level: str = "critical"
    title: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    channels: Optional[List[str]] = ["call", "message", "in_app"]
    dispatch_mode: Optional[str] = "manual"  # manual | automatic
    monitored_area: Optional[str] = "Pollachi Catchment Basin"
    evacuation_point: Optional["MobUserEvacuationRequest"] = None


class MobUserEvacuationRequest(BaseModel):
    shelter_name: str = Field(min_length=1)
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    instructions: Optional[str] = "Proceed immediately to the designated safe elevation zone."


class MobUserBroadcastAlertRequest(BaseModel):
    target: str = "monitored_zone"  # monitored_zone | all | selected
    user_ids: Optional[List[str]] = None
    hazard_type: str = "Flash Flood"  # Flash Flood | Landslide
    risk_level: str = "critical"
    title: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    channels: List[str] = ["call", "message", "in_app"]
    dispatch_mode: str = "manual"  # manual | automatic
    monitored_area: Optional[str] = "Pollachi Catchment Basin"
    evacuation_point: Optional[MobUserEvacuationRequest] = None

