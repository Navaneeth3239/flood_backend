"""Deployment status backed by strict weight loading and a real forward pass."""
from ml.data.feature_schema import feature_schema
from ml.inference.model_loader import ModelNotReady, get_model


def model_info() -> dict:
    # Optional ML dependencies are loaded lazily by get_model.
    result = {
        "status": "checkpoint_missing", "ready": False, "calibrated": False,
        "reason": None, "model_version": None, "trained_at": None,
        "temporal_context_hours": 24, "forecast_horizon_hours": 1,
        "architecture": {"temporal": "chronos", "spatial": "GATv2 + Graph Transformer",
                         "heads": ["flood", "landslide", "combined_risk", "lead_time"]},
        "features": feature_schema(),
        "risk_thresholds": {"moderate": 0.30, "high": 0.55, "critical": 0.75},
        "heatmap": {"format": "GeoJSON FeatureCollection",
                    "endpoint": "/api/digital-twin/multi-hazard/heatmap/latest",
                    "colors": {"LOW": "#16a34a", "MODERATE": "#eab308", "HIGH": "#f97316", "CRITICAL": "#dc2626"}},
    }
    try:
        bundle = get_model(require_ready=False)
        config = bundle.config
        calibration = bundle.checkpoint.get("calibration", {})
        result.update(
            status="model_ready" if bundle.ready else "model_loaded", ready=bundle.ready,
            reason=bundle.reason, model_version=bundle.version,
            trained_at=bundle.checkpoint["trained_at"],
            calibrated=all(name in calibration.get("temperatures", {}) for name in ("flood", "landslide", "combined_risk")),
            temporal_context_hours=config["data"]["sequence_length"] * bundle.preprocessor.interval_minutes / 60,
            forecast_horizon_hours=config["data"]["forecast_horizon_minutes"] / 60,
            risk_thresholds=config["risk"],
            device=bundle.device,
            trained_on=bundle.checkpoint["training"].get("data_provenance"),
            forward_pass_verified=True,
        )
        result["architecture"]["temporal"] = config["model"]["temporal_backbone"]
        result["architecture"]["pretrained_model"] = config["model"].get("pretrained_model_name") if config["model"]["temporal_backbone"] == "chronos" else None
    except ModelNotReady as exc:
        result.update(status=exc.status, reason=exc.reason, forward_pass_verified=False)
    return result
