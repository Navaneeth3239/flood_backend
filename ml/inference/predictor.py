"""Batch graph inference and a bounded, process-local latest GeoJSON snapshot."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock

from .model_loader import ModelBundle, get_model
from .schemas import GraphPredictionRequest

COLORS = {"LOW": "#16a34a", "MODERATE": "#eab308", "HIGH": "#f97316", "CRITICAL": "#dc2626"}
_snapshots: OrderedDict[str | None, dict] = OrderedDict()
_snapshot_lock = RLock()
_inference_lock = RLock()


def request_frame(request: GraphPredictionRequest):
    import pandas as pd
    rows = []
    for node in request.nodes:
        for reading in node.readings:
            rows.append({
                **node.static_features.model_dump(),
                **reading.model_dump(),
                "node_id": node.node_id, "latitude": node.latitude, "longitude": node.longitude,
                "downstream_node_id": node.downstream_node_id, "river_id": node.river_id,
            })
    return pd.DataFrame(rows)


def risk_class(probability: float, thresholds: dict) -> str:
    moderate, high, critical = [thresholds[k] for k in ("moderate", "high", "critical")]
    if not 0 < moderate < high < critical < 1:
        raise ValueError("Invalid risk thresholds in checkpoint")
    return "CRITICAL" if probability >= critical else "HIGH" if probability >= high else "MODERATE" if probability >= moderate else "LOW"


def predict_graph(request: GraphPredictionRequest, publish: bool = True) -> dict:
    import torch
    from ..data.dataset import build_graph_sample
    from ..training.calibration import confidence_for_probability

    bundle = get_model()
    frame = request_frame(request)
    prepared = bundle.preprocessor.transform(frame)
    sample = build_graph_sample(prepared, bundle.preprocessor,
                                sequence_length=bundle.config["data"]["sequence_length"],
                                as_of=request.timestamp,
                                graph_config=bundle.config.get("graph", {}))
    sample = sample.to(bundle.device)
    with _inference_lock, torch.inference_mode():
        output = bundle.model(sample.x_temporal, sample.x_static, sample.x_satellite,
                              sample.edge_index, sample.edge_attr)
    calibration = bundle.checkpoint.get("calibration", {})
    temperatures = calibration.get("temperatures", {})
    probabilities = {}
    for name in ("flood", "landslide", "combined_risk"):
        logits = output[f"{name}_logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite logits")
        probabilities[name] = torch.sigmoid(logits / temperatures.get(name, 1.0)).cpu().tolist()
    lead_times = output["lead_time"].cpu().tolist()
    if not torch.isfinite(output["lead_time"]).all():
        raise RuntimeError("Model produced non-finite lead times")
    by_id = {node.node_id: node for node in request.nodes}
    prediction_timestamp = str(sample.timestamp)
    nodes = []
    lead_trained = bundle.checkpoint["training"].get("label_counts", {}).get("lead_time", 0) > 0
    for index, node_id in enumerate(sample.node_ids):
        source = by_id[node_id]
        combined = probabilities["combined_risk"][index]
        category = risk_class(combined, bundle.config["risk"])
        nodes.append({
            "node_id": node_id, "latitude": source.latitude, "longitude": source.longitude,
            "timestamp": prediction_timestamp, "model_version": bundle.version,
            "flood_probability": probabilities["flood"][index],
            "landslide_probability": probabilities["landslide"][index],
            "combined_risk": combined, "risk_class": category, "color": COLORS[category],
            "confidence": confidence_for_probability(calibration, combined),
            "lead_time_minutes": lead_times[index] if lead_trained else None,
            "area_id": request.area_id,
        })
    result = {"timestamp": prediction_timestamp, "generated_at": datetime.now(timezone.utc).isoformat(),
              "model_version": bundle.version, "area_id": request.area_id, "nodes": nodes,
              "calibrated": all(name in temperatures for name in ("flood", "landslide", "combined_risk")),
              "confidence_method": "Held-out empirical correctness of the combined-risk binary decision; not event probability.",
              "graph_scope": "All nodes supplied in this request; missing neighboring nodes are not synthesized."}
    if publish:
        with _snapshot_lock:
            previous = _snapshots.get(request.area_id)
            # A slow older request cannot overwrite a more recent prediction.
            if previous is None or previous["identity"] != bundle.identity or _parse_time(previous["result"]["timestamp"]) <= _parse_time(prediction_timestamp):
                _snapshots[request.area_id] = {"identity": bundle.identity, "result": deepcopy(result)}
                _snapshots.move_to_end(request.area_id)
                while len(_snapshots) > 128:
                    _snapshots.popitem(last=False)
    return result


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def latest_heatmap(area_id: str | None = None) -> dict:
    bundle = get_model()
    metadata = {"timestamp": None, "model_version": bundle.version, "area_id": area_id,
                "stale": False, "storage": "process-local; regenerated by predict-graph"}
    empty = {"type": "FeatureCollection", "features": [], "metadata": metadata}
    with _snapshot_lock:
        snapshot = deepcopy(_snapshots.get(area_id))
    if not snapshot or snapshot["identity"] != bundle.identity:
        metadata["reason"] = "No graph prediction is available for this area and checkpoint."
        return empty
    result = snapshot["result"]
    metadata["timestamp"] = result["timestamp"]
    max_age = bundle.config.get("inference", {}).get("max_prediction_age_minutes", 60)
    age_seconds = (datetime.now(timezone.utc) - _parse_time(result["timestamp"])).total_seconds()
    if age_seconds > max_age * 60:
        metadata.update(stale=True, reason="The latest prediction has expired. Run graph inference with fresh observations.")
        return empty
    features = []
    for node in result["nodes"]:
        properties = {key: value for key, value in node.items() if key not in ("latitude", "longitude")}
        features.append({"type": "Feature", "id": node["node_id"],
                         "geometry": {"type": "Point", "coordinates": [node["longitude"], node["latitude"]]},
                         "properties": properties})
    return {"type": "FeatureCollection", "features": features, "metadata": metadata}


def clear_predictions():
    with _snapshot_lock:
        _snapshots.clear()
