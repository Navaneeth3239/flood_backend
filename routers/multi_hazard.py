"""The model executes in FastAPI's thread pool, not its async event loop."""
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from ml.inference.model_loader import ModelNotReady, get_model
from ml.inference.schemas import GraphPredictionRequest, NodePredictionRequest
from ml.inference.predictor import latest_heatmap, predict_graph

router = APIRouter(prefix="/multi-hazard", tags=["multi-hazard"])
logger = logging.getLogger(__name__)


def _unavailable(exc: ModelNotReady):
    return HTTPException(status_code=503, detail={"code": "MODEL_NOT_READY", "status": exc.status, "reason": exc.reason})


@router.get("/model-info")
def model_info():
    from services.multi_hazard_status import model_info as info
    return info()


@router.post("/predict-graph")
def graph_prediction(payload: GraphPredictionRequest):
    try:
        return predict_graph(payload)
    except ModelNotReady as exc:
        raise _unavailable(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_FEATURE_HISTORY", "reason": str(exc)}) from exc
    except Exception as exc:
        logger.exception("Multi-hazard graph inference failed")
        raise HTTPException(status_code=503, detail={"code": "INFERENCE_FAILED", "reason": "Model inference failed; no predictions were published."}) from exc


@router.post("/predict-node")
def node_prediction(payload: NodePredictionRequest):
    try:
        result = predict_graph(payload.as_graph(), publish=False)
        return next(node for node in result["nodes"] if node["node_id"] == payload.node.node_id)
    except ModelNotReady as exc:
        raise _unavailable(exc) from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_FEATURE_HISTORY", "reason": str(exc)}) from exc
    except Exception as exc:
        logger.exception("Multi-hazard node inference failed")
        raise HTTPException(status_code=503, detail={"code": "INFERENCE_FAILED", "reason": "Model inference failed; no predictions were published."}) from exc


@router.get("/heatmap/latest")
def heatmap(area_id: str | None = Query(None, max_length=128)):
    try:
        return latest_heatmap(area_id)
    except ModelNotReady as exc:
        raise _unavailable(exc) from exc


@router.get("/metrics")
def metrics():
    try:
        bundle = get_model(require_ready=False)
    except ModelNotReady as exc:
        return {"available": False, "status": exc.status, "message": "metrics unavailable - labeled dataset required", "metrics": None}
    values = bundle.checkpoint.get("metrics")
    real = bundle.checkpoint["training"].get("data_provenance") == "real"
    return {"available": bool(real and values), "model_version": bundle.version,
            "message": None if real and values else "metrics unavailable - labeled dataset required",
            "metrics": values if real and values else None}
