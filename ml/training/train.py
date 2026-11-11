"""Train with chronological, horizon-purged splits and train-only preprocessing.

Run from the repository root:
  python -m backend.ml.training.train --data data/training.parquet --config backend/ml/config.yaml

Use --data-provenance synthetic for development fixtures. Such checkpoints are
explicitly ineligible for operational inference, even after optimizer updates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from ..config import load_config, resolve_config
from ..data.feature_schema import schema_dict
from ..data.preprocessing import FeaturePreprocessor
from ..models import DisasterGNNTransformer
from .calibration import calibrated_probabilities, fit_calibration
from .evaluate import collect_predictions, forward_graph
from .losses import MultiHazardLoss, TASKS
from .metrics import compute_metrics

LOGGER = logging.getLogger(__name__)


def _device(name: str) -> torch.device:
    # CUDA AMP is supported; CPU remains the portable default on other systems.
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _training_label_statistics(dataset) -> tuple[list[float], dict]:
    counts = {task: 0 for task in (*TASKS, "lead_time")}
    positives = np.zeros(3, dtype=np.int64)
    for graph in dataset:
        for i, task in enumerate((*TASKS, "lead_time")):
            valid = graph.label_mask[:, i].bool()
            counts[task] += int(valid.sum())
            if i < 3:
                values = graph.y[valid, i]
                if not torch.all((values == 0) | (values == 1)):
                    raise ValueError(f"{task} labels must be binary observations")
                positives[i] += int(values.sum())
    weights = []
    for i, task in enumerate(TASKS):
        negative = counts[task] - int(positives[i])
        if positives[i] <= 0 or negative <= 0:
            raise ValueError(f"{task} needs labeled positive and negative training examples; labels are not generated")
        weights.append(float(np.clip(negative / positives[i], 0.01, 100.0)))
    return weights, counts


def _save_checkpoint(checkpoint: dict, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement avoids exposing partially written checkpoint files.
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".multi-hazard-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_frame(frame: pd.DataFrame, config: dict, *, data_provenance: str = "real", output: str | Path | None = None) -> dict:
    from ..data.dataset import GraphSequenceDataset, chronological_split

    config = resolve_config(config)
    if data_provenance not in {"real", "synthetic"}:
        raise ValueError("data_provenance must explicitly identify real or synthetic observations")
    if "data_provenance" in frame and frame["data_provenance"].astype(str).str.lower().eq("synthetic").any():
        data_provenance = "synthetic"
    if "is_synthetic" in frame and frame["is_synthetic"].fillna(False).astype(bool).any():
        data_provenance = "synthetic"
    train_config, data_config = config["training"], config["data"]
    seed = int(train_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_frame_, validation_frame, test_frame = chronological_split(
        frame, train_fraction=data_config["train_fraction"], val_fraction=data_config["val_fraction"],
        forecast_horizon_minutes=data_config["forecast_horizon_minutes"],
        sequence_length=data_config["sequence_length"], frequency=data_config["frequency"],
    )
    preprocessor = FeaturePreprocessor(frequency=data_config["frequency"]).fit(train_frame_)
    datasets = [GraphSequenceDataset(
        preprocessor.transform(part), preprocessor,
        sequence_length=data_config["sequence_length"],
        forecast_horizon_minutes=data_config["forecast_horizon_minutes"], graph_config=config["graph"],
    ) for part in (train_frame_, validation_frame, test_frame)]
    train_set, validation_set, test_set = datasets
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("More timestamped observations are required for nonempty purged train/validation/test windows")
    positive_weights, label_counts = _training_label_statistics(train_set)
    device = _device(train_config["device"])
    model = DisasterGNNTransformer(config).to(device)
    criterion = MultiHazardLoss(positive_weights, train_config["lead_time_weight"],
                               data_config["forecast_horizon_minutes"]).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, train_config["patience"] // 3))
    amp = bool(train_config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_loss, best_state, best_epoch, stale_epochs, optimizer_steps = float("inf"), None, 0, 0, 0
    history = []
    for epoch in range(1, train_config["epochs"] + 1):
        model.train()
        train_losses = []
        # Shuffling occurs only inside the already separated training interval.
        for index in torch.randperm(len(train_set)).tolist():
            graph = train_set[index].clone().to(device)
            if not graph.label_mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(forward_graph(model, graph), graph.y, graph.label_mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss; no checkpoint was published")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["gradient_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            train_losses.append(float(loss.detach().item()))
        validation = collect_predictions(model, validation_set, device, criterion)
        validation_loss = validation["loss"]
        if validation_loss is None or not np.isfinite(validation_loss):
            raise ValueError("Labeled validation observations are required for early stopping")
        scheduler.step(validation_loss)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss})
        LOGGER.info("epoch=%s train_loss=%.5f validation_loss=%.5f", epoch, history[-1]["train_loss"], validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= train_config["patience"]:
                break
    if best_state is None or optimizer_steps == 0:
        raise RuntimeError("No trained model state was produced")
    model.load_state_dict(best_state, strict=True)
    validation = collect_predictions(model, validation_set, device)
    calibration = fit_calibration(validation["logits"], validation["labels"], validation["label_mask"],
                                  min_samples=train_config["min_calibration_samples"])
    test = collect_predictions(model, test_set, device)
    metrics = compute_metrics(calibrated_probabilities(test["logits"], calibration),
                              test["lead_times"], test["labels"], test["label_mask"])
    metrics["split"] = "held_out_test"
    metrics["data_provenance"] = data_provenance
    now = datetime.now(timezone.utc).isoformat()
    checkpoint = {
        "format_version": 1,
        "feature_schema": schema_dict(),
        "config": model.checkpoint_config(),
        "model_state_dict": best_state,
        "preprocessor": preprocessor.to_dict(),
        "training": {
            "completed": True, "optimizer_steps": optimizer_steps, "label_counts": label_counts,
            "best_epoch": best_epoch, "data_provenance": data_provenance,
            "split": "chronological_horizon_purged", "history": history,
            "positive_weights": positive_weights,
            "intervals": [{"start": pd.to_datetime(part["timestamp"], utc=True).min().isoformat(),
                           "end": pd.to_datetime(part["timestamp"], utc=True).max().isoformat(),
                           "graph_windows": len(dataset)}
                          for part, dataset in zip((train_frame_, validation_frame, test_frame), datasets)],
        },
        "calibration": calibration,
        "metrics": metrics,
        "model_version": f"multi-hazard-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}",
        "trained_at": now,
    }
    _save_checkpoint(checkpoint, Path(output or train_config["checkpoint_path"]))
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--config", default="backend/ml/config.yaml", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-provenance", choices=("real", "synthetic"), default="real",
                        help="Synthetic development data can never produce a deployment-ready checkpoint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = pd.read_parquet(args.data) if args.data.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(args.data)
    checkpoint = train_frame(frame, load_config(args.config), data_provenance=args.data_provenance, output=args.output)
    print(json.dumps({"model_version": checkpoint["model_version"], "training": checkpoint["training"],
                      "metrics": checkpoint["metrics"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
