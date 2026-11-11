"""Graph inference and held-out evaluation, reusable by the training command."""
from __future__ import annotations

import numpy as np
import torch

from .calibration import calibrated_probabilities
from .losses import TASKS
from .metrics import compute_metrics


def forward_graph(model, graph):
    return model(graph.x_temporal, graph.x_static, graph.x_satellite, graph.edge_index, graph.edge_attr)


@torch.inference_mode()
def collect_predictions(model, dataset, device="cpu", criterion=None) -> dict:
    model.eval()
    logits, lead_times, labels, masks, losses = [], [], [], [], []
    for graph in dataset:
        graph = graph.clone().to(device)
        outputs = forward_graph(model, graph)
        logits.append(torch.stack([outputs[f"{task}_logits"] for task in TASKS], dim=-1).cpu().numpy())
        lead_times.append(outputs["lead_time"].cpu().numpy())
        labels.append(graph.y.cpu().numpy())
        masks.append(graph.label_mask.cpu().numpy())
        if criterion is not None and graph.label_mask.any():
            losses.append(float(criterion(outputs, graph.y, graph.label_mask).item()))
    return {
        "logits": np.concatenate(logits) if logits else np.empty((0, 3)),
        "lead_times": np.concatenate(lead_times) if lead_times else np.empty(0),
        "labels": np.concatenate(labels) if labels else np.empty((0, 4)),
        "label_mask": np.concatenate(masks) if masks else np.empty((0, 4), dtype=bool),
        "loss": float(np.mean(losses)) if losses else None,
    }


def evaluate(model, dataset, device="cpu", calibration=None) -> dict:
    predictions = collect_predictions(model, dataset, device)
    return compute_metrics(calibrated_probabilities(predictions["logits"], calibration),
                           predictions["lead_times"], predictions["labels"], predictions["label_mask"])
