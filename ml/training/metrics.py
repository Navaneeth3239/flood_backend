"""Metrics computed exclusively from observed labels and model predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score,
)

from .losses import TASKS

UNAVAILABLE = "metrics unavailable - labeled dataset required"


def compute_metrics(probabilities, lead_times, labels, label_mask) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    labels, mask = np.asarray(labels, dtype=float), np.asarray(label_mask, dtype=bool)
    if labels.size == 0 or not mask.any():
        return {"status": "unavailable", "message": UNAVAILABLE}
    result: dict = {"status": "available"}
    for i, task in enumerate(TASKS):
        valid = mask[:, i] & np.isfinite(labels[:, i]) & np.isfinite(probabilities[:, i])
        if not valid.any():
            result[task] = {"status": "unavailable", "message": UNAVAILABLE}
            continue
        y, p = labels[valid, i].astype(int), probabilities[valid, i]
        pred, two_classes = (p >= 0.5).astype(int), len(np.unique(y)) == 2
        result[task] = {
            "sample_count": int(len(y)),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
            "pr_auc": float(average_precision_score(y, p)) if two_classes else None,
            "brier_score": float(brier_score_loss(y, p)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        }
    lead_times = np.asarray(lead_times, dtype=float)
    valid = mask[:, 3] & np.isfinite(labels[:, 3]) & np.isfinite(lead_times)
    result["lead_time"] = ({
        "sample_count": int(valid.sum()),
        "mae_minutes": float(mean_absolute_error(labels[valid, 3], lead_times[valid])),
        "rmse_minutes": float(np.sqrt(np.mean((labels[valid, 3] - lead_times[valid]) ** 2))),
    } if valid.any() else {"status": "unavailable", "message": UNAVAILABLE})
    return result
