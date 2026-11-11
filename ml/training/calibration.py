"""Held-out temperature scaling and empirical prediction-correctness confidence.

Confidence is the observed accuracy in a held-out reliability bin, not the
hazard probability. Insufficient calibration data yields None, never certainty.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .losses import TASKS


def fit_calibration(logits, labels, label_mask, min_samples: int = 30) -> dict:
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels, dtype=float)
    mask = np.asarray(label_mask, dtype=bool)
    result: dict = {"method": "held-out-temperature-and-reliability", "temperatures": {},
                   "confidence": {}, "sample_counts": {}}
    for i, task in enumerate(TASKS):
        valid = mask[:, i] & np.isfinite(labels[:, i]) & np.isfinite(logits[:, i])
        x, y = logits[valid, i], labels[valid, i]
        result["sample_counts"][task] = int(len(y))
        midpoint = len(y) // 2
        if midpoint < min_samples or len(y) - midpoint < min_samples or len(np.unique(y[:midpoint])) < 2:
            continue
        def objective(log_temperature):
            scaled = x[:midpoint] / np.exp(log_temperature)
            return float(np.mean(np.logaddexp(0, scaled) - y[:midpoint] * scaled))
        fitted = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        if not fitted.success or not np.isfinite(fitted.fun):
            continue
        temperature = float(np.exp(fitted.x))
        result["temperatures"][task] = temperature
        probability = expit(x[midpoint:] / temperature)
        score = np.maximum(probability, 1 - probability)
        correct = ((probability >= 0.5) == y[midpoint:]).astype(float)
        edges = np.linspace(0.5, 1.0, 6)
        edges[-1] = 1.000001
        accuracy, counts = [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            members = (score >= lower) & (score < upper)
            count = int(members.sum())
            counts.append(count)
            accuracy.append(float(correct[members].mean()) if count >= 5 else None)
        result["confidence"][task] = {"bin_edges": edges.tolist(), "accuracy": accuracy, "counts": counts}
    return result


def calibrated_probabilities(logits, calibration: dict | None = None):
    values = np.asarray(logits, dtype=float)
    temperatures = (calibration or {}).get("temperatures", {})
    scale = np.asarray([temperatures.get(task, 1.0) for task in TASKS])
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Calibration temperatures must be finite positive numbers")
    return expit(values / scale)


def confidence_for_probability(calibration: dict, probability: float, task: str = "combined_risk") -> float | None:
    bins = calibration.get("confidence", {}).get(task)
    if not bins or not np.isfinite(probability) or not 0 <= probability <= 1:
        return None
    score = max(probability, 1 - probability)
    for low, high, accuracy in zip(bins["bin_edges"][:-1], bins["bin_edges"][1:], bins["accuracy"]):
        if low <= score < high:
            return float(accuracy) if accuracy is not None else None
    return None
