"""Masked multi-task loss; missing labels never become negative examples."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

TASKS = ("flood", "landslide", "combined_risk")


class MultiHazardLoss(nn.Module):
    def __init__(self, positive_weights=None, lead_time_weight: float = 0.1, horizon_minutes: float = 360):
        super().__init__()
        self.register_buffer("positive_weights", torch.as_tensor(
            positive_weights if positive_weights is not None else [1.0, 1.0, 1.0], dtype=torch.float32))
        self.lead_time_weight = lead_time_weight
        self.horizon_minutes = horizon_minutes

    def forward(self, outputs: dict, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        loss = sum(value.sum() * 0 for value in outputs.values())
        for index, task in enumerate(TASKS):
            valid = label_mask[:, index].bool() & torch.isfinite(labels[:, index])
            if valid.any():
                loss = loss + F.binary_cross_entropy_with_logits(
                    outputs[f"{task}_logits"][valid], labels[valid, index],
                    pos_weight=self.positive_weights[index],
                )
        valid = label_mask[:, 3].bool() & torch.isfinite(labels[:, 3])
        if valid.any():
            # Scale minutes so regression gradients do not overwhelm BCE.
            loss = loss + self.lead_time_weight * F.huber_loss(
                outputs["lead_time"][valid] / self.horizon_minutes,
                labels[valid, 3] / self.horizon_minutes,
                delta=0.1,
            )
        return loss
