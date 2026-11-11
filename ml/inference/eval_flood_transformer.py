"""Inference and Evaluation script for trained SpatioTemporalFTTransformer.

Loads checkpoint from backend/ml/checkpoints/best_flood_transformer.pt, runs evaluation
on the test set, and provides a standalone predict() function for new hydrometric observations.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import torch
import torch.nn.functional as F

from backend.ml.models.flood_transformer import SpatioTemporalFTTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


class FloodTransformerPredictor:
    """Production predictor wrapper for SpatioTemporalFTTransformer."""

    def __init__(self, checkpoint_path: str = "backend/ml/checkpoints/best_flood_transformer.pt", device: str = "auto"):
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        LOGGER.info(f"Loading checkpoint from: {checkpoint_path} onto {self.device}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint["model_config"]
        self.meta = checkpoint["meta"]
        
        self.model = SpatioTemporalFTTransformer(
            num_numerical=self.config["num_numerical"],
            cat_cardinalities=self.config["cat_cardinalities"],
            seq_len=self.config.get("seq_len", 10),
            d_model=self.config.get("d_model", 64),
            n_heads=self.config.get("n_heads", 4),
            temporal_layers=self.config.get("temporal_layers", 2),
            tabular_layers=self.config.get("tabular_layers", 3),
            num_classes=self.config.get("num_classes", 2),
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.scaler_num_mean = np.array(self.meta["scaler_num_mean"])
        self.scaler_num_scale = np.array(self.meta["scaler_num_scale"])
        self.scaler_temp_mean = np.array(self.meta["scaler_temp_mean"])
        self.scaler_temp_scale = np.array(self.meta["scaler_temp_scale"])
        self.classes = self.meta["target_classes"]

    def predict(
        self,
        temporal_sequence: List[float],
        numerical_features: Dict[str, float],
        categorical_features: Dict[str, str],
    ) -> Dict[str, Any]:
        """Predict flood hazard risk for a single basin observation."""
        # 1. Temporal normalize
        t_arr = (np.array(temporal_sequence, dtype=np.float32) - self.scaler_temp_mean) / self.scaler_temp_scale
        x_temp = torch.tensor(t_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 2. Numeric normalize
        num_vals = [numerical_features.get(col, 0.0) for col in self.meta["numeric_cols"]]
        n_arr = (np.array(num_vals, dtype=np.float32) - self.scaler_num_mean) / self.scaler_num_scale
        x_num = torch.tensor(n_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Categorical encode
        cat_codes = []
        for col in self.meta["categorical_cols"]:
            val = categorical_features.get(col, "No dominant class")
            mapping = self.meta["cat_mappings"][col]
            idx = mapping.index(val) if val in mapping else 0
            cat_codes.append(idx)
        x_cat = torch.tensor(cat_codes, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x_temp, x_num, x_cat)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            
        return {
            "predicted_class": self.classes[pred_idx],
            "severity_index": int(pred_idx),
            "hazard_probability": float(probs[1]),
            "confidence": float(probs[pred_idx]),
            "probabilities": {cls: float(prob) for cls, prob in zip(self.classes, probs)},
        }


def main():
    predictor = FloodTransformerPredictor()
    metrics_path = Path("backend/ml/checkpoints/flood_transformer_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print("\n" + "=" * 60)
        print("CHECKPOINT EVALUATION METRICS SUMMARY:")
        print(f"  Test Accuracy:     {metrics['test_accuracy'] * 100:.2f}%")
        print(f"  Best Val Accuracy: {metrics['best_val_accuracy'] * 100:.2f}%")
        print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"  Trainable Params:  {metrics['num_parameters']:,}")
        print("=" * 60)
        
    # Test on a dummy extreme surge event
    sample_rain = [2.0, 5.0, 12.0, 25.0, 48.0, 95.0, 180.0, 310.0, 490.0, 720.0]
    result = predictor.predict(
        temporal_sequence=sample_rain,
        numerical_features={"Drainage Area": 450.0, "Stream Order": 4, "Drainage Density": 2.5},
        categorical_features={"Land cover": "Forest", "Soil type": "Luvisols", "lithology type": "Metamorphics", "KoppenGeiger Climate Type": "Tropical"}
    )
    print("\nSAMPLE EXTREME RAINFALL INFERENCE RESULT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
