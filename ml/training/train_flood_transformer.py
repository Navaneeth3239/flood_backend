"""Training script for Spatio-Temporal FT-Transformer on Flash Flood Dataset.

Usage:
  ./backend/venv/bin/python -m backend.ml.training.train_flood_transformer
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.ml.models.flood_transformer import SpatioTemporalFTTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess_data(
    data_path: str,
    floodevents_path: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Loads dataset and prepares temporal, numerical, and categorical matrices."""
    LOGGER.info(f"Loading primary dataset: {data_path}")
    df = pd.read_csv(data_path)
    
    # Merge with floodevents ground truth
    if os.path.exists(floodevents_path):
        LOGGER.info(f"Merging ground truth from: {floodevents_path}")
        fe = pd.read_csv(floodevents_path)
        merged = pd.merge(df, fe[["EventID", "Peak Flood Level (m)", "Flood Type"]], on="EventID", how="left")
    else:
        LOGGER.warning(f"Ground truth file {floodevents_path} not found. Checking if target exists in dataset.")
        merged = df.copy()
        
    # Derive Flood Hazard Severity Target (High Flood Hazard vs Moderate Flood Hazard)
    if "Peak Flood Level (m)" in merged.columns:
        gauge_medians = merged.groupby("GaugeID")["Peak Flood Level (m)"].transform("median")
        merged["Peak Flood Level (m)"] = merged["Peak Flood Level (m)"].fillna(gauge_medians)
        overall_median = merged["Peak Flood Level (m)"].median()
        y_data = (merged["Peak Flood Level (m)"] > overall_median).astype(int).values
    else:
        # Fallback to stream order or composite flood hazard index
        LOGGER.info("Using hydrometric classification target")
        y_data = (merged["Stream Order"] > merged["Stream Order"].median()).astype(int).values
        
    # Temporal columns: T1d to T10d
    temporal_cols = [f"T{i}d" for i in range(1, 11)]
    categorical_cols = ["Land cover", "Soil type", "lithology type", "KoppenGeiger Climate Type"]
    numeric_cols = [
        c for c in df.columns
        if c not in ["EventID", "GaugeID", "EventNumber", *temporal_cols, *categorical_cols]
    ]
    
    # Categorical vocabulary mapping
    cat_mappings = {}
    cat_dims = []
    X_cat_list = []
    for col in categorical_cols:
        codes, uniques = pd.factorize(merged[col])
        X_cat_list.append(codes)
        cat_mappings[col] = list(uniques)
        cat_dims.append(len(uniques))
    X_cat = np.stack(X_cat_list, axis=1)
    
    # Raw numeric & temporal arrays
    X_num_raw = merged[numeric_cols].fillna(merged[numeric_cols].median()).values
    X_temp_raw = merged[temporal_cols].fillna(0).values
    
    # Chronological / Stratified Split: 80% train, 10% val, 10% test
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=y_data)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=seed, stratify=y_data[test_idx])
    
    # Scalers (fit ONLY on train split to prevent data leakage)
    scaler_num = StandardScaler()
    X_num_train = scaler_num.fit_transform(X_num_raw[train_idx])
    X_num_val = scaler_num.transform(X_num_raw[val_idx])
    X_num_test = scaler_num.transform(X_num_raw[test_idx])
    
    scaler_temp = StandardScaler()
    X_temp_train = scaler_temp.fit_transform(X_temp_raw[train_idx])
    X_temp_val = scaler_temp.transform(X_temp_raw[val_idx])
    X_temp_test = scaler_temp.transform(X_temp_raw[test_idx])
    
    return {
        "train": (X_temp_train, X_num_train, X_cat[train_idx], y_data[train_idx]),
        "val": (X_temp_val, X_num_val, X_cat[val_idx], y_data[val_idx]),
        "test": (X_temp_test, X_num_test, X_cat[test_idx], y_data[test_idx]),
        "meta": {
            "numeric_cols": numeric_cols,
            "temporal_cols": temporal_cols,
            "categorical_cols": categorical_cols,
            "cat_dims": cat_dims,
            "cat_mappings": cat_mappings,
            "scaler_num_mean": scaler_num.mean_.tolist(),
            "scaler_num_scale": scaler_num.scale_.tolist(),
            "scaler_temp_mean": scaler_temp.mean_.tolist(),
            "scaler_temp_scale": scaler_temp.scale_.tolist(),
            "target_classes": ["Low/Moderate Hazard", "Severe/High Hazard"],
        }
    }


def train_model(
    data: Dict[str, Any],
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 4,
    checkpoint_dir: str = "backend/ml/checkpoints",
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Trains the SpatioTemporalFTTransformer and evaluates it."""
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    LOGGER.info(f"Target compute device: {device}")
    
    meta = data["meta"]
    X_temp_tr, X_num_tr, X_cat_tr, y_tr = data["train"]
    X_temp_va, X_num_va, X_cat_va, y_va = data["val"]
    X_temp_te, X_num_te, X_cat_te, y_te = data["test"]
    
    # Initialize Model
    model = SpatioTemporalFTTransformer(
        num_numerical=len(meta["numeric_cols"]),
        cat_cardinalities=meta["cat_dims"],
        seq_len=10,
        d_model=d_model,
        n_heads=n_heads,
        temporal_layers=2,
        tabular_layers=3,
        num_classes=2,
        dropout=0.1,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"Initialized SpatioTemporalFTTransformer with {num_params:,} trainable parameters")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Convert to PyTorch tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_temp_tr, dtype=torch.float32),
        torch.tensor(X_num_tr, dtype=torch.float32),
        torch.tensor(X_cat_tr, dtype=torch.long),
        torch.tensor(y_tr, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_tensors = (
        torch.tensor(X_temp_va, dtype=torch.float32).to(device),
        torch.tensor(X_num_va, dtype=torch.float32).to(device),
        torch.tensor(X_cat_va, dtype=torch.long).to(device),
        torch.tensor(y_va, dtype=torch.long).to(device),
    )
    
    test_tensors = (
        torch.tensor(X_temp_te, dtype=torch.float32).to(device),
        torch.tensor(X_num_te, dtype=torch.float32).to(device),
        torch.tensor(X_cat_te, dtype=torch.long).to(device),
        torch.tensor(y_te, dtype=torch.long).to(device),
    )
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = Path(checkpoint_dir) / "best_flood_transformer.pt"
    
    best_val_acc = 0.0
    history = []
    
    LOGGER.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for b_temp, b_num, b_cat, b_y in train_loader:
            b_temp, b_num, b_cat, b_y = b_temp.to(device), b_num.to(device), b_cat.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_temp, b_num, b_cat)
            loss = criterion(out, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(b_y)
            
        scheduler.step()
        train_loss = total_loss / len(train_dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_tensors[0], val_tensors[1], val_tensors[2])
            val_pred = val_out.argmax(dim=-1)
            val_acc = (val_pred == val_tensors[3]).float().mean().item()
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_numerical": len(meta["numeric_cols"]),
                    "cat_cardinalities": meta["cat_dims"],
                    "seq_len": 10,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "temporal_layers": 2,
                    "tabular_layers": 3,
                    "num_classes": 2,
                },
                "meta": meta,
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_checkpoint_path)
            
        if epoch % 5 == 0 or epoch == epochs:
            LOGGER.info(
                f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)"
            )
            
    # Load best checkpoint for test evaluation
    LOGGER.info(f"Loading best checkpoint from {best_checkpoint_path} for final test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        test_out = model(test_tensors[0], test_tensors[1], test_tensors[2])
        test_pred = test_out.argmax(dim=-1).cpu().numpy()
        test_probs = F.softmax(test_out, dim=-1)[:, 1].cpu().numpy()
        y_true = test_tensors[3].cpu().numpy()
        
    test_acc = accuracy_score(y_true, test_pred)
    roc_auc = roc_auc_score(y_true, test_probs)
    conf_mat = confusion_matrix(y_true, test_pred).tolist()
    report = classification_report(y_true, test_pred, target_names=meta["target_classes"], output_dict=True)
    
    LOGGER.info("=" * 60)
    LOGGER.info(f"FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
    LOGGER.info(f"ROC-AUC SCORE:       {roc_auc:.4f}")
    LOGGER.info("CONFUSION MATRIX:")
    LOGGER.info(f"  TN={conf_mat[0][0]}, FP={conf_mat[0][1]}")
    LOGGER.info(f"  FN={conf_mat[1][0]}, TP={conf_mat[1][1]}")
    LOGGER.info("=" * 60)
    
    print("\n" + classification_report(y_true, test_pred, target_names=meta["target_classes"]))
    
    metrics = {
        "test_accuracy": float(test_acc),
        "roc_auc": float(roc_auc),
        "best_val_accuracy": float(best_val_acc),
        "confusion_matrix": conf_mat,
        "classification_report": report,
        "history": history,
        "num_parameters": num_params,
        "device": str(device),
    }
    
    metrics_path = Path(checkpoint_dir) / "flood_transformer_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f"Saved evaluation metrics to: {metrics_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Flood Transformer")
    parser.add_argument("--data", default="dataset/flash_flood_augmented_5000_cleaned.csv")
    parser.add_argument("--floodevents", default="/Users/vijay/Downloads/floodevents_indofloods.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default="backend/ml/checkpoints")
    args = parser.parse_args()
    
    set_seed(args.seed)
    data = load_and_preprocess_data(args.data, args.floodevents, seed=args.seed)
    train_model(
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
