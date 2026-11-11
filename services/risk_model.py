"""GNN-Transformer Flood Prediction Engine: Spatial Message-Passing Graph Neural Network + Temporal Multi-Head Self-Attention."""
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


class SpatialGNNConv(nn.Module):
    """Spatial Graph Convolution: aggregates neighbor features across hydrological and road topologies."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Standard GCN: D^(-0.5) * A * D^(-0.5) * X * W
        deg = torch.sum(adj, dim=-1, keepdim=True)
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-6), -0.5)
        norm_adj = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(norm_adj, support) + self.bias
        return F.relu(output)


class TemporalTransformerBlock(nn.Module):
    """Temporal Transformer: Multi-head attention across time sequence of rainfall, river depth, soil moisture."""

    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1) if causal else None
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class GNNTransformerFloodModel(nn.Module):
    """Unified GNN-Transformer architecture for spatial-temporal flash flood and landslide risk forecasting."""

    def __init__(self, node_features: int = 8, hidden_dim: int = 32):
        super().__init__()
        # Spatial GNN layers
        self.gnn1 = SpatialGNNConv(node_features, hidden_dim)
        self.gnn2 = SpatialGNNConv(hidden_dim, hidden_dim)

        # Temporal Transformer block
        self.transformer = TemporalTransformerBlock(d_model=hidden_dim, nhead=4)

        # Prediction Heads
        self.flood_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # 4 classes: Low, Medium, High, Critical
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, return_sequence: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial propagation
        h = self.gnn1(x, adj)
        h = self.gnn2(h, adj)

        # Accept legacy (nodes, features) and real (time, nodes, features) input.
        h_seq = h.unsqueeze(1) if h.ndim == 2 else h.transpose(0, 1)
        if x.ndim == 3:
            position = torch.arange(h_seq.shape[1], device=h.device, dtype=h.dtype).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, h_seq.shape[-1], 2, device=h.device, dtype=h.dtype)
                                  * (-math.log(10000.0) / h_seq.shape[-1]))
            encoding = torch.zeros_like(h_seq[0])
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            h_seq = h_seq + encoding.unsqueeze(0)
        h_all = self.transformer(h_seq, causal=True)
        h_trans = h_all.transpose(0, 1) if return_sequence else h_all[:, -1]

        flood_prob = self.flood_prob_head(h_trans)
        severity_logits = self.severity_head(h_trans)
        return flood_prob, severity_logits


class FloodRiskEngine:
    def __init__(self):
        self.model = GNNTransformerFloodModel(node_features=8, hidden_dim=32)
        self.model.eval()
        self.transformer_predictor = None
        checkpoint_path = "backend/ml/checkpoints/best_flood_transformer.pt"
        try:
            from backend.ml.inference.eval_flood_transformer import FloodTransformerPredictor
            import os
            if os.path.exists(checkpoint_path):
                self.transformer_predictor = FloodTransformerPredictor(checkpoint_path=checkpoint_path, device="cpu")
        except Exception:
            self.transformer_predictor = None

    def predict_graph_risk(
        self,
        unified_graph: nx.DiGraph,
        rainfall_intensity_mm: float = 45.0,
        soil_saturation_pct: float = 80.0,
    ) -> Dict[str, Any]:
        """
        Executes GNN-Transformer forward pass over the unified spatial graph.
        Returns node/edge flood probabilities, severity classifications, and hazard extent.
        """
        nodes = list(unified_graph.nodes())
        n_count = len(nodes)
        if n_count == 0:
            return {"node_predictions": {}, "high_risk_zones": [], "overall_severity": "low"}

        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Build feature matrix X (N, 8)
        X = np.zeros((n_count, 8), dtype=np.float32)
        for i, node in enumerate(nodes):
            d = unified_graph.nodes[node]
            # Normalization
            lat = d.get("lat", 0.0)
            lng = d.get("lng", 0.0)
            elev = d.get("elevation", 200.0) / 1000.0
            slope = d.get("slope", 2.0) / 45.0
            rain = (d.get("rainfall", rainfall_intensity_mm)) / 100.0
            water = d.get("water_level", 1.5) / 10.0
            soil = (d.get("soil_moisture", soil_saturation_pct)) / 100.0
            dist_riv = min(1.0, d.get("distance_to_river", 500.0) / 1000.0)

            X[i] = [lat, lng, elev, slope, rain, water, soil, dist_riv]

        # Build adjacency matrix A (N, N)
        A = np.eye(n_count, dtype=np.float32)
        for u, v in unified_graph.edges():
            if u in node_to_idx and v in node_to_idx:
                ui, vi = node_to_idx[u], node_to_idx[v]
                A[ui, vi] = 1.0
                A[vi, ui] = 1.0  # Undirected message passing

        with torch.no_grad():
            x_tensor = torch.from_numpy(X)
            adj_tensor = torch.from_numpy(A)
            flood_probs, severity_logits = self.model(x_tensor, adj_tensor)

            probs = flood_probs.squeeze(-1).numpy()
            severities = torch.argmax(severity_logits, dim=-1).numpy()

        severity_labels = ["low", "medium", "high", "critical"]
        predictions = {}
        high_risk_points = []
        critical_count = 0
        high_count = 0

        for i, node in enumerate(nodes):
            d = unified_graph.nodes[node]
            domain = d.get("domain", "road")

            # Calibrate model output with hydro-topographic physical baseline
            dist_to_riv = d.get("distance_to_river", 500.0)
            elevation = d.get("elevation", 200.0)

            raw_prob = float(probs[i])
            # Physics-guided blend: low-elevation + close to river + intense rain increases probability
            phys_factor = (rainfall_intensity_mm / 60.0) * max(0.0, 1.0 - (dist_to_riv / 400.0))
            calibrated_prob = min(0.98, max(0.05, 0.4 * raw_prob + 0.6 * phys_factor))

            if domain == "shelter":
                calibrated_prob = 0.02  # Designated safe elevation shelter
            elif domain == "river":
                calibrated_prob = min(0.99, max(0.65, calibrated_prob + 0.3))

            if calibrated_prob >= 0.75:
                sev = "critical"
                critical_count += 1
            elif calibrated_prob >= 0.50:
                sev = "high"
                high_count += 1
            elif calibrated_prob >= 0.25:
                sev = "medium"
            else:
                sev = "low"

            predictions[node] = {
                "id": node,
                "domain": domain,
                "name": d.get("name", node),
                "lat": d.get("lat"),
                "lng": d.get("lng"),
                "flood_probability": round(calibrated_prob, 3),
                "severity": sev,
                "is_blocked": calibrated_prob >= 0.70 and domain == "road",
            }

            # Update graph node in-place with prediction
            unified_graph.nodes[node]["predicted_flood_prob"] = calibrated_prob
            unified_graph.nodes[node]["predicted_severity"] = sev

            if calibrated_prob >= 0.60 and d.get("lat") and d.get("lng"):
                high_risk_points.append({
                    "node_id": node,
                    "lat": d["lat"],
                    "lng": d["lng"],
                    "probability": round(calibrated_prob, 3),
                    "severity": sev,
                })

        overall_sev = "critical" if critical_count > 3 else ("high" if high_count > 5 else "medium")

        return {
            "node_predictions": predictions,
            "high_risk_zones": high_risk_points,
            "overall_severity": overall_sev,
            "critical_nodes_count": critical_count,
            "high_nodes_count": high_count,
            "mean_flood_probability": round(float(np.mean([p["flood_probability"] for p in predictions.values()])), 3),
            "model_architecture": "GNN-Spatial-Conv + Temporal-Transformer (PyTorch)",
        }
