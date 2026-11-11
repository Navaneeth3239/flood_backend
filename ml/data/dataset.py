"""Chronological graph snapshots and label-aligned sequence windows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .feature_schema import LABEL_COLUMNS, feature_columns
from .graph_builder import build_graph
from .preprocessing import FeaturePreprocessor


def chronological_split(
    frame: pd.DataFrame, *, train_fraction: float = 0.7, val_fraction: float = 0.15,
    forecast_horizon_minutes: float = 360, sequence_length: int = 24,
    frequency: str = "1h",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split whole timestamp groups; purge a forecast horizon at both boundaries.

    Splits have independent history windows. No validation/test features influence
    training medians, scalers or category vocabularies. Dataset construction also
    requires label timestamps to stay within the same split.
    """
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Train/validation fractions must leave a positive test split")
    if forecast_horizon_minutes < 0 or sequence_length < 1:
        raise ValueError("Horizon must be nonnegative and sequence_length positive")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    timestamps = np.sort(result["timestamp"].unique())
    train_end = int(len(timestamps) * train_fraction)
    validation_end = int(len(timestamps) * (train_fraction + val_fraction))
    if train_end == 0 or validation_end <= train_end or validation_end >= len(timestamps):
        raise ValueError("At least three chronological timestamp groups are required")
    first_boundary = pd.Timestamp(timestamps[train_end])
    second_boundary = pd.Timestamp(timestamps[validation_end])
    purge = pd.Timedelta(minutes=forecast_horizon_minutes)
    train = result[result["timestamp"] < first_boundary - purge].copy()
    validation = result[(result["timestamp"] >= first_boundary) & (result["timestamp"] < second_boundary - purge)].copy()
    test = result[result["timestamp"] >= second_boundary].copy()
    interval = pd.Timedelta(frequency)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if split.empty or split["timestamp"].max() - split["timestamp"].min() < interval * (sequence_length - 1) + purge:
            raise ValueError(f"{name} split has insufficient history for sequence length and forecast horizon")
    return train, validation, test


def build_graph_sample(
    transformed_frame: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    sequence_length: int = 24,
    as_of: Any = None,
    graph_config: dict | None = None,
) -> Data:
    """Build an inference-ready PyG Data snapshot containing every input node.

    An incomplete node history is an error, never a silently omitted location.
    Input must already have been transformed with checkpoint preprocessing state.
    The most recent common node timestamp is used if ``as_of`` is omitted.
    """
    if sequence_length < 1 or transformed_frame.empty:
        raise ValueError("A positive sequence length and nonempty observations are required")
    if not set(feature_columns("temporal")).issubset(transformed_frame.columns):
        raise ValueError("Graph sample requires preprocessor.transform output")
    frame = transformed_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    node_ids = sorted(frame["node_id"].astype(str).unique())
    frame["node_id"] = frame["node_id"].astype(str)
    timestamp = pd.Timestamp(as_of) if as_of is not None else frame.groupby("node_id")["timestamp"].max().min()
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    timestamp = timestamp.floor(preprocessor.frequency)
    expected = pd.date_range(end=timestamp, periods=sequence_length, freq=preprocessor.frequency)
    temporal, static, satellite, metadata = [], [], [], []
    for node_id in node_ids:
        records = frame[frame["node_id"] == node_id].set_index("timestamp").sort_index()
        if records.index.has_duplicates:
            raise ValueError(f"Node {node_id} has duplicate aligned timestamps")
        if not expected.isin(records.index).all():
            raise ValueError(f"Node {node_id} requires {sequence_length} aligned history steps ending {timestamp.isoformat()}")
        window = records.loc[expected]
        last = window.iloc[-1]
        temporal.append(window[feature_columns("temporal")].to_numpy(dtype=np.float32))
        static.append(last[feature_columns("static")].to_numpy(dtype=np.float32))
        satellite.append(last[feature_columns("satellite")].to_numpy(dtype=np.float32))
        metadata.append(last.to_dict())
    metadata_frame = pd.DataFrame(metadata)
    edge_index, edge_attr = build_graph(metadata_frame, **(graph_config or {}))
    return Data(
        x_temporal=torch.from_numpy(np.stack(temporal)),
        x_static=torch.from_numpy(np.stack(static)),
        x_satellite=torch.from_numpy(np.stack(satellite)),
        edge_index=edge_index, edge_attr=edge_attr,
        node_ids=node_ids,
        latitude=torch.tensor(metadata_frame["latitude"].to_numpy(dtype=np.float64), dtype=torch.float64),
        longitude=torch.tensor(metadata_frame["longitude"].to_numpy(dtype=np.float64), dtype=torch.float64),
        timestamp=timestamp.isoformat(), num_nodes=len(node_ids),
    )


class GraphSequenceDataset(Dataset):
    """Each item is a full graph at one time with labels at its forecast horizon.

    Target columns are ``flood_label``, ``landslide_label``, ``combined_risk_label``
    and ``lead_time_minutes``. Labels are never interpolated or filled. Their
    availability is recorded in ``label_mask[N,4]``. Unknown labels are neutral
    tensor storage only and must be excluded by the loss using that mask.
    """

    def __init__(
        self, transformed_frame: pd.DataFrame, preprocessor: FeaturePreprocessor,
        sequence_length: int = 24, forecast_horizon_minutes: float = 360,
        graph_config: dict | None = None, timestamps: list[Any] | None = None,
    ):
        if sequence_length < 1 or forecast_horizon_minutes < 0:
            raise ValueError("Sequence length must be positive and horizon nonnegative")
        self.frame = transformed_frame.copy()
        self.frame["timestamp"] = pd.to_datetime(self.frame["timestamp"], utc=True)
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length
        self.horizon = pd.Timedelta(minutes=forecast_horizon_minutes)
        if self.horizon % preprocessor.interval != pd.Timedelta(0):
            raise ValueError("Forecast horizon must be a multiple of the alignment interval")
        self.graph_config = graph_config or {}
        nodes = sorted(self.frame["node_id"].astype(str).unique())
        earliest = self.frame.groupby("node_id")["timestamp"].min().max() + (sequence_length - 1) * preprocessor.interval
        latest = self.frame.groupby("node_id")["timestamp"].max().min() - self.horizon
        candidates = pd.date_range(earliest, latest, freq=preprocessor.frequency) if latest >= earliest else []
        if timestamps is not None:
            requested = set(pd.to_datetime(timestamps, utc=True))
            candidates = [timestamp for timestamp in candidates if timestamp in requested]
        indexed = self.frame.set_index(["timestamp", "node_id"])
        self.timestamps = []
        for timestamp in candidates:
            target = timestamp + self.horizon
            if target not in indexed.index.get_level_values("timestamp"):
                continue
            labels = indexed.xs(target, level="timestamp").reindex(nodes).reindex(columns=LABEL_COLUMNS)
            if labels.notna().to_numpy().any():
                self.timestamps.append(timestamp)

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, index: int) -> Data:
        timestamp = self.timestamps[index]
        sample = build_graph_sample(self.frame, self.preprocessor, self.sequence_length, timestamp, self.graph_config)
        target_timestamp = timestamp + self.horizon
        target = self.frame[self.frame["timestamp"] == target_timestamp].set_index("node_id").reindex(sample.node_ids)
        values = target.reindex(columns=LABEL_COLUMNS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        for column in range(3):
            if ((values[:, column][mask[:, column]] < 0) | (values[:, column][mask[:, column]] > 1)).any():
                raise ValueError("Classification labels must be probabilities in [0,1]")
        if (values[:, 3][mask[:, 3]] < 0).any():
            raise ValueError("Lead-time labels must be nonnegative minutes")
        sample.y = torch.from_numpy(np.where(mask, values, 0.0))
        sample.label_mask = torch.from_numpy(mask)
        sample.target_timestamp = target_timestamp.isoformat()
        return sample
