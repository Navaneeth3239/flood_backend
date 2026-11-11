"""Synthetic fixtures test mechanics only; they do not establish model skill."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.ml.data.dataset import GraphSequenceDataset, build_graph_sample, chronological_split
from backend.ml.data.feature_schema import (
    DYNAMIC_FEATURES, DERIVED_FEATURES, EDGE_FEATURES, SATELLITE_FEATURES,
    STATIC_FEATURES, TEMPORAL_INPUT_DIM, feature_columns, feature_schema,
)
from backend.ml.data.graph_builder import build_graph
from backend.ml.data.preprocessing import FeaturePreprocessor


def observations(periods=30, nodes=("a", "b")):
    rows = []
    for hour, timestamp in enumerate(pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")):
        for index, node in enumerate(nodes):
            rows.append({
                "node_id": node, "timestamp": timestamp,
                "latitude": 11.6 + index * 0.01, "longitude": 76.1,
                "soil_moisture": float(hour), "water_level": float(hour * 2),
                "rainfall": 1.0, "temperature": 20.0 + hour,
                "accelerometer_x": 3.0, "accelerometer_y": 4.0, "accelerometer_z": 0.0,
                "elevation": 100.0 - index * 10, "slope": 5.0 + index,
                "soil_type": "clay" if index == 0 else "sandy", "lulc": "forest",
                "sentinel1_vv": 0.6, "sentinel1_vh": 0.2,
                "flood_label": float(hour % 2), "landslide_label": float((hour + 1) % 2),
                "combined_risk_label": float(hour % 2), "lead_time_minutes": float(hour),
            })
    return pd.DataFrame(rows)


def test_schema_is_complete_and_masks_have_same_order():
    assert len(DYNAMIC_FEATURES) == 10
    assert len(DERIVED_FEATURES) == 8
    assert len(STATIC_FEATURES) == 11
    assert len(SATELLITE_FEATURES) == 8
    assert len(EDGE_FEATURES) == 4
    assert TEMPORAL_INPUT_DIM == 36
    for branch, size in (("temporal", 36), ("static", 22), ("satellite", 16)):
        columns = feature_columns(branch)
        assert len(columns) == size
        assert columns[size // 2:] == [f"{column}__available" for column in columns[:size // 2]]
        assert feature_schema()[branch] == columns


def test_train_statistics_round_trip_and_unseen_sensors_are_masked():
    train = observations()
    train.loc[train["node_id"].eq("a") & train["timestamp"].eq(train["timestamp"].min()), "soil_moisture"] = np.nan
    processor = FeaturePreprocessor().fit(train)
    state = json.loads(json.dumps(processor.to_dict(), allow_nan=False))
    restored = FeaturePreprocessor.from_dict(state)
    pd.testing.assert_frame_equal(processor.transform(train), restored.transform(train))
    future = observations()
    future["temperature"] = 10000.0
    future["humidity"] = 85.0  # never measured in training
    future["soil_type"] = "unseen soil"
    transformed = restored.transform(future)
    assert transformed["temporal__temperature"].min() > 100
    assert (transformed["temporal__humidity"] == 0).all()
    assert (transformed["temporal__humidity__available"] == 0).all()
    assert (transformed["static__soil_type__available"] == 0).all()
    assert restored.to_dict() == state
    earliest_a = processor.transform(train).query("node_id == 'a'").iloc[0]
    assert earliest_a["temporal__soil_moisture__available"] == 0
    assert pd.isna(earliest_a["soil_moisture"])


def test_alignment_and_derived_features_are_causal():
    frame = observations(nodes=("a",))
    processor = FeaturePreprocessor().fit(frame)
    aligned = processor.align(frame)
    assert aligned.iloc[0]["accelerometer_magnitude"] == 5
    assert pd.isna(aligned.iloc[0]["accelerometer_change"])
    assert aligned.iloc[1]["soil_moisture_rate"] == 1
    assert aligned.iloc[1]["water_level_rate"] == 2
    assert aligned.iloc[23]["rainfall_24h"] == 24
    assert pd.isna(aligned.iloc[22]["rainfall_24h"])
    assert aligned.iloc[0]["vv_vh_ratio"] == pytest.approx(3)
    modified = frame.copy()
    modified.loc[modified["timestamp"] > frame.iloc[10]["timestamp"], "rainfall"] = 1000
    before = processor.transform(frame).iloc[:11]
    after = processor.transform(modified).iloc[:11]
    pd.testing.assert_frame_equal(before, after)
    irregular = frame.iloc[:3].copy()
    irregular.iloc[-1, irregular.columns.get_loc("timestamp")] += pd.Timedelta(minutes=30)
    aligned_irregular = processor.align(irregular)
    assert aligned_irregular["timestamp"].max() == irregular.iloc[1]["timestamp"]
    assert aligned_irregular.iloc[-1]["rainfall"] == 1  # future 02:30 excluded


def test_forward_filled_values_keep_missing_sensor_mask():
    frame = observations(nodes=("a",))
    processor = FeaturePreprocessor(max_ffill_steps=1).fit(frame)
    missing = frame.copy()
    missing.loc[1, "temperature"] = np.nan
    transformed = processor.transform(missing)
    assert transformed.iloc[1]["temporal__temperature"] == transformed.iloc[0]["temporal__temperature"]
    assert transformed.iloc[1]["temporal__temperature__available"] == 0


def test_preprocessor_rejects_corrupt_state():
    state = FeaturePreprocessor().fit(observations()).to_dict()
    state["statistics"]["temporal__temperature"]["scale"] = 0
    with pytest.raises(ValueError, match="statistics"):
        FeaturePreprocessor.from_dict(state)


def test_graph_geometry_and_directed_hydrology():
    nodes = pd.DataFrame([
        {"node_id": "upstream", "latitude": 0, "longitude": 0, "elevation": 100, "slope": 5, "downstream_node_id": "downstream"},
        {"node_id": "downstream", "latitude": 0, "longitude": 1, "elevation": 10, "slope": 2},
    ])
    edges, attributes = build_graph(nodes, k_neighbors=1, max_distance_km=0)
    assert edges.dtype == torch.long
    assert tuple(edges.shape) == (2, 2)
    forward = attributes[(edges[0] == 0) & (edges[1] == 1)][0]
    reverse = attributes[(edges[0] == 1) & (edges[1] == 0)][0]
    assert forward[0].item() == pytest.approx(111.195, rel=1e-4)
    assert forward[1:].tolist() == [-90, -3, 1]
    assert reverse[1:].tolist() == [90, 3, 0]
    single, attrs = build_graph(nodes.iloc[:1])
    assert single.tolist() == [[0], [0]]
    assert attrs.tolist() == [[0, 0, 0, 0]]
    with pytest.raises(ValueError, match="coordinates"):
        build_graph(nodes.assign(latitude=100))


def test_river_connectivity_uses_directed_elevation_order():
    nodes = observations(periods=1).assign(river_id="test-river")
    edges, attributes = build_graph(nodes, k_neighbors=0, max_distance_km=0)
    assert edges.tolist() == [[0], [1]]
    assert attributes[0, 3] == 1


def test_sequence_graph_and_future_labels_are_separate():
    frame = observations()
    processor = FeaturePreprocessor().fit(frame)
    transformed = processor.transform(frame)
    graph = build_graph_sample(transformed, processor, sequence_length=4)
    assert tuple(graph.x_temporal.shape) == (2, 4, 36)
    assert tuple(graph.x_static.shape) == (2, 22)
    assert tuple(graph.x_satellite.shape) == (2, 16)
    assert graph.node_ids == ["a", "b"]
    dataset = GraphSequenceDataset(transformed, processor, sequence_length=4, forecast_horizon_minutes=60)
    first = dataset[0]
    assert first.timestamp == "2024-01-01T03:00:00+00:00"
    assert first.target_timestamp == "2024-01-01T04:00:00+00:00"
    assert first.y[:, 3].tolist() == [4, 4]
    assert first.label_mask.all()
    transformed.loc[transformed["timestamp"].eq(pd.Timestamp(first.target_timestamp)), "landslide_label"] = np.nan
    missing_labels = GraphSequenceDataset(transformed, processor, 4, 60)[0]
    assert not missing_labels.label_mask[:, 1].any()
    assert missing_labels.label_mask[:, 0].all()
    with pytest.raises(ValueError, match="history"):
        build_graph_sample(transformed, processor, sequence_length=100)


def test_chronological_split_preserves_timestamp_groups_and_embargo():
    frame = observations(periods=200)
    train, validation, test = chronological_split(frame, forecast_horizon_minutes=120, sequence_length=3)
    assert train.groupby("timestamp")["node_id"].nunique().eq(2).all()
    assert train["timestamp"].max() + pd.Timedelta(minutes=120) < validation["timestamp"].min()
    assert validation["timestamp"].max() + pd.Timedelta(minutes=120) < test["timestamp"].min()
    assert not set(train["timestamp"]) & set(validation["timestamp"])
    assert not set(validation["timestamp"]) & set(test["timestamp"])
    with pytest.raises(ValueError, match="insufficient history"):
        chronological_split(observations(periods=10))
