"""Train-only normalization with causal timestamp alignment and explicit masks.

Missing observations are median-imputed from training data and accompanied by a
zero availability mask. Features entirely absent in the training set receive a
*standardized-neutral* value of zero and mask zero at serving time, even if newly
observed: the model cannot interpret a sensor it never saw during training.
This zero is never presented as a measured physical value. No backward filling,
interpolation from future records, or fitting on validation/test data occurs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .feature_schema import (
    DYNAMIC_FEATURES, LABEL_COLUMNS, SATELLITE_FEATURES, STATIC_CATEGORICAL_FEATURES,
    STATIC_FEATURES, TEMPORAL_FEATURES, feature_columns, schema_dict,
)


class FeaturePreprocessor:
    def __init__(self, frequency: str = "1h", *, interval_minutes: int | None = None, max_ffill_steps: int = 0):
        self.frequency = f"{interval_minutes}min" if interval_minutes is not None else frequency
        self.interval = pd.Timedelta(self.frequency)
        if self.interval <= pd.Timedelta(0):
            raise ValueError("frequency must be a positive, fixed-duration interval")
        if max_ffill_steps < 0:
            raise ValueError("max_ffill_steps cannot be negative")
        self.max_ffill_steps = int(max_ffill_steps)
        self.statistics: dict[str, dict[str, float | bool]] = {}
        self.category_vocabularies: dict[str, dict[str, int]] = {}
        self.weather_sources: dict[str, str] = {}
        self.fitted = False

    @property
    def interval_minutes(self) -> float:
        return self.interval.total_seconds() / 60

    @property
    def input_dims(self) -> dict[str, int]:
        return {branch: len(feature_columns(branch)) for branch in ("temporal", "static", "satellite")}

    def align(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Right-edge bins only consume observations at or before their timestamp."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("At least one timestamped node observation is required")
        required = {"node_id", "timestamp"}
        if not required.issubset(frame.columns):
            raise ValueError("Input requires node_id and timestamp columns")
        source = frame.copy()
        if source["node_id"].isna().any():
            raise ValueError("node_id must not be missing")
        source["node_id"] = source["node_id"].astype(str)
        if source["node_id"].str.strip().eq("").any():
            raise ValueError("node_id must not be empty")
        source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="raise")
        if source["timestamp"].isna().any():
            raise ValueError("timestamp must not be missing")
        numeric = set(DYNAMIC_FEATURES + STATIC_FEATURES + SATELLITE_FEATURES + LABEL_COLUMNS) - set(STATIC_CATEGORICAL_FEATURES)
        numeric |= {"latitude", "longitude", "satellite_temperature", "satellite_humidity", "satellite_rainfall"}
        for column in numeric:
            if column not in source:
                source[column] = np.nan
            source[column] = pd.to_numeric(source[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for column in STATIC_CATEGORICAL_FEATURES:
            if column not in source:
                source[column] = None

        aligned_nodes = []
        for node_id, group in source.groupby("node_id", sort=True):
            group = group.sort_values("timestamp", kind="stable").set_index("timestamp")
            # Duplicate timestamps often indicate a source merge. Last-known
            # values are retained without counting duplicated rainfall twice.
            group = group.groupby(level=0, sort=True).last()
            sampled = group.resample(self.frequency, label="right", closed="right").last()
            for rain in ("rainfall", "satellite_rainfall"):
                sampled[rain] = group[rain].resample(self.frequency, label="right", closed="right").sum(min_count=1)
            # The final partly observed interval has not ended and therefore
            # cannot be labelled as a complete future timestamp.
            sampled = sampled[sampled.index <= group.index.max()]
            # Drop trailing bins that contained no actual observations
            while len(sampled) > 1 and sampled.iloc[-1].isna().all():
                sampled = sampled.iloc[:-1]
            if sampled.empty:
                raise ValueError(f"Node {node_id} has no completed alignment interval")
            sampled["node_id"] = node_id
            # Terrain and coordinates persist once known, never before first observation.
            for column in STATIC_FEATURES + ("latitude", "longitude", "river_id", "downstream_node_id"):
                if column in sampled:
                    sampled[column] = sampled[column].ffill()
            xyz = sampled[["accelerometer_x", "accelerometer_y", "accelerometer_z"]]
            sampled["accelerometer_magnitude"] = np.sqrt(xyz.pow(2).sum(axis=1, min_count=3))
            sampled["accelerometer_change"] = np.sqrt(xyz.diff().pow(2).sum(axis=1, min_count=3))
            hours = self.interval.total_seconds() / 3600
            for column in ("soil_moisture", "water_level"):
                sampled[f"{column}_rate"] = sampled[column].diff() / hours
            for period in (1, 3, 6, 24):
                # Require every interval in the accumulation window. A partial
                # rain history is missing, rather than a falsely small rainfall.
                count = math.ceil(period / hours)
                sampled[f"rainfall_{period}h"] = sampled["rainfall"].rolling(f"{period}h", min_periods=count).sum() if period >= hours else np.nan
            denominator = sampled["sentinel1_vh"].where(sampled["sentinel1_vh"].abs() > 1e-12)
            sampled["vv_vh_ratio"] = sampled["sentinel1_vv"] / denominator
            aligned_nodes.append(sampled.reset_index())
        return pd.concat(aligned_nodes, ignore_index=True).sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    def _raw_features(self, aligned: pd.DataFrame, branch: str, *, fitting: bool = False) -> pd.DataFrame:
        features = {"temporal": TEMPORAL_FEATURES, "static": STATIC_FEATURES, "satellite": SATELLITE_FEATURES}[branch]
        values = pd.DataFrame(index=aligned.index)
        for feature in features:
            column = feature
            if branch == "satellite" and feature in ("temperature", "humidity", "rainfall"):
                alias = f"satellite_{feature}"
                # If a separate weather source exists, its missing values remain
                # missing; otherwise use the corresponding dynamic weather input.
                if fitting:
                    self.weather_sources[feature] = alias if aligned[alias].notna().any() else feature
                column = self.weather_sources[feature]
            series = aligned[column]
            if feature in STATIC_CATEGORICAL_FEATURES:
                categories = series.map(lambda value: str(value).strip() if pd.notna(value) else None)
                categories = categories.mask(categories.eq(""))
                if fitting:
                    self.category_vocabularies[feature] = {value: index + 1 for index, value in enumerate(sorted(categories.dropna().unique()))}
                series = categories.map(self.category_vocabularies.get(feature, {}))
            values[feature] = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return values

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        aligned = self.align(frame)
        self.statistics = {}
        self.category_vocabularies = {}
        self.weather_sources = {}
        for branch in ("temporal", "static", "satellite"):
            raw = self._raw_features(aligned, branch, fitting=True)
            for feature in raw:
                observed = raw[feature].dropna().to_numpy(dtype=np.float64)
                learned = bool(observed.size)
                self.statistics[f"{branch}__{feature}"] = {
                    "learned": learned,
                    "median": float(np.median(observed)) if learned else 0.0,
                    "mean": float(np.mean(observed)) if learned else 0.0,
                    "scale": max(float(np.std(observed)), 1e-8) if learned else 1.0,
                }
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Fit the preprocessor on training data before transform")
        aligned = self.align(frame)
        output = aligned.copy()
        normalized = {}
        for branch in ("temporal", "static", "satellite"):
            raw = self._raw_features(aligned, branch)
            for feature in raw:
                key = f"{branch}__{feature}"
                stats = self.statistics[key]
                observed = raw[feature].notna()
                if not stats["learned"]:
                    normalized[key] = np.zeros(len(aligned), dtype=np.float32)
                    normalized[f"{key}__available"] = np.zeros(len(aligned), dtype=np.float32)
                    continue
                values = raw[feature]
                if self.max_ffill_steps:
                    values = values.groupby(aligned["node_id"]).ffill(limit=self.max_ffill_steps)
                values = values.fillna(stats["median"])
                normalized[key] = ((values - stats["mean"]) / stats["scale"]).astype(np.float32)
                normalized[f"{key}__available"] = observed.to_numpy(dtype=np.float32)
        return pd.concat([output, pd.DataFrame(normalized, index=output.index)], axis=1)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise ValueError("Cannot persist an unfitted preprocessor")
        return {
            "schema": schema_dict(), "frequency": self.frequency,
            "max_ffill_steps": self.max_ffill_steps,
            "statistics": self.statistics,
            "category_vocabularies": self.category_vocabularies,
            "weather_sources": self.weather_sources,
            "missing_policy": "train_median_with_availability_mask; unseen_feature_standardized_neutral",
        }

    state_dict = to_dict

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "FeaturePreprocessor":
        if state.get("schema") != schema_dict():
            raise ValueError("Checkpoint feature schema does not match this runtime")
        result = cls(state["frequency"], max_ffill_steps=int(state.get("max_ffill_steps", 0)))
        expected = {f"{branch}__{feature}" for branch, features in (("temporal", TEMPORAL_FEATURES), ("static", STATIC_FEATURES), ("satellite", SATELLITE_FEATURES)) for feature in features}
        statistics = state.get("statistics", {})
        if set(statistics) != expected:
            raise ValueError("Checkpoint preprocessing statistics are incomplete")
        for value in statistics.values():
            if not isinstance(value.get("learned"), bool) or not all(np.isfinite(value.get(key, np.nan)) for key in ("mean", "median", "scale")) or value["scale"] <= 0:
                raise ValueError("Checkpoint preprocessing statistics are invalid")
        vocabularies = state.get("category_vocabularies", {})
        if set(vocabularies) != set(STATIC_CATEGORICAL_FEATURES):
            raise ValueError("Checkpoint categorical encoders are missing")
        for vocab in vocabularies.values():
            if not isinstance(vocab, dict) or sorted(vocab.values()) != list(range(1, len(vocab) + 1)) or not all(isinstance(key, str) for key in vocab):
                raise ValueError("Checkpoint categorical encoder is invalid")
        result.statistics = statistics
        result.category_vocabularies = vocabularies
        weather_sources = state.get("weather_sources", {})
        if set(weather_sources) != {"temperature", "humidity", "rainfall"} or any(source not in (name, f"satellite_{name}") for name, source in weather_sources.items()):
            raise ValueError("Checkpoint weather source mapping is invalid")
        result.weather_sources = weather_sources
        result.fitted = True
        return result

    from_state_dict = from_dict
