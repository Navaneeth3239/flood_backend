"""Validated public graph-input schema. All timestamps must carry a UTC offset."""
from __future__ import annotations

from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: AwareDatetime
    soil_moisture: Finite | None = None
    water_level: Finite | None = None
    rainfall: Finite | None = None
    temperature: Finite | None = None
    humidity: Finite | None = None
    accelerometer_x: Finite | None = None
    accelerometer_y: Finite | None = None
    accelerometer_z: Finite | None = None
    tilt: Finite | None = None
    vibration: Finite | None = None
    sentinel1_vv: Finite | None = None
    sentinel1_vh: Finite | None = None
    vv_vh_ratio: Finite | None = None
    ndvi: Finite | None = None
    ndwi: Finite | None = None
    satellite_temperature: Finite | None = None
    satellite_humidity: Finite | None = None
    satellite_rainfall: Finite | None = None


class StaticFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elevation: Finite | None = None
    slope: Finite | None = None
    aspect: Finite | None = None
    flow_accumulation: Finite | None = None
    distance_to_river: Finite | None = None
    soil_type: str | int | None = None
    clay_fraction: Finite | None = None
    sand_fraction: Finite | None = None
    lulc: str | int | None = None
    historical_flood_frequency: Finite | None = None
    historical_landslide_frequency: Finite | None = None


class NodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: Identifier
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    static_features: StaticFeatures = Field(default_factory=StaticFeatures)
    downstream_node_id: Identifier | None = None
    river_id: Identifier | None = None
    readings: list[Observation] = Field(min_length=1, max_length=2048)


class GraphPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: AwareDatetime
    area_id: Identifier | None = None
    nodes: list[NodeInput] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_graph(self):
        ids = [node.node_id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node_id must be unique within a graph")
        if sum(len(node.readings) for node in self.nodes) > 65536:
            raise ValueError("A graph request supports at most 65536 observations")
        for node in self.nodes:
            if any(reading.timestamp > self.timestamp for reading in node.readings):
                raise ValueError("Readings cannot occur after the prediction timestamp")
            dates = [reading.timestamp for reading in node.readings]
            if len(set(dates)) != len(dates):
                raise ValueError("A node cannot contain duplicate observation timestamps")
        return self


class NodePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: AwareDatetime
    area_id: Identifier | None = None
    node: NodeInput
    context_nodes: list[NodeInput] = Field(default_factory=list, max_length=511)

    def as_graph(self) -> GraphPredictionRequest:
        return GraphPredictionRequest(timestamp=self.timestamp, area_id=self.area_id,
                                      nodes=[self.node, *self.context_nodes])
