"""Versioned feature order. Checkpoints include this schema verbatim.

Rainfall is incremental millimetres per observation; rates are per hour. Satellite
VV/VH must use linear power (not dB) before their ratio is calculated. Temperature,
humidity and rainfall appear in both the temporal and satellite/weather contracts;
``satellite_<name>`` columns can supply a separate weather source.
"""

SCHEMA_VERSION = "multi-hazard-v1"
DYNAMIC_FEATURES = (
    "soil_moisture", "water_level", "rainfall", "temperature", "humidity",
    "accelerometer_x", "accelerometer_y", "accelerometer_z", "tilt", "vibration",
)
DERIVED_FEATURES = (
    "accelerometer_magnitude", "accelerometer_change", "soil_moisture_rate",
    "water_level_rate", "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_24h",
)
TEMPORAL_FEATURES = DYNAMIC_FEATURES + DERIVED_FEATURES
STATIC_FEATURES = (
    "elevation", "slope", "aspect", "flow_accumulation", "distance_to_river",
    "soil_type", "clay_fraction", "sand_fraction", "lulc",
    "historical_flood_frequency", "historical_landslide_frequency",
)
STATIC_CATEGORICAL_FEATURES = ("soil_type", "lulc")
STATIC_NUMERIC_FEATURES = tuple(f for f in STATIC_FEATURES if f not in STATIC_CATEGORICAL_FEATURES)
SATELLITE_FEATURES = (
    "sentinel1_vv", "sentinel1_vh", "vv_vh_ratio", "ndvi", "ndwi",
    "temperature", "humidity", "rainfall",
)
SATELLITE_WEATHER_FEATURES = SATELLITE_FEATURES
EDGE_FEATURES = ("distance_km", "elevation_difference", "slope_difference", "hydrological_connection")
LABEL_COLUMNS = ("flood_label", "landslide_label", "combined_risk_label", "lead_time_minutes")
TEMPORAL_INPUT_DIM = 2 * len(TEMPORAL_FEATURES)
STATIC_INPUT_DIM = 2 * len(STATIC_FEATURES)
SATELLITE_INPUT_DIM = 2 * len(SATELLITE_FEATURES)


def feature_columns(branch: str) -> list[str]:
    """Normalized values, followed by availability masks in the same feature order."""
    features = {"temporal": TEMPORAL_FEATURES, "static": STATIC_FEATURES, "satellite": SATELLITE_FEATURES}[branch]
    return [f"{branch}__{name}" for name in features] + [f"{branch}__{name}__available" for name in features]


def schema_dict() -> dict:
    return {
        "version": SCHEMA_VERSION,
        "dynamic_features": list(DYNAMIC_FEATURES),
        "derived_features": list(DERIVED_FEATURES),
        "temporal_features": list(TEMPORAL_FEATURES),
        "static_features": list(STATIC_FEATURES),
        "static_categorical_features": list(STATIC_CATEGORICAL_FEATURES),
        "satellite_features": list(SATELLITE_FEATURES),
        "edge_features": list(EDGE_FEATURES),
        "label_columns": list(LABEL_COLUMNS),
        "mask_layout": "all feature values followed by all availability masks",
        "temporal": feature_columns("temporal"),
        "static": feature_columns("static"),
        "satellite": feature_columns("satellite"),
    }


FEATURE_SCHEMA = schema_dict()
feature_schema = schema_dict
