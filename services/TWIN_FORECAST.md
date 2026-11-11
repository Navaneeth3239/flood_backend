# Digital twin surface forecast

`POST /api/digital-twin/surface-forecast` accepts bounds, grid size (3–21),
and elevations in metres, row-major from south-west to north-east. The viewer
samples a 21 × 21 terrain grid and clips the resulting imagery to its selected
polygon. The imagery follows Cesium terrain; it does not colour building roofs.

Open-Meteo provides 12 hourly frames at the area's centre in UTC. Weather is
shared across nodes; the fine-grained spatial variation comes from terrain,
not from independently resolved weather at every grid point. Missing terrain
or weather produces an error rather than fabricated values. Refresh fetches
new data; the panel shows fetch time (not weather model issuance time).

By default, the surface is an experimental relative hazard index using forecast
rainfall, accumulated forecast rainfall, soil moisture, relative elevation,
slope, and neighbouring cells. It is not a calibrated flood probability.
Temperature, humidity, and wind are displayed and supplied to neural features;
they do not affect the fallback index. There is no antecedent rainfall history,
river discharge or drainage network in this terrain-grid model.

To enable trained inference, install `requirements-twin.txt`, train a compatible
GNNTransformerFloodModel (8 features, 32 hidden units), and set
`TWIN_GNN_CHECKPOINT` to its checkpoint file before starting the backend.
The checkpoint must contain `feature_schema: terrain-weather-v1` and `state_dict`.
Do not label a random initialization as a trained checkpoint.

Feature order: relative elevation [0,1], slope radians/(pi/2), hourly rain/100,
cumulative forecast rain/150, humidity/100, temperature Celsius/50,
wind km/h/100, clipped volumetric soil moisture/0.5. Inputs are time × nodes × 8;
the undirected four-neighbour graph includes self-loops. Spatial convolutions
run for every hour; sinusoidal positions and temporal self-attention process
each hour with a causal attention mask. One forward pass emits every hourly
head; future weather tokens cannot influence earlier predictions. This replaces
12 repeated prefix passes. Existing single-time callers still work.

Validation: `../.venv/bin/python -m pytest tests/test_twin_forecast.py` from backend.

Open the viewer simulation menu and choose **GNN–Transformer heatmap** to show
the hourly surface overlay. Opacity changes update the existing imagery layer.

The disaster-intelligence chat writes the complete selected-area context,
simulation state, buildings, sensors, mesh nodes, and risk zones to the
configured Qdrant collection and retrieves matching knowledge for each answer.
Set `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` in `backend/.env`;
the key is server-only and is never sent to the frontend.
