import json
import numpy as np
import pytest
from pydantic import ValidationError
from services.arrival_model import FEATURES, SCHEMA, make_model, validate_config, load_model, model_status, predict_arrivals
from routers.digital_twin import ArrivalGraphRequest


def config():
    return {"feature_schema": SCHEMA, "features": FEATURES, "architecture": "SparseArrivalGNNTransformer",
            "target": "remaining_seconds_and_reach_logit", "threshold_m": 0.1,
            "training_dataset": "unit-test-fixture-only", "validation": {"fixture": True},
            "mean": [0] * len(FEATURES), "std": [1] * len(FEATURES), "hidden_dim": 8}


def test_rejects_incompatible_hugging_face_models_and_invalid_normalization():
    with pytest.raises(ValueError):
        validate_config({"model_type": "graphormer"})
    bad = config(); bad["std"][0] = 0
    with pytest.raises(ValueError):
        validate_config(bad)


def test_no_random_predictions_without_a_checkpoint(monkeypatch):
    monkeypatch.delenv("TWIN_ARRIVAL_MODEL_DIR", raising=False)
    assert model_status()["available"] is False


def test_sparse_attention_is_permutation_equivariant():
    import torch
    torch.manual_seed(10)
    model = make_model(8).eval()
    x = torch.randn(3, len(FEATURES))
    neighbors = torch.tensor([[0, 1, 0], [1, 0, 2], [2, 1, 0]])
    padding = torch.tensor([[False, False, True], [False, False, False], [False, False, True]])
    expected = model(x, neighbors, padding)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    actual = model(x[permutation], inverse[neighbors[permutation]], padding[permutation])
    for first, second in zip(expected, actual):
        torch.testing.assert_close(first[permutation], second)


def test_safetensors_loading_and_masked_node_inference(tmp_path, monkeypatch):
    # Deliberately synthetic zero weights test serialization only, not model accuracy.
    import torch
    from safetensors.torch import save_file
    model = make_model(8)
    for parameter in model.parameters():
        parameter.data.zero_()
    (tmp_path / "config.json").write_text(json.dumps(config()))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    monkeypatch.setenv("TWIN_ARRIVAL_MODEL_DIR", str(tmp_path))
    data = {"cols": 2, "rows": 1, "bed": [2, 0], "depth": [0, 0], "inside": [1, 0],
            "sources": [0, 0], "paths": [0, 0], "roughness": 0.03, "netRainfall": 100,
            "stormRemaining": 60, "sourceRise": 0, "dx": 10, "dy": 10, "horizonSeconds": 60, "elapsed": 20}
    result = predict_arrivals(data)
    assert result[0] == pytest.approx(20 + np.log(2), abs=1e-5)
    assert result[1] == -1
    data["depth"][0] = 0.2
    assert predict_arrivals(data)[0] == 20
    load_model.cache_clear()


def test_rejects_grid_shape_and_nonfinite_input():
    data = {"cols": 1, "rows": 1, "bed": [0], "depth": [0], "inside": [1], "sources": [0], "paths": [0],
            "dx": 10, "dy": 10, "elapsed": 0, "horizonSeconds": 60, "roughness": 0.03,
            "netRainfall": 100, "stormRemaining": 60, "sourceRise": 0}
    assert ArrivalGraphRequest(**data)
    for key, value in [("bed", [float("nan")]), ("depth", [-1]), ("inside", [2]), ("paths", []), ("cols", 129)]:
        with pytest.raises(ValidationError):
            ArrivalGraphRequest(**{**data, key: value})
