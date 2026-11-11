"""Unit tests for SpatioTemporalFTTransformer and FloodTransformerPredictor."""
import pytest
import torch

from backend.ml.models.flood_transformer import (
    TemporalRainfallTransformer,
    TabularFeatureTokenizer,
    SpatioTemporalFTTransformer,
)
from backend.ml.inference.eval_flood_transformer import FloodTransformerPredictor


def test_temporal_rainfall_transformer_shape():
    batch_size = 4
    seq_len = 10
    d_model = 32
    model = TemporalRainfallTransformer(seq_len=seq_len, d_model=d_model, n_heads=2, num_layers=1)
    x = torch.randn(batch_size, seq_len)
    pooled, seq = model(x)
    assert pooled.shape == (batch_size, d_model)
    assert seq.shape == (batch_size, seq_len, d_model)


def test_tabular_feature_tokenizer_shape():
    batch_size = 4
    num_num = 15
    cat_dims = [3, 5, 2]
    d_model = 32
    tokenizer = TabularFeatureTokenizer(num_numerical=num_num, cat_cardinalities=cat_dims, d_model=d_model)
    x_num = torch.randn(batch_size, num_num)
    x_cat = torch.tensor([[0, 2, 1], [2, 4, 0], [1, 0, 1], [0, 1, 0]])
    tokens = tokenizer(x_num, x_cat)
    assert tokens.shape == (batch_size, num_num + len(cat_dims), d_model)


def test_spatio_temporal_ft_transformer_forward():
    batch_size = 4
    num_num = 20
    cat_dims = [4, 6]
    d_model = 32
    model = SpatioTemporalFTTransformer(
        num_numerical=num_num,
        cat_cardinalities=cat_dims,
        seq_len=10,
        d_model=d_model,
        n_heads=2,
        temporal_layers=1,
        tabular_layers=1,
        num_classes=2,
    )
    x_temp = torch.randn(batch_size, 10)
    x_num = torch.randn(batch_size, num_num)
    x_cat = torch.tensor([[0, 2], [1, 5], [3, 0], [2, 1]])
    logits = model(x_temp, x_num, x_cat)
    assert logits.shape == (batch_size, 2)


def test_flood_transformer_predictor():
    predictor = FloodTransformerPredictor(device="cpu")
    sample_rain = [1.0, 3.0, 5.0, 10.0, 20.0, 45.0, 80.0, 150.0, 280.0, 450.0]
    result = predictor.predict(
        temporal_sequence=sample_rain,
        numerical_features={"Drainage Area": 500.0, "Stream Order": 4},
        categorical_features={"Land cover": "Forest", "Soil type": "Luvisols"},
    )
    assert "predicted_class" in result
    assert "hazard_probability" in result
    assert 0.0 <= result["hazard_probability"] <= 1.0
    assert result["predicted_class"] in predictor.classes
