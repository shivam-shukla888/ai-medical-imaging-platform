"""
Tests for ChestXRayMiniCNN architecture, forward pass shapes, and metrics calculation.
"""
import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

from app.ml.model import ChestXRayMiniCNN, build_model
from app.ml.metrics import calculate_metrics
from app.ml.utils import load_config


def test_model_forward_shape():
    """Verify that MiniCNN outputs shape (B, 2) for various batch sizes."""
    model = ChestXRayMiniCNN(in_channels=1, num_classes=2)
    model.eval()

    for batch_size in [1, 4, 16]:
        dummy_input = torch.randn(batch_size, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()


def test_target_conv_layer():
    """Verify that target conv layer for Grad-CAM is accessible and is an nn.Conv2d."""
    model = ChestXRayMiniCNN()
    target_layer = model.get_target_layer_for_gradcam()
    assert isinstance(target_layer, torch.nn.Conv2d)
    assert target_layer.out_channels == 128


def test_model_save_and_load_consistency():
    """Verify state_dict serialization and load consistency."""
    model = ChestXRayMiniCNN()
    dummy_input = torch.randn(2, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        out1 = model(dummy_input)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        torch.save({"model_state_dict": model.state_dict()}, tmp_path)
        loaded_model = ChestXRayMiniCNN()
        ckpt = torch.load(tmp_path)
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_model.eval()

        with torch.no_grad():
            out2 = loaded_model(dummy_input)

        assert torch.allclose(out1, out2, atol=1e-5)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_metrics_calculation():
    """Verify metrics calculation on synthetic deterministic predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_probs = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4],
        [0.1, 0.9],
    ])

    metrics = calculate_metrics(y_true, y_pred, y_probs)
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"]["true_positives"] == 1
    assert metrics["confusion_matrix"]["true_negatives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["specificity"] == 0.5
    assert metrics["roc_auc"] is not None
