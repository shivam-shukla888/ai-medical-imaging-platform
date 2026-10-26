"""
Tests for Grad-CAM activation heatmap generation, overlay blending, and hook cleanup.
"""
import pytest
import numpy as np
import torch

from app.ml.model import ChestXRayMiniCNN
from app.ml.gradcam import GradCAM


def test_gradcam_heatmap_dimensions_and_values():
    """Verify that GradCAM produces a valid 2D normalized heatmap."""
    model = ChestXRayMiniCNN()
    target_layer = model.get_target_layer_for_gradcam()

    gradcam = GradCAM(model, target_layer)
    dummy_input = torch.randn(1, 1, 28, 28)

    try:
        heatmap = gradcam.generate_heatmap(dummy_input, target_class=1)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (28, 28)
        assert np.all(np.isfinite(heatmap))
        assert 0.0 <= heatmap.min()
        assert heatmap.max() <= 1.0
    finally:
        gradcam.remove_hooks()


def test_gradcam_overlay_synthesis():
    """Verify overlay blending produces valid RGB uint8 image."""
    raw_img = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
    heatmap = np.random.uniform(0.0, 1.0, size=(28, 28)).astype(np.float32)

    overlay = GradCAM.overlay_heatmap(raw_img, heatmap, alpha=0.5)

    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (28, 28, 3)
    assert overlay.dtype == np.uint8
    assert overlay.min() >= 0
    assert overlay.max() <= 255


def test_gradcam_hook_cleanup():
    """Verify that hooks are cleanly removed and don't leak."""
    model = ChestXRayMiniCNN()
    target_layer = model.get_target_layer_for_gradcam()

    gradcam = GradCAM(model, target_layer)
    assert len(gradcam._hooks) == 2

    gradcam.remove_hooks()
    assert len(gradcam._hooks) == 0
