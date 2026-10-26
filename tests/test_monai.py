"""
Unit tests for MONAI preprocessing pipeline and 3D volumetric demonstration.
"""
import pytest
import numpy as np
import torch

from app.ml.monai_transforms import MonaiChestXRayPreprocessor
from app.ml.monai_3d_demo import generate_synthetic_3d_volume, run_3d_unet_demonstration


def test_monai_preprocessor_shape_and_range():
    """Verify that MONAI pipeline converts raw 2D array to (1, 1, 28, 28) normalized tensor."""
    preprocessor = MonaiChestXRayPreprocessor(target_size=(28, 28))
    raw_img = np.random.randint(0, 256, size=(40, 50), dtype=np.uint8)

    tensor = preprocessor.preprocess_array(raw_img, augment=False)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32
    assert -1.05 <= tensor.min() <= 1.05
    assert -1.05 <= tensor.max() <= 1.05
    assert preprocessor.validate_tensor(tensor) is True


def test_monai_preprocessor_rgb_handling():
    """Verify that MONAI pipeline safely handles RGB 3-channel input."""
    preprocessor = MonaiChestXRayPreprocessor(target_size=(28, 28))
    rgb_img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

    tensor = preprocessor.preprocess_array(rgb_img, augment=False)
    assert tensor.shape == (1, 1, 28, 28)
    assert preprocessor.validate_tensor(tensor) is True


def test_monai_3d_unet_demo():
    """Verify that MONAI 3D U-Net demonstration runs and outputs valid shapes and scores."""
    result = run_3d_unet_demonstration()

    assert result["status"] == "success"
    assert result["spatial_dimensions"] == 3
    assert result["input_shape"] == [1, 1, 32, 32, 32]
    assert result["mask_shape"] == [1, 1, 32, 32, 32]
    assert result["initial_dice_loss"] > 0.0
    assert result["execution_time_seconds"] > 0.0
