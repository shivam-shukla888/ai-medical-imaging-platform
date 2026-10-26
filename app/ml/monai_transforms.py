"""
MONAI (Medical Open Network for AI) preprocessing transforms and pipelines.
Provides deterministic medical-imaging transforms for 2D chest radiographs.
Reference: MONAI Consortium (https://monai.io/)
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from PIL import Image

try:
    import monai
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        EnsureType,
        Resize,
        ScaleIntensityRange,
        CastToType,
        RandGaussianNoise,
        RandRotate,
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


class MonaiChestXRayPreprocessor:
    """
    Deterministic MONAI-based preprocessing pipeline for 2D chest radiographs.
    Features:
    1. EnsureChannelFirst: guarantees (1, H, W) channel-first layout.
    2. Resize: resizes with bilinear interpolation to target spatial dimensions (e.g. 28x28).
    3. ScaleIntensityRange: min-max normalization to standard [-1.0, 1.0] range.
    4. CastToType & EnsureType: produces strict torch.float32 tensors.
    """

    def __init__(self, target_size: Tuple[int, int] = (28, 28)):
        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is not installed. Please install via: pip install monai"
            )

        self.target_size = target_size
        self.transforms = Compose([
            # Input: (H, W) or (H, W, 1) -> (1, H, W)
            EnsureChannelFirst(channel_dim="no_channel"),
            # Resize spatially to (28, 28)
            Resize(spatial_size=target_size, mode="bilinear", anti_aliasing=True),
            # Normalize [0, 255] -> [-1.0, 1.0]
            ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
            # Cast and ensure PyTorch Tensor
            CastToType(dtype=np.float32),
            EnsureType(data_type="tensor", dtype=torch.float32),
        ])

        # Optional augmentation pipeline for training exploration
        self.train_augmentation = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(spatial_size=target_size, mode="bilinear", anti_aliasing=True),
            RandRotate(range_x=0.1, prob=0.5, mode="bilinear"),
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
            ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
            CastToType(dtype=np.float32),
            EnsureType(data_type="tensor", dtype=torch.float32),
        ])

    def preprocess_array(self, image_array: np.ndarray, augment: bool = False) -> torch.Tensor:
        """
        Preprocesses a 2D numpy array (H, W) with uint8 [0, 255] values.
        Returns: 4D PyTorch tensor (1, 1, target_H, target_W) on CPU.
        """
        if image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = image_array.squeeze(2)
        elif image_array.ndim == 3 and image_array.shape[2] == 3:
            # Grayscale luminosity conversion
            image_array = (
                0.2989 * image_array[..., 0] +
                0.5870 * image_array[..., 1] +
                0.1140 * image_array[..., 2]
            ).astype(np.float32)

        pipeline = self.train_augmentation if augment else self.transforms
        tensor = pipeline(image_array)

        # Ensure batch dimension: (1, 1, H, W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Validate tensor output
        if tensor.shape[1:] != (1, *self.target_size):
            raise ValueError(f"MONAI output shape mismatch: got {tensor.shape}, expected (1, 1, {self.target_size})")

        return tensor

    def validate_tensor(self, tensor: torch.Tensor) -> bool:
        """Validates shape, data type, and numerical range."""
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            return False
        if tensor.dtype != torch.float32:
            return False
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
        if tensor.min() < -1.05 or tensor.max() > 1.05:
            return False
        return True
