"""
Educational 3D Volumetric Medical Imaging Demo using MONAI 3D U-Net.
Demonstrates 3D volumetric segmentation on a synthetic volumetric phantom without downloading external datasets.
"""
from typing import Dict, Any, Tuple
import time
import numpy as np
import torch
import torch.nn as nn

try:
    import monai
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


def generate_synthetic_3d_volume(
    shape: Tuple[int, int, int] = (32, 32, 32),
    radius: float = 6.0,
    noise_level: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes a 3D volume (e.g., simulated CT phantom) with a spherical nodule lesion:
    - Volume: 3D numpy array with voxel intensity [0.0, 1.0] and Gaussian noise
    - Ground Truth Mask: binary 3D sphere mask (1 = nodule, 0 = background)
    """
    d, h, w = shape
    center = (d // 2, h // 2, w // 2)

    # Coordinate grid
    z, y, x = np.ogrid[:d, :h, :w]
    dist_sq = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2

    # Sphere mask
    mask = (dist_sq <= radius ** 2).astype(np.float32)

    # Simulated HU intensity: tissue baseline 0.3, lesion 0.85 + noise
    volume = np.full(shape, 0.3, dtype=np.float32)
    volume[mask > 0] = 0.85
    noise = np.random.normal(0, noise_level, size=shape).astype(np.float32)
    volume = np.clip(volume + noise, 0.0, 1.0)

    return volume, mask


def run_3d_unet_demonstration() -> Dict[str, Any]:
    """
    Executes a complete 3D volumetric segmentation demo:
    1. Synthesizes 3D volume (1, 1, 32, 32, 32) and ground truth mask
    2. Builds a lightweight MONAI 3D U-Net
    3. Runs forward pass and calculates Dice Loss
    4. Evaluates segmentation overlap using MONAI DiceMetric
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for 3D demonstration.")

    start_time = time.time()
    volume_np, mask_np = generate_synthetic_3d_volume(shape=(32, 32, 32), radius=6.0)

    # Convert to PyTorch tensors with shape (B=1, C=1, D=32, H=32, W=32)
    volume_tensor = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    # Instantiate lightweight MONAI 3D U-Net
    model_3d = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32),
        strides=(2, 2),
        num_res_units=1,
    )
    model_3d.eval()

    # Forward pass
    with torch.no_grad():
        raw_output = model_3d(volume_tensor)
        predicted_probs = torch.sigmoid(raw_output)
        predicted_binary = (predicted_probs > 0.5).float()

    # Calculate Dice loss and metric
    dice_loss_fn = DiceLoss(sigmoid=True)
    loss = float(dice_loss_fn(raw_output, mask_tensor).item())

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric(y_pred=predicted_binary, y=mask_tensor)
    dice_score = float(dice_metric.aggregate().item())

    elapsed = time.time() - start_time

    results = {
        "status": "success",
        "spatial_dimensions": 3,
        "input_shape": list(volume_tensor.shape),
        "mask_shape": list(mask_tensor.shape),
        "model_architecture": "MONAI 3D U-Net (3-stage, spatial_dims=3)",
        "num_trainable_parameters": sum(p.numel() for p in model_3d.parameters() if p.requires_grad),
        "initial_dice_loss": round(loss, 4),
        "initial_dice_score": round(dice_score, 4),
        "execution_time_seconds": round(elapsed, 3),
        "disclaimer": "Educational 3D demonstration on synthetic volumetric data. Not for clinical diagnosis.",
    }
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  MONAI 3D U-NET VOLUMETRIC SEGMENTATION DEMO")
    print("=" * 60)
    res = run_3d_unet_demonstration()
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("=" * 60)
