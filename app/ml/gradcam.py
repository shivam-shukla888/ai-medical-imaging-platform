"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Accurately generates visual explanations for decisions made by convolutional neural networks.
Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
"""
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    """
    Computes Grad-CAM heatmaps and visual overlays for a specified convolutional layer.
    Manages hook registration and clean lifecycle management.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple; first element contains gradients w.r.t layer output
            self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h_fwd, h_bwd])

    def remove_hooks(self):
        """Safely removes all registered hooks to avoid memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates a 2D normalized Grad-CAM heatmap for the given input tensor.
        Input: input_tensor with shape (1, 1, H, W)
        Returns: numpy array of shape (H, W) with values normalized to [0, 1]
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"Expected input_tensor with shape (1, C, H, W), got {input_tensor.shape}")

        self.model.eval()
        self.model.zero_grad()

        # Requires grad on input or model weights for backprop
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients. Ensure target layer is traversed.")

        # Compute importance weights via Global Average Pooling of gradients: shape (1, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps: shape (1, 1, H', W')
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to only emphasize features that have a positive influence on the target class
        cam = F.relu(cam)

        # Interpolate heatmap to match input tensor spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Handle degenerate cases (all zeros or constant)
        min_val = float(np.min(cam_np))
        max_val = float(np.max(cam_np))
        if max_val - min_val > 1e-7:
            cam_np = (cam_np - min_val) / (max_val - min_val)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    @staticmethod
    def overlay_heatmap(
        raw_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap_name: str = "jet",
    ) -> np.ndarray:
        """
        Overlays the 2D heatmap on the original grayscale image.
        - raw_image: (H, W) or (H, W, 1) or (H, W, 3) in uint8 [0, 255]
        - heatmap: (H, W) in float [0, 1]
        Returns: RGB image (H, W, 3) as uint8 [0, 255]
        """
        if raw_image.ndim == 2:
            base_rgb = np.stack([raw_image] * 3, axis=-1).astype(np.float32)
        elif raw_image.ndim == 3 and raw_image.shape[2] == 1:
            base_rgb = np.repeat(raw_image, 3, axis=2).astype(np.float32)
        else:
            base_rgb = raw_image.astype(np.float32)

        # Apply colormap to heatmap
        try:
            import matplotlib
            cmap = matplotlib.colormaps[colormap_name]
        except Exception:
            cmap = cm.get_cmap(colormap_name)
        colored_heatmap = cmap(heatmap)[:, :, :3] * 255.0  # RGB in [0, 255]

        # Alpha blend
        blended = (1.0 - alpha) * base_rgb + alpha * colored_heatmap
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return blended
