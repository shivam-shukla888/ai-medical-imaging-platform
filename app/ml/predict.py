"""
Shared inference engine for single-image prediction and Grad-CAM explanation.
Provides input validation, image preprocessing, timing, and base64 rendering.
"""
from pathlib import Path
import io
import time
import base64
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .utils import find_project_root, load_config, get_device
from .model import build_model
from .gradcam import GradCAM


def image_to_base64_png(image_array: np.ndarray) -> str:
    """Encodes a uint8 2D or 3D numpy image into a base64 PNG data URL."""
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 2:
        pil_img = Image.fromarray(image_array, mode="L")
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        pil_img = Image.fromarray(image_array.squeeze(2), mode="L")
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        pil_img = Image.fromarray(image_array, mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape for base64 encoding: {image_array.shape}")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def preprocess_image_bytes(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (28, 28),
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Validates uploaded image bytes, converts to grayscale 28x28 uint8 array,
    and constructs normalized PyTorch tensor (1, 1, 28, 28).
    """
    if len(image_bytes) == 0:
        raise ValueError("Image file is empty (0 bytes).")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
    except Exception as e:
        raise ValueError(f"Uploaded file is not a valid, decodable image: {e}")

    # Re-open after verify() invalidates the handle
    pil_img = Image.open(io.BytesIO(image_bytes))

    # Convert to single-channel 8-bit grayscale
    pil_img = pil_img.convert("L")

    # Resize to model resolution (28x28) using Bilinear interpolation
    if pil_img.size != target_size:
        pil_img = pil_img.resize(target_size, resample=Image.Resampling.BILINEAR)

    raw_28x28 = np.array(pil_img, dtype=np.uint8)

    # Convert to normalized tensor: (x / 255.0 - 0.5) / 0.5 -> [-1.0, 1.0]
    tensor = torch.from_numpy(raw_28x28).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5

    return raw_28x28, tensor


def preprocess_raw_array(
    raw_array: np.ndarray,
    target_size: Tuple[int, int] = (28, 28),
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Preprocesses an existing uint8 numpy array (e.g. from the test split).
    """
    if raw_array.ndim == 3 and raw_array.shape[2] == 1:
        raw_array = raw_array.squeeze(2)
    elif raw_array.ndim == 3 and raw_array.shape[2] == 3:
        # Convert RGB to grayscale
        raw_array = np.dot(raw_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    if raw_array.shape != target_size:
        pil_img = Image.fromarray(raw_array.astype(np.uint8), mode="L")
        pil_img = pil_img.resize(target_size, resample=Image.Resampling.BILINEAR)
        raw_array = np.array(pil_img, dtype=np.uint8)

    tensor = torch.from_numpy(raw_array).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return raw_array, tensor


class InferenceEngine:
    """
    Singleton-capable inference engine managing model loading, execution,
    scoring, and Grad-CAM visual explanation generation.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        checkpoint_path: Optional[Path | str] = None,
        device_preference: str = "auto",
    ):
        self.root = find_project_root()
        self.config = config or load_config()
        self.device = get_device(self.config.get("training", {}).get("device", device_preference))

        paths = self.config.get("paths", {})
        if checkpoint_path is None:
            self.checkpoint_path = self.root / paths.get("model_checkpoint", "artifacts/best_model.pt")
        else:
            self.checkpoint_path = Path(checkpoint_path)

        self.model: Optional[torch.nn.Module] = None
        self.gradcam: Optional[GradCAM] = None
        self.class_mapping = {
            int(k): v for k, v in self.config.get("data", {}).get("classes", {0: "Normal", 1: "Pneumonia"}).items()
        }
        self.model_metadata = {}
        self._load_model()

    def _load_model(self):
        """Loads model weights into memory and configures Grad-CAM hooks."""
        if not self.checkpoint_path.is_file():
            print(f"[InferenceEngine] Checkpoint not found at {self.checkpoint_path}. Engine initialized in unloaded state.")
            self.model = None
            return

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Initialize Grad-CAM on conv3
        target_layer = self.model.get_target_layer_for_gradcam()
        self.gradcam = GradCAM(self.model, target_layer)

        self.model_metadata = {
            "architecture": checkpoint.get("model_architecture", "ChestXRayMiniCNN"),
            "checkpoint_epoch": checkpoint.get("epoch", -1),
            "val_loss": checkpoint.get("val_loss", None),
            "val_accuracy": checkpoint.get("val_accuracy", None),
            "checkpoint_timestamp": checkpoint.get("timestamp", "unknown"),
            "device": str(self.device),
        }
        print(f"[InferenceEngine] Model successfully loaded on {self.device} from {self.checkpoint_path}")

    def is_ready(self) -> bool:
        """Returns True if the model weights are loaded and ready for inference."""
        return self.model is not None

    def predict(
        self,
        raw_image: np.ndarray,
        tensor: torch.Tensor,
        generate_explanation: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes prediction and optional Grad-CAM visual explanation.
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Train the model or provide a valid checkpoint.")

        start_time = time.perf_counter()
        tensor = tensor.to(self.device)

        # 1. Forward pass for prediction
        with torch.no_grad():
            logits = self.model(tensor)
            # Numerically stable softmax
            scores = F.softmax(logits, dim=1).cpu().numpy().squeeze()
            pred_class_id = int(np.argmax(scores))

        pred_class_name = self.class_mapping.get(pred_class_id, f"Class {pred_class_id}")
        model_scores = {
            self.class_mapping.get(i, f"Class {i}"): float(scores[i])
            for i in range(len(scores))
        }

        # 2. Grad-CAM visual explanation
        gradcam_results = None
        if generate_explanation and self.gradcam is not None:
            try:
                heatmap = self.gradcam.generate_heatmap(tensor, target_class=pred_class_id)
                overlay = GradCAM.overlay_heatmap(raw_image, heatmap, alpha=0.45)

                gradcam_results = {
                    "target_class": pred_class_name,
                    "heatmap_min": float(heatmap.min()),
                    "heatmap_max": float(heatmap.max()),
                    "heatmap_base64": image_to_base64_png((heatmap * 255.0).astype(np.uint8)),
                    "overlay_base64": image_to_base64_png(overlay),
                }
            except Exception as e:
                print(f"[InferenceEngine] Grad-CAM generation warning: {e}")
                gradcam_results = {"error": str(e)}

        inference_duration_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "prediction": {
                "class_id": pred_class_id,
                "class_name": pred_class_name,
                "model_scores": model_scores,
                "score_note": "Model scores are raw algorithmic outputs and do NOT represent clinically calibrated probabilities.",
            },
            "explanation": gradcam_results,
            "image_preview_base64": image_to_base64_png(raw_image),
            "inference_time_ms": round(inference_duration_ms, 2),
            "model_metadata": self.model_metadata,
            "disclaimer": self.config.get("disclaimer"),
        }
