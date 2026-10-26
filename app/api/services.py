"""
Service layer connecting FastAPI endpoints to ML Inference and Dataset services.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from app.ml.utils import load_config, resolve_data_path
from app.ml.dataset import load_raw_data, CLASS_NAMES
from app.ml.predict import (
    InferenceEngine,
    preprocess_image_bytes,
    preprocess_raw_array,
    image_to_base64_png,
)


class PlatformService:
    """
    Central service singleton managing inference and dataset access.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.engine = InferenceEngine(config=self.config)
        self.cached_samples: Optional[List[Dict[str, Any]]] = None
        self._init_samples()

    def _init_samples(self, max_samples: int = 50):
        """Loads a subset of test samples for instant preview and evaluation in UI."""
        try:
            data_path = resolve_data_path(self.config)
            raw = load_raw_data(data_path)
            test_imgs = raw["test_images"][:max_samples]
            test_lbls = raw["test_labels"][:max_samples].squeeze()

            samples = []
            for idx in range(len(test_imgs)):
                img = test_imgs[idx]
                lbl = int(test_lbls[idx])
                samples.append({
                    "index": idx,
                    "split": "test",
                    "true_label_id": lbl,
                    "true_label_name": CLASS_NAMES.get(lbl, f"Class {lbl}"),
                    "raw_image": img,
                    "preview_base64": image_to_base64_png(img),
                })
            self.cached_samples = samples
        except Exception as e:
            print(f"[PlatformService] Sample initialization notice: {e}")
            self.cached_samples = []

    def get_model_info(self) -> Dict[str, Any]:
        """Returns model architecture, parameters, and metadata."""
        meta = self.engine.model_metadata
        return {
            "name": self.config.get("project_name", "AI Chest X-Ray Research Platform"),
            "architecture": meta.get("architecture", "ChestXRayMiniCNN"),
            "num_classes": self.config.get("model", {}).get("num_classes", 2),
            "classes": self.engine.class_mapping,
            "checkpoint_epoch": meta.get("checkpoint_epoch"),
            "val_loss": meta.get("val_loss"),
            "val_accuracy": meta.get("val_accuracy"),
            "device": str(self.engine.device),
            "disclaimer": self.config.get("disclaimer"),
        }

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        generate_explanation: bool = True,
    ) -> Dict[str, Any]:
        """Runs prediction and Grad-CAM on uploaded raw image bytes."""
        raw_img, tensor = preprocess_image_bytes(image_bytes)
        return self.engine.predict(raw_img, tensor, generate_explanation=generate_explanation)

    def predict_sample_by_index(
        self,
        sample_idx: int,
        generate_explanation: bool = True,
    ) -> Dict[str, Any]:
        """Runs prediction and Grad-CAM on a test split sample by index."""
        if not self.cached_samples or sample_idx < 0 or sample_idx >= len(self.cached_samples):
            raise IndexError(f"Sample index {sample_idx} is out of range (0-{len(self.cached_samples) - 1 if self.cached_samples else 0}).")

        sample = self.cached_samples[sample_idx]
        raw_img, tensor = preprocess_raw_array(sample["raw_image"])
        result = self.engine.predict(raw_img, tensor, generate_explanation=generate_explanation)
        result["sample_info"] = {
            "index": sample_idx,
            "split": sample["split"],
            "true_label_id": sample["true_label_id"],
            "true_label_name": sample["true_label_name"],
        }
        return result

    def list_samples(self) -> List[Dict[str, Any]]:
        """Returns metadata and preview base64 for test dataset samples."""
        if not self.cached_samples:
            return []
        return [
            {
                "index": s["index"],
                "split": s["split"],
                "true_label_id": s["true_label_id"],
                "true_label_name": s["true_label_name"],
                "preview_base64": s["preview_base64"],
            }
            for s in self.cached_samples
        ]


_service_instance: Optional[PlatformService] = None


def get_service() -> PlatformService:
    """Returns singleton instance of PlatformService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PlatformService()
    return _service_instance
