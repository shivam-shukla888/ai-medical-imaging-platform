"""
Dataset loading, verification, and PyTorch Dataset/DataLoader implementations.
"""
from pathlib import Path
import hashlib
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


EXPECTED_ARRAYS = [
    "train_images",
    "val_images",
    "test_images",
    "train_labels",
    "val_labels",
    "test_labels",
]

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


def verify_dataset_file(
    file_path: Path | str,
    expected_md5: Optional[str] = "28209eda62fecd6e6a2d98b1501bb15f",
) -> Dict:
    """
    Performs comprehensive verification on the local dataset archive:
    - File existence & size
    - MD5 checksum validation against official MedMNIST metadata
    - Array completeness and naming
    - Shape, data type, and value ranges
    - Split counts and label distributions
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset archive not found at: {path}")

    # Check MD5
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 16):
            hasher.update(chunk)
    actual_md5 = hasher.hexdigest()
    md5_match = (expected_md5.lower() == actual_md5.lower()) if expected_md5 else None

    # Load arrays safely
    with np.load(path) as data:
        present_keys = list(data.keys())
        missing_keys = [k for k in EXPECTED_ARRAYS if k not in present_keys]
        if missing_keys:
            raise ValueError(f"Dataset corrupted or missing expected keys: {missing_keys}")

        report = {
            "file_path": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "md5_checksum": actual_md5,
            "md5_matches_expected": md5_match,
            "splits": {},
        }

        for split in ["train", "val", "test"]:
            imgs = data[f"{split}_images"]
            lbls = data[f"{split}_labels"]

            if imgs.ndim != 3 or imgs.shape[1:] != (28, 28):
                raise ValueError(
                    f"Unexpected image dimensions for {split}: {imgs.shape}. Expected (N, 28, 28)."
                )

            if len(imgs) != len(lbls):
                raise ValueError(
                    f"Count mismatch in {split}: {len(imgs)} images vs {len(lbls)} labels."
                )

            unique_labels, counts = np.unique(lbls, return_counts=True)
            dist = {
                CLASS_NAMES.get(int(lbl), f"Class {lbl}"): int(cnt)
                for lbl, cnt in zip(unique_labels, counts)
            }

            report["splits"][split] = {
                "num_samples": int(len(imgs)),
                "image_shape": list(imgs.shape),
                "image_dtype": str(imgs.dtype),
                "min_value": float(imgs.min()),
                "max_value": float(imgs.max()),
                "label_distribution": dist,
            }

    return report


class PneumoniaDataset(Dataset):
    """
    PyTorch Dataset wrapper for PneumoniaMNIST grayscale 28x28 chest X-rays.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ):
        self.images = images  # shape: (N, 28, 28), uint8 [0, 255]
        self.labels = labels.squeeze()  # shape: (N,), uint8/int64
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]  # (28, 28)
        lbl = int(self.labels[idx])

        # Normalize to float32 tensor [0, 1] then standardize to [-1, 1]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, 28, 28)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        else:
            # Default standard normalization: (x - 0.5) / 0.5 -> [-1, 1]
            img_tensor = (img_tensor - 0.5) / 0.5

        return img_tensor, lbl

    def get_raw_image(self, idx: int) -> np.ndarray:
        """Returns the raw unnormalized 28x28 uint8 image."""
        return self.images[idx]


def load_raw_data(data_path: Path | str) -> Dict[str, np.ndarray]:
    """Loads all arrays from the .npz archive into memory as numpy arrays."""
    with np.load(data_path) as data:
        return {k: np.array(data[k]) for k in EXPECTED_ARRAYS}


def get_dataloaders(
    config: dict,
    data_path: Optional[Path | str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates train, validation, and test PyTorch DataLoaders using official splits.
    Returns: (train_loader, val_loader, test_loader, split_counts)
    """
    from .utils import resolve_data_path

    if data_path is None:
        data_path = resolve_data_path(config)

    raw_data = load_raw_data(data_path)

    batch_size = config.get("training", {}).get("batch_size", 64)

    train_ds = PneumoniaDataset(
        raw_data["train_images"],
        raw_data["train_labels"],
    )
    val_ds = PneumoniaDataset(
        raw_data["val_images"],
        raw_data["val_labels"],
    )
    test_ds = PneumoniaDataset(
        raw_data["test_images"],
        raw_data["test_labels"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    split_counts = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }

    return train_loader, val_loader, test_loader, split_counts
