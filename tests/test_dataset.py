"""
Tests for dataset integrity, verification, and PyTorch dataset loader.
"""
from pathlib import Path
import pytest
import numpy as np
import torch

from app.ml.utils import load_config, resolve_data_path
from app.ml.dataset import (
    verify_dataset_file,
    PneumoniaDataset,
    get_dataloaders,
    CLASS_NAMES,
)


def test_dataset_verification():
    """Verify that dataset archive passes full integrity and checksum validation."""
    config = load_config()
    data_path = resolve_data_path(config)
    report = verify_dataset_file(data_path, expected_md5=config["data"]["expected_md5"])

    assert report["md5_matches_expected"] is True
    assert set(report["splits"].keys()) == {"train", "val", "test"}
    assert report["splits"]["train"]["num_samples"] == 4708
    assert report["splits"]["val"]["num_samples"] == 524
    assert report["splits"]["test"]["num_samples"] == 624

    for split in ["train", "val", "test"]:
        assert report["splits"][split]["image_shape"][1:] == [28, 28]


def test_pneumonia_dataset_loader():
    """Verify that PneumoniaDataset produces correct tensor dimensions and normalized ranges."""
    fake_imgs = np.zeros((10, 28, 28), dtype=np.uint8)
    fake_imgs[0, 10:18, 10:18] = 255
    fake_lbls = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.uint8)

    ds = PneumoniaDataset(fake_imgs, fake_lbls)
    assert len(ds) == 10

    tensor, label = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype == torch.float32
    assert -1.0 <= tensor.min() <= 1.0
    assert -1.0 <= tensor.max() <= 1.0
    assert label == 0


def test_dataloaders_split_separation():
    """Verify that get_dataloaders produces isolated, non-overlapping splits."""
    config = load_config()
    train_loader, val_loader, test_loader, counts = get_dataloaders(config)

    assert counts["train"] == 4708
    assert counts["val"] == 524
    assert counts["test"] == 624

    # Sample batch shape check
    batch_imgs, batch_lbls = next(iter(train_loader))
    assert batch_imgs.shape[1:] == (1, 28, 28)
    assert batch_lbls.ndim == 1
