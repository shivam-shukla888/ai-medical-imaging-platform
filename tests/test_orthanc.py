"""
Tests for DICOM ingestion service and Orthanc client offline/online handling.
"""
from pathlib import Path
import pytest
import numpy as np

from app.api.orthanc_client import (
    OrthancClient,
    DICOMIngestionService,
    create_sample_dicom_file,
)


def test_dicom_metadata_extraction_and_anonymization(tmp_path):
    """Verify that DICOM ingestion extracts metadata safely without PHI leaks."""
    dcm_path = str(tmp_path / "test.dcm")
    create_sample_dicom_file(dcm_path)

    with open(dcm_path, "rb") as f:
        dcm_bytes = f.read()

    meta = DICOMIngestionService.inspect_and_anonymize(dcm_bytes)
    assert meta["modality"] == "DX"
    assert meta["body_part_examined"] == "CHEST"
    assert meta["rows"] == 28
    assert meta["columns"] == 28
    assert meta["photometric_interpretation"] == "MONOCHROME2"
    # Ensure sensitive personal fields are absent from metadata
    assert "PatientAddress" not in meta
    assert "PatientBirthDate" not in meta


def test_dicom_pixel_extraction(tmp_path):
    """Verify that DICOM pixel array is extracted as a normalized uint8 array."""
    dcm_path = str(tmp_path / "test_pixels.dcm")
    create_sample_dicom_file(dcm_path)

    with open(dcm_path, "rb") as f:
        dcm_bytes = f.read()

    arr = DICOMIngestionService.extract_image_array(dcm_bytes)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (28, 28)
    assert arr.dtype == np.uint8
    assert 0 <= arr.min() <= 255
    assert 0 <= arr.max() <= 255


def test_orthanc_client_offline_graceful_handling():
    """Verify that OrthancClient gracefully handles an unreachable server without raising exceptions."""
    # Use non-existent port to simulate offline PACS
    client = OrthancClient(base_url="http://localhost:59999", timeout=0.5)

    assert client.is_online() is False
    info = client.get_system_info()
    assert info["online"] is False
    assert "unreachable" in info["message"].lower() or "offline" in info["message"].lower()

    assert client.list_patients() == []
    assert client.list_studies() == []
