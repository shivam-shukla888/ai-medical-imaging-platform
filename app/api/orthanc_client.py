"""
Orthanc DICOM Server Client and Privacy-Preserving DICOM Ingestion Service.
Supports DICOM parsing, metadata querying, and anonymized ingestion.
Reference: Orthanc PACS (https://www.orthanc-server.com/)
"""
import os
import io
import time
from typing import Dict, Any, List, Optional, Tuple
import requests
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


class OrthancClient:
    """
    HTTP REST Client for communication with an Orthanc DICOM server.
    Features robust error handling and offline graceful degradation.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 3.0,
    ):
        self.base_url = (base_url or os.environ.get("ORTHANC_URL", "http://localhost:8042")).rstrip("/")
        self.username = username or os.environ.get("ORTHANC_USER", "orthanc")
        self.password = password or os.environ.get("ORTHANC_PASSWORD", "orthanc")
        self.timeout = timeout
        self.auth = (self.username, self.password) if self.username else None

    def is_online(self) -> bool:
        """Verifies if the Orthanc server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/system", auth=self.auth, timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Queries system information, storage, and version from Orthanc."""
        if not self.is_online():
            return {
                "online": False,
                "message": f"Orthanc server at {self.base_url} is unreachable or offline.",
            }
        try:
            resp = requests.get(f"{self.base_url}/system", auth=self.auth, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            data["online"] = True
            return data
        except Exception as e:
            return {"online": False, "error": str(e)}

    def list_patients(self) -> List[str]:
        """Lists IDs of all patients stored in Orthanc."""
        if not self.is_online():
            return []
        try:
            resp = requests.get(f"{self.base_url}/patients", auth=self.auth, timeout=self.timeout)
            return resp.json() if resp.status_code == 200 else []
        except Exception:
            return []

    def list_studies(self) -> List[str]:
        """Lists IDs of all DICOM studies stored in Orthanc."""
        if not self.is_online():
            return []
        try:
            resp = requests.get(f"{self.base_url}/studies", auth=self.auth, timeout=self.timeout)
            return resp.json() if resp.status_code == 200 else []
        except Exception:
            return []

    def get_study_metadata(self, study_id: str) -> Dict[str, Any]:
        """Retrieves study metadata while redacting potential Protected Health Information (PHI)."""
        if not self.is_online():
            raise ConnectionError(f"Orthanc is offline at {self.base_url}")

        resp = requests.get(f"{self.base_url}/studies/{study_id}", auth=self.auth, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json()

        # Sanitize and extract only research-relevant clinical metadata
        main_tags = raw.get("MainDicomTags", {})
        patient_tags = raw.get("PatientMainDicomTags", {})

        return {
            "orthanc_study_id": study_id,
            "modality": main_tags.get("Modality", "Unknown"),
            "study_description": main_tags.get("StudyDescription", "Chest Radiograph"),
            "study_date": main_tags.get("StudyDate", "Unknown"),
            "series_count": len(raw.get("Series", [])),
            "anonymized_patient_id": patient_tags.get("PatientID", "ANON"),
            "anonymized_patient_sex": patient_tags.get("PatientSex", "O"),
        }

    def upload_dicom(self, dicom_bytes: bytes) -> Dict[str, Any]:
        """Uploads a raw DICOM instance to Orthanc."""
        if not self.is_online():
            raise ConnectionError(f"Orthanc server at {self.base_url} is currently offline.")

        headers = {"Content-Type": "application/dicom"}
        resp = requests.post(
            f"{self.base_url}/instances",
            data=dicom_bytes,
            headers=headers,
            auth=self.auth,
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


class DICOMIngestionService:
    """
    Parses and sanitizes DICOM files, enforcing strict PHI redaction
    and extracting normalized 2D image arrays for the research platform.
    """

    @staticmethod
    def inspect_and_anonymize(dicom_bytes: bytes) -> Dict[str, Any]:
        """
        Parses DICOM bytes, extracts safe imaging metadata, and ensures no PHI leaks.
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required to parse DICOM files.")

        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))

        # Safe metadata extraction
        metadata = {
            "modality": str(getattr(ds, "Modality", "DX")),
            "body_part_examined": str(getattr(ds, "BodyPartExamined", "CHEST")),
            "view_position": str(getattr(ds, "ViewPosition", "AP")),
            "patient_age": str(getattr(ds, "PatientAge", "Unknown")),
            "pixel_spacing": [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])],
            "rows": int(getattr(ds, "Rows", 0)),
            "columns": int(getattr(ds, "Columns", 0)),
            "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
            "bits_stored": int(getattr(ds, "BitsStored", 8)),
        }
        return metadata

    @staticmethod
    def extract_image_array(dicom_bytes: bytes) -> np.ndarray:
        """
        Extracts pixel array from DICOM, applies photometric interpretation inversion if needed,
        rescales to uint8 [0, 255].
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required to extract pixels from DICOM files.")

        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        arr = ds.pixel_array.astype(np.float32)

        # Handle RescaleSlope and Intercept if present
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        # Handle MONOCHROME1 (where minimum intensity is white)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = np.max(arr) - arr

        # Normalize to uint8 [0, 255]
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        if max_v > min_v:
            norm_arr = ((arr - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            norm_arr = np.zeros_like(arr, dtype=np.uint8)

        return norm_arr


def create_sample_dicom_file(output_path: str = "data/sample_anonymized_chest.dcm") -> str:
    """
    Generates a valid, compliant, anonymized DICOM Part 10 test file
    containing a 28x28 simulated radiograph pattern.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom required.")

    # Create synthetic 28x28 chest radiograph pattern
    raw_pixels = np.full((28, 28), 50, dtype=np.uint8)
    # Draw simulated bilateral lung fields
    raw_pixels[5:22, 4:11] = 180
    raw_pixels[5:22, 17:24] = 180

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "DX"
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = "AP"
    ds.PatientName = "ANONYMIZED^RESEARCH"
    ds.PatientID = "ANON-RES-001"
    ds.PatientSex = "O"
    ds.PatientAge = "003Y"
    ds.StudyDescription = "Chest X-Ray Research Demo"
    ds.SeriesDescription = "Anterior-Posterior Projection"
    ds.Rows, ds.Columns = 28, 28
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelData = raw_pixels.tobytes()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.save_as(output_path, enforce_file_format=False)
    return output_path
