#!/usr/bin/env python3
"""
CLI script to verify Orthanc PACS server connectivity and test DICOM ingestion.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.orthanc_client import OrthancClient, create_sample_dicom_file


def main():
    print("=" * 60)
    print("  ORTHANC PACS CONNECTIVITY & DICOM INGESTION TEST")
    print("=" * 60)

    client = OrthancClient()
    print(f"Target Orthanc Server URL: {client.base_url}")

    if client.is_online():
        print("Status: ONLINE (Connected successfully!)")
        sys_info = client.get_system_info()
        print(f"  Orthanc Version: {sys_info.get('Version')}")
        print(f"  Storage Directory: {sys_info.get('StorageDirectory')}")
        print(f"  Active Patients Count: {len(client.list_patients())}")
        print(f"  Active Studies Count:  {len(client.list_studies())}")

        # Test uploading sample DICOM
        sample_path = PROJECT_ROOT / "data" / "sample_anonymized_chest.dcm"
        if not sample_path.exists():
            create_sample_dicom_file(str(sample_path))

        with open(sample_path, "rb") as f:
            upload_res = client.upload_dicom(f.read())
        print(f"\nUploaded test DICOM study to Orthanc:")
        print(f"  Instance ID: {upload_res.get('ID')}")
        print(f"  Status: {upload_res.get('Status')}")
        return 0
    else:
        print("Status: OFFLINE (Orthanc PACS service is currently stopped)")
        print("\nNote: The ML inference pipeline functions completely independently of Orthanc.")
        print("To start Orthanc via Docker Compose:")
        print("  docker compose --profile pacs up -d orthanc")
        print("  or: docker compose up -d (when orthanc is configured in active services)")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
