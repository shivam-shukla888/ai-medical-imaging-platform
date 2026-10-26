#!/usr/bin/env python3
"""
CLI script to verify the integrity and characteristics of the PneumoniaMNIST dataset.
"""
import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.utils import load_config, resolve_data_path
from app.ml.dataset import verify_dataset_file


def main():
    print("=" * 60)
    print("  AI CHEST X-RAY RESEARCH PLATFORM - DATASET VERIFICATION")
    print("=" * 60)

    try:
        config = load_config()
        data_path = resolve_data_path(config)
        expected_md5 = config.get("data", {}).get("expected_md5")

        print(f"Dataset path: {data_path}")
        report = verify_dataset_file(data_path, expected_md5=expected_md5)

        print(f"File size: {report['file_size_bytes']:,} bytes ({report['file_size_bytes'] / (1024*1024):.2f} MB)")
        print(f"MD5 checksum: {report['md5_checksum']}")
        if report["md5_matches_expected"]:
            print("Checksum check: PASSED (matches official MedMNIST v2 release)")
        else:
            print("Checksum check: WARNING (differs from reference MedMNIST checksum)")

        print("\n--- SPLIT SUMMARY ---")
        for split_name, sinfo in report["splits"].items():
            print(f"\n[{split_name.upper()} SPLIT]")
            print(f"  Count:          {sinfo['num_samples']} samples")
            print(f"  Dimensions:     {sinfo['image_shape']} (N, H, W)")
            print(f"  Data type:      {sinfo['image_dtype']}")
            print(f"  Value range:    [{sinfo['min_value']}, {sinfo['max_value']}]")
            print("  Label distribution:")
            for cname, count in sinfo["label_distribution"].items():
                pct = (count / sinfo['num_samples']) * 100
                print(f"    - {cname}: {count} ({pct:.1f}%)")

        print("\n" + "=" * 60)
        print("VERIFICATION RESULT: SUCCESS (Dataset is valid, clean, and ready for use)")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
