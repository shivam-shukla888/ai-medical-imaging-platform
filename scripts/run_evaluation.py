#!/usr/bin/env python3
"""
CLI script to evaluate the saved checkpoint against the official PneumoniaMNIST test split.
Generates metrics.json, evaluation_report.md, confusion matrix, and ROC curve.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.evaluate import evaluate_model
from app.ml.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on official test split")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("  AI CHEST X-RAY RESEARCH PLATFORM - MODEL EVALUATION")
    print("=" * 60)

    try:
        evaluate_model(config=config)
        return 0
    except Exception as e:
        print(f"\nEVALUATION FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
