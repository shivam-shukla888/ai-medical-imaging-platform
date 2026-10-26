#!/usr/bin/env python3
"""
CLI script to train the ChestXRayMiniCNN model on PneumoniaMNIST.
Supports configurable epochs, learning rate, and batch size.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.train import train_model
from app.ml.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train ChestXRayMiniCNN on PneumoniaMNIST")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("  AI CHEST X-RAY RESEARCH PLATFORM - MODEL TRAINING")
    print("=" * 60)

    try:
        results = train_model(
            config=config,
            epochs_override=args.epochs,
            batch_size_override=args.batch_size,
            lr_override=args.lr,
        )
        print("\n" + "=" * 60)
        print("TRAINING SUCCEEDED!")
        print(f"Best Epoch:    {results['best_epoch']}")
        print(f"Best Val Loss: {results['best_val_loss']:.4f}")
        print(f"Best Val F1:   {results['best_val_f1']:.4f}")
        print(f"Checkpoint:    {results['checkpoint_path']}")
        print(f"Plots saved:   {results['plot_path']}")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\nTRAINING FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
