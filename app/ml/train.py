"""
Model training module with validation tracking, checkpointing, and metrics plotting.
"""
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .utils import find_project_root, load_config, get_device, set_seed
from .dataset import get_dataloaders
from .model import build_model
from .metrics import calculate_metrics


def compute_class_weights(train_labels: np.ndarray) -> torch.Tensor:
    """
    Computes inverse-frequency balanced class weights:
    weight[c] = total_samples / (num_classes * count[c])
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    num_classes = len(unique)
    weights = [total / (num_classes * cnt) for cnt in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs one training epoch and returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs evaluation over a dataset split and calculates loss + full metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    total = len(all_labels)
    avg_loss = total_loss / total
    y_probs = np.vstack(all_probs)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), y_probs)
    metrics["loss"] = avg_loss
    return metrics


def plot_training_history(history: Dict[str, list], output_path: Path):
    """Generates and saves clean loss and accuracy learning curve plots."""
    epochs = range(1, len(history["train_loss"]) + 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.set_title("Cross-Entropy Loss over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy / F1 curve
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c")
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Accuracy", color="#d62728")
    if "val_f1" in history:
        ax2.plot(epochs, history["val_f1"], "^--", label="Val F1", color="#9467bd")
    ax2.set_title("Classification Accuracy over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def train_model(
    config: Optional[dict] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes the training pipeline:
    1. Sets seed & device
    2. Loads DataLoaders
    3. Initializes CNN model
    4. Applies weighted CrossEntropyLoss
    5. Tracks validation performance & checkpoints the best model
    6. Saves metrics history and training plot
    """
    root = find_project_root()
    if config is None:
        config = load_config()

    train_cfg = config.get("training", {})
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    epochs = epochs_override or train_cfg.get("epochs", 10)
    batch_size = batch_size_override or train_cfg.get("batch_size", 64)
    lr = lr_override or train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    device_pref = train_cfg.get("device", "auto")
    device = get_device(device_pref)
    use_class_weights = train_cfg.get("use_class_weights", True)

    print(f"[Training] Using device: {device}")
    print(f"[Training] Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}")

    # Data loaders
    config["training"]["batch_size"] = batch_size
    train_loader, val_loader, test_loader, split_counts = get_dataloaders(config)
    print(f"[Training] Dataset loaded. Train: {split_counts['train']}, Val: {split_counts['val']}, Test: {split_counts['test']}")

    # Build model
    model = build_model(config).to(device)

    # Class weighting
    if use_class_weights:
        raw_train_labels = train_loader.dataset.labels
        weights = compute_class_weights(raw_train_labels).to(device)
        print(f"[Training] Class weights: Normal={weights[0]:.4f}, Pneumonia={weights[1]:.4f}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Output paths
    paths = config.get("paths", {})
    checkpoint_path = root / paths.get("model_checkpoint", "artifacts/best_model.pt")
    history_path = root / paths.get("training_history", "artifacts/training_history.json")
    plot_path = root / paths.get("loss_curve_plot", "reports/loss_curve.png")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = -1
    start_time = time.time()

    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1_score"]
        ep_time = time.time() - ep_start

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(val_f1))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )

        # Checkpoint selection: prioritize val loss with tie-breaking on F1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "config": config,
                    "model_architecture": "ChestXRayMiniCNN",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                },
                checkpoint_path,
            )
            print(f"  -> Best model saved at epoch {epoch} (Val Loss: {val_loss:.4f})")

    total_training_time = time.time() - start_time
    print(f"\n[Training] Completed in {total_training_time:.2f} seconds. Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")

    # Save history JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
                "total_training_time_seconds": total_training_time,
                "history": history,
            },
            f,
            indent=2,
        )

    # Save plot
    plot_training_history(history, plot_path)
    print(f"[Training] Learning curves saved to {plot_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "total_time_seconds": total_training_time,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "plot_path": str(plot_path),
    }
