"""
Official test split evaluation module.
Generates comprehensive diagnostic metrics, confusion matrix figures, ROC curves,
machine-readable reports/metrics.json, and formatted reports/evaluation_report.md.
"""
from pathlib import Path
import json
import time
import platform
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from .utils import find_project_root, load_config, get_device, resolve_data_path
from .dataset import get_dataloaders, verify_dataset_file
from .model import build_model
from .metrics import calculate_metrics


def plot_confusion_matrix(cm_data: list, labels: list, output_path: Path):
    """Plots and saves an annotated confusion matrix heatmap."""
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_ylabel("True (Reference) Class", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_title("Test Split Confusion Matrix", fontsize=14, pad=15, fontweight="bold")

    # Annotate numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, roc_auc: float, output_path: Path):
    """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
    prob_pos = y_probs[:, 1] if y_probs.ndim == 2 else y_probs.ravel()
    fpr, tpr, _ = roc_curve(y_true, prob_pos)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#007acc", lw=2, label=f"MiniCNN (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Test Split ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_markdown_report(
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: Path,
):
    """Generates a structured, clinical-grade documentation markdown report."""
    cm = metrics["confusion_matrix"]
    cm_mat = cm["matrix"]

    report_content = f"""# AI Chest X-Ray Research Platform — Test Evaluation Report

> **Important Educational Disclaimer:**
> {metadata['disclaimer']}
> Benchmark performance on 28x28 preprocessed images does NOT reflect clinical diagnostic reliability or safety on real-world patients.

---

## 1. Executive Summary

Evaluation was conducted on the **untouched official test split** of the MedMNIST v2 PneumoniaMNIST benchmark.

| Metric | Score | Clinical / Statistical Definition |
| :--- | :--- | :--- |
| **Accuracy** | **{metrics['accuracy']:.4f}** ({metrics['accuracy']*100:.2f}%) | Overall correct classification fraction |
| **Sensitivity (Recall)** | **{metrics['recall_sensitivity']:.4f}** ({metrics['recall_sensitivity']*100:.2f}%) | True Positive Rate (Pneumonia identified correctly) |
| **Specificity** | **{metrics['specificity']:.4f}** ({metrics['specificity']*100:.2f}%) | True Negative Rate (Normal identified correctly) |
| **Precision** | **{metrics['precision']:.4f}** ({metrics['precision']*100:.2f}%) | Positive Predictive Value for Pneumonia |
| **F1 Score** | **{metrics['f1_score']:.4f}** | Harmonic mean of Precision and Sensitivity |
| **ROC-AUC** | **{metrics['roc_auc']:.4f}** | Area Under Receiver Operating Characteristic Curve |

---

## 2. Confusion Matrix

| Reference \\ Predicted | Normal (Predicted) | Pneumonia (Predicted) | Total Reference |
| :--- | :---: | :---: | :---: |
| **Normal (True 0)** | **{cm_mat[0][0]}** (TN) | **{cm_mat[0][1]}** (FP) | {cm_mat[0][0] + cm_mat[0][1]} |
| **Pneumonia (True 1)** | **{cm_mat[1][0]}** (FN) | **{cm_mat[1][1]}** (TP) | {cm_mat[1][0] + cm_mat[1][1]} |
| **Total Predicted** | {cm_mat[0][0] + cm_mat[1][0]} | {cm_mat[0][1] + cm_mat[1][1]} | **{sum(sum(r) for r in cm_mat)}** |

- **True Negatives (TN)**: {cm_mat[0][0]}
- **False Positives (FP)**: {cm_mat[0][1]}
- **False Negatives (FN)**: {cm_mat[1][0]}
- **True Positives (TP)**: {cm_mat[1][1]}

---

## 3. Per-Class Diagnostic Breakdown

| Class | Precision | Recall (Sensitivity) | F1-Score | Support (N) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal (0)** | {metrics['per_class']['Normal']['precision']:.4f} | {metrics['per_class']['Normal']['recall']:.4f} | {metrics['per_class']['Normal']['f1_score']:.4f} | {metrics['per_class']['Normal']['support']} |
| **Pneumonia (1)** | {metrics['per_class']['Pneumonia']['precision']:.4f} | {metrics['per_class']['Pneumonia']['recall']:.4f} | {metrics['per_class']['Pneumonia']['f1_score']:.4f} | {metrics['per_class']['Pneumonia']['support']} |
| **Macro Average** | {metrics['macro_avg']['precision']:.4f} | {metrics['macro_avg']['recall']:.4f} | {metrics['macro_avg']['f1_score']:.4f} | {metrics['per_class']['Normal']['support'] + metrics['per_class']['Pneumonia']['support']} |
| **Weighted Average** | {metrics['weighted_avg']['precision']:.4f} | {metrics['weighted_avg']['recall']:.4f} | {metrics['weighted_avg']['f1_score']:.4f} | {metrics['per_class']['Normal']['support'] + metrics['per_class']['Pneumonia']['support']} |

---

## 4. Evaluation Provenance & Environment

- **Model Architecture**: `{metadata['model_architecture']}`
- **Checkpoint Epoch**: {metadata['checkpoint_epoch']}
- **Checkpoint Timestamp**: `{metadata['checkpoint_timestamp']}`
- **Evaluation Date**: `{metadata['evaluation_date']}`
- **Device Used**: `{metadata['device']}`
- **Random Seed**: `{metadata['seed']}`
- **Dataset File**: `{metadata['dataset_file']}`
- **Dataset MD5**: `{metadata['dataset_md5']}`
- **PyTorch Version**: `{torch.__version__}`
- **Python Version**: `{platform.python_version()}`
- **OS Platform**: `{platform.system()} {platform.release()}`

---

## 5. Methodological Notes & Limitations

1. **Strict Held-Out Separation**: The test split ({metrics['per_class']['Normal']['support'] + metrics['per_class']['Pneumonia']['support']} samples) was held untouched during training and checkpoint selection. No hyperparameters were adjusted based on these test figures.
2. **Resolution Limitations**: Downsampling chest radiographs from native high resolution (e.g. 2048x2048) to 28x28 discards fine pulmonary parenchymal details, vascular branching, and subtle interstitial opacities.
3. **Distribution Shift**: The model was evaluated solely on pediatric chest X-rays from the Guangzhou Women and Children's Medical Center cohort (Kermany et al., 2018). It does not represent adult populations, multi-morbid conditions, or different X-ray machine manufacturers.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def evaluate_model(config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Evaluates the saved model checkpoint against the official test set.
    Generates reports/metrics.json, reports/evaluation_report.md, and plots.
    """
    root = find_project_root()
    if config is None:
        config = load_config()

    device = get_device(config.get("training", {}).get("device", "auto"))
    paths = config.get("paths", {})
    checkpoint_path = root / paths.get("model_checkpoint", "artifacts/best_model.pt")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {checkpoint_path}. Run training first."
        )

    print(f"[Evaluation] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test dataloader
    _, _, test_loader, split_counts = get_dataloaders(config)
    print(f"[Evaluation] Evaluating on untouched test split ({split_counts['test']} samples)...")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.vstack(all_probs)

    class_names = [config.get("data", {}).get("classes", {}).get(i, f"Class {i}") for i in [0, 1]]
    metrics = calculate_metrics(y_true, y_pred, y_probs, class_names=class_names)

    # Check dataset info
    data_path = resolve_data_path(config)
    data_audit = verify_dataset_file(data_path)

    metadata = {
        "disclaimer": config.get("disclaimer"),
        "model_architecture": checkpoint.get("model_architecture", "ChestXRayMiniCNN"),
        "checkpoint_epoch": checkpoint.get("epoch", -1),
        "checkpoint_timestamp": checkpoint.get("timestamp", "unknown"),
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "seed": config.get("training", {}).get("seed", 42),
        "dataset_file": str(data_path.name),
        "dataset_md5": data_audit["md5_checksum"],
    }

    # Output paths
    cm_plot_path = root / paths.get("confusion_matrix_plot", "reports/confusion_matrix.png")
    roc_plot_path = root / paths.get("roc_curve_plot", "reports/roc_curve.png")
    metrics_json_path = root / paths.get("metrics_json", "reports/metrics.json")
    report_md_path = root / paths.get("evaluation_report", "reports/evaluation_report.md")

    # Generate figures
    plot_confusion_matrix(metrics["confusion_matrix"]["matrix"], class_names, cm_plot_path)
    plot_roc_curve(y_true, y_probs, metrics["roc_auc"], roc_plot_path)

    # Save metrics JSON
    metrics_record = {
        "metadata": metadata,
        "metrics": metrics,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    # Save markdown report
    generate_markdown_report(metrics, metadata, report_md_path)

    print("\n" + "=" * 60)
    print("  TEST SPLIT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:            {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Sensitivity/Recall:  {metrics['recall_sensitivity']:.4f} ({metrics['recall_sensitivity']*100:.2f}%)")
    print(f"Specificity:         {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"Precision:           {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"F1 Score:            {metrics['f1_score']:.4f}")
    if metrics['roc_auc']:
        print(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"Metrics saved to:    {metrics_json_path}")
    print(f"Report saved to:     {report_md_path}")
    print(f"Plots saved to:      {cm_plot_path}, {roc_plot_path}")
    print("=" * 60)

    return metrics_record
