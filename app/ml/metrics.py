"""
Comprehensive metric calculations for medical image classification evaluation.
Handles edge cases safely and generates structured, reproducible evaluation outputs.
"""
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Computes standard diagnostic evaluation metrics:
    - Accuracy
    - Precision (positive predictive value)
    - Recall (sensitivity / true positive rate)
    - Specificity (true negative rate)
    - F1 Score
    - ROC-AUC
    - Confusion Matrix [TN, FP, FN, TP]
    - Per-class precision, recall, f1, and support
    """
    if class_names is None:
        class_names = ["Normal", "Pneumonia"]

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Binary metrics (Class 1 = Pneumonia)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC AUC
    roc_auc = None
    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            if y_probs.ndim == 2 and y_probs.shape[1] == 2:
                prob_pos = y_probs[:, 1]
            else:
                prob_pos = y_probs.ravel()
            roc_auc = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            roc_auc = None

    # Detailed per-class metrics
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": cm.tolist(),
            "labels": class_names,
        },
        "per_class": {
            cname: {
                "precision": float(report_dict[cname]["precision"]),
                "recall": float(report_dict[cname]["recall"]),
                "f1_score": float(report_dict[cname]["f1-score"]),
                "support": int(report_dict[cname]["support"]),
            }
            for cname in class_names
        },
        "macro_avg": {
            "precision": float(report_dict["macro avg"]["precision"]),
            "recall": float(report_dict["macro avg"]["recall"]),
            "f1_score": float(report_dict["macro avg"]["f1-score"]),
        },
        "weighted_avg": {
            "precision": float(report_dict["weighted avg"]["precision"]),
            "recall": float(report_dict["weighted avg"]["recall"]),
            "f1_score": float(report_dict["weighted avg"]["f1-score"]),
        },
    }

    return metrics
