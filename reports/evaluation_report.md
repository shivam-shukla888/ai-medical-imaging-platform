# AI Chest X-Ray Research Platform — Test Evaluation Report

> **Important Educational Disclaimer:**
> Educational research demo only. Not intended for clinical use, diagnosis, treatment, or medical decision-making.
> Benchmark performance on 28x28 preprocessed images does NOT reflect clinical diagnostic reliability or safety on real-world patients.

---

## 1. Executive Summary

Evaluation was conducted on the **untouched official test split** of the MedMNIST v2 PneumoniaMNIST benchmark.

| Metric | Score | Clinical / Statistical Definition |
| :--- | :--- | :--- |
| **Accuracy** | **0.8974** (89.74%) | Overall correct classification fraction |
| **Sensitivity (Recall)** | **0.9821** (98.21%) | True Positive Rate (Pneumonia identified correctly) |
| **Specificity** | **0.7564** (75.64%) | True Negative Rate (Normal identified correctly) |
| **Precision** | **0.8705** (87.05%) | Positive Predictive Value for Pneumonia |
| **F1 Score** | **0.9229** | Harmonic mean of Precision and Sensitivity |
| **ROC-AUC** | **0.9418** | Area Under Receiver Operating Characteristic Curve |

---

## 2. Confusion Matrix

| Reference \ Predicted | Normal (Predicted) | Pneumonia (Predicted) | Total Reference |
| :--- | :---: | :---: | :---: |
| **Normal (True 0)** | **177** (TN) | **57** (FP) | 234 |
| **Pneumonia (True 1)** | **7** (FN) | **383** (TP) | 390 |
| **Total Predicted** | 184 | 440 | **624** |

- **True Negatives (TN)**: 177
- **False Positives (FP)**: 57
- **False Negatives (FN)**: 7
- **True Positives (TP)**: 383

---

## 3. Per-Class Diagnostic Breakdown

| Class | Precision | Recall (Sensitivity) | F1-Score | Support (N) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal (0)** | 0.9620 | 0.7564 | 0.8469 | 234 |
| **Pneumonia (1)** | 0.8705 | 0.9821 | 0.9229 | 390 |
| **Macro Average** | 0.9162 | 0.8692 | 0.8849 | 624 |
| **Weighted Average** | 0.9048 | 0.8974 | 0.8944 | 624 |

---

## 4. Evaluation Provenance & Environment

- **Model Architecture**: `ChestXRayMiniCNN`
- **Checkpoint Epoch**: 4
- **Checkpoint Timestamp**: `2026-10-26 15:43:44 UTC`
- **Evaluation Date**: `2026-10-26 15:44:32 UTC`
- **Device Used**: `cpu`
- **Random Seed**: `42`
- **Dataset File**: `pneumoniamnist.npz`
- **Dataset MD5**: `28209eda62fecd6e6a2d98b1501bb15f`
- **PyTorch Version**: `2.10.0+cpu`
- **Python Version**: `3.14.3`
- **OS Platform**: `Windows 11`

---

## 5. Methodological Notes & Limitations

1. **Strict Held-Out Separation**: The test split (624 samples) was held untouched during training and checkpoint selection. No hyperparameters were adjusted based on these test figures.
2. **Resolution Limitations**: Downsampling chest radiographs from native high resolution (e.g. 2048x2048) to 28x28 discards fine pulmonary parenchymal details, vascular branching, and subtle interstitial opacities.
3. **Distribution Shift**: The model was evaluated solely on pediatric chest X-rays from the Guangzhou Women and Children's Medical Center cohort (Kermany et al., 2018). It does not represent adult populations, multi-morbid conditions, or different X-ray machine manufacturers.
