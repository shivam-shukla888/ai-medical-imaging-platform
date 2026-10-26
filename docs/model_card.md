# Model Card — ChestXRayMiniCNN

## 1. Model Details

- **Model Name**: ChestXRayMiniCNN
- **Model Version**: 1.0.0
- **Model Date**: October 2026
- **Architecture**: 3-Stage Convolutional Neural Network with Batch Normalization, Dropout, and Adaptive Average Pooling
- **Framework**: PyTorch 2.x
- **License**: MIT License
- **Target Task**: Binary classification of pediatric chest radiographs (0 = Normal, 1 = Pneumonia)

---

## 2. Intended Use & Excluded Use

### Intended Use
- **Educational Demonstrations**: Demonstrating deep learning workflows in medical imaging.
- **Explainability Research**: Inspecting gradient-based class activation mapping (Grad-CAM) behavior on compact anatomical features.
- **Pedagogical Prototyping**: Fast turnaround prototyping on consumer laptops without cloud infrastructure.

### Excluded & Prohibited Uses
- **Clinical Diagnosis**: This model is strictly NOT approved by the FDA, EMA, or any national health authority.
- **Autonomous Triage**: It must not be deployed in hospital emergency departments or clinics.
- **Treatment Decisions**: Do not base antibiotic prescription, hospitalization, or diagnostic decisions on model outputs.

---

## 3. Training Data & Procedure

- **Dataset**: MedMNIST v2 PneumoniaMNIST
- **Training Samples**: 4,708 images (1,214 Normal [25.8%], 3,494 Pneumonia [74.2%])
- **Validation Samples**: 524 images (135 Normal [25.8%], 389 Pneumonia [74.2%])
- **Resolution**: 28 x 28 pixels, 8-bit grayscale
- **Loss Function**: Weighted Cross-Entropy Loss to counter class imbalance (Class 0: 1.939, Class 1: 0.674)
- **Optimizer**: AdamW (Learning Rate: 0.001, Weight Decay: 0.0001)
- **Batch Size**: 64
- **Epochs**: 8 (Best checkpoint selected via lowest validation loss at Epoch 4)
- **Hardware**: CPU / NVIDIA RTX 3050 Laptop GPU

---

## 4. Evaluation Performance on Official Test Split

Evaluation was conducted on the **624 held-out test images** (234 Normal, 390 Pneumonia).

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **89.74%** | Correct classifications across all test cases |
| **Sensitivity (Recall)** | **98.21%** | Successfully identified 383 out of 390 pneumonia cases |
| **Specificity** | **75.64%** | Successfully identified 177 out of 234 normal cases |
| **Precision** | **87.05%** | When predicting Pneumonia, correct 87.05% of the time |
| **F1 Score** | **0.9229** | High harmonic mean between precision and recall |
| **ROC-AUC** | **0.9418** | Strong discriminative ability between normal and pneumonia |

---

## 5. Explainability (Grad-CAM)

- **Target Layer**: `features.conv3` (final convolutional layer with 128 feature channels).
- **Mechanism**: Calculates importance weights via global average pooling of gradients:
  $$\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y_c}{\partial A_{i,j}^k}$$
- **Visualization**: Rectified linear combination yields an activation map identifying pulmonary infiltrates, consolidation, or pleural opacities.
- **Caution**: Heatmaps show correlation with model feature activation, NOT definitive microbiological pathogen localization.
