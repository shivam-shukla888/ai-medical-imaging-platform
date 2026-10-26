# AI Medical Imaging Platform — Presentation Slides (10-Minute Walkthrough)

---

## Slide 1: Title & Overview
- **Project**: AI Medical Imaging Platform
- **Sub-title**: Deep Learning Interpretability, MONAI Transforms, and PACS Integration on PneumoniaMNIST
- **Speaker**: Senior Biomedical ML Engineer / Intern
- **Mandatory Disclaimer**: *Educational Research Demo Only. Not Intended for Clinical Diagnosis or Patient Decision-Making.*

---

## Slide 2: The Clinical & Engineering Problem
- **Context**: Pneumonia is a leading infectious cause of pediatric mortality globally.
- **Challenge in ML**:
  - Medical image files are large, complex, and heterogeneous (DICOM Part 10 vs web formats).
  - Class imbalance (e.g., 74% pneumonia prevalence in pediatric cohorts).
  - "Black box" neural networks lack interpretability for clinical researchers.
- **Our Goal**: Build an end-to-end, reproducible, lightweight research platform with explainability (Grad-CAM), MONAI, Orthanc PACS, and dual modern frontends.

---

## Slide 3: System Architecture
- **Data Layer**: MedMNIST v2 PneumoniaMNIST benchmark ($28 \times 28$ grayscale), verified via MD5.
- **ML & XAI Core**: PyTorch `ChestXRayMiniCNN` + Grad-CAM feature attribution on `conv3`.
- **Medical Transforms**: MONAI deterministic pipelines for intensity scaling and channel validation.
- **PACS & DICOM**: Orthanc server integration with automated PHI redaction.
- **Service Tier**: FastAPI asynchronous backend with typed Pydantic validation & append-only audit logger.
- **User Interface**: Modern React dashboard (Vite + React 18) + Streamlit fallback.

---

## Slide 4: Dataset & Preprocessing Pipeline
- **Dataset**: 5,856 pediatric chest X-rays from Guangzhou Women and Children's Medical Center.
  - Train: 4,708 (1,214 Normal, 3,494 Pneumonia)
  - Validation: 524 (135 Normal, 389 Pneumonia)
  - Test: 624 (234 Normal, 390 Pneumonia)
- **Preprocessing**:
  - Grayscale 8-bit normalization: $[0, 255] \rightarrow [-1.0, 1.0]$.
  - MONAI `EnsureChannelFirst` + `ScaleIntensityRange` + `Resize`.
  - Strict split separation: Zero test set data leakage into training or hyperparameter tuning.

---

## Slide 5: Model Architecture & Training Strategy
- **Architecture (`ChestXRayMiniCNN`)**:
  - 3 Convolutional stages with BatchNorm2d, ReLU, MaxPool2d.
  - Dropout (0.25) and AdaptiveAvgPool2d head.
  - Parameter count: ~240,000 weights (< 1 MB serialized).
- **Mitigating Class Imbalance**:
  - Training set has ~3:1 ratio of pneumonia to normal.
  - Computed inverse-frequency class weights ($W_0 \approx 1.94, W_1 \approx 0.67$).
  - Passed directly to `CrossEntropyLoss(weight=weights)` with AdamW optimizer.

---

## Slide 6: Measured Evaluation Results (Official Held-Out Test Split)
- **Evaluated on 624 untouched test images**:
  - **Accuracy**: 89.74%
  - **Sensitivity (Recall)**: 98.21% (383 of 390 pneumonia cases detected)
  - **Specificity**: 75.64% (177 of 234 normal cases detected)
  - **Precision**: 87.05%
  - **F1-Score**: 0.9229
  - **ROC-AUC**: 0.9418
- **Clinical Implication**: High sensitivity minimizes catastrophic false-negative pneumonia misses in screening workflows.

---

## Slide 7: Explainable AI with Grad-CAM
- **Why Grad-CAM?**
  - Doctors need to know *why* the model made a prediction.
- **Mathematical Execution**:
  - Forward hook captures activation maps $A^k$ at layer `conv3`.
  - Backward pass computes gradients $\frac{\partial y_c}{\partial A^k}$.
  - Global Average Pooling generates feature map importance weights $\alpha_k^c$.
  - Rectified linear combination highlights positive activation areas, blended over anatomy.
- **Integrity**: Real mathematical execution; zero fabricated or mocked heatmaps.

---

## Slide 8: MONAI & 3D Volumetric Innovation
- **2D Transforms**: MONAI medical transforms integrated for intensity rescaling and tensor validation.
- **3D Volumetric U-Net Demonstration**:
  - Built a 3D U-Net using `monai.networks.nets.UNet(spatial_dims=3)`.
  - Tested on a synthetic 3D volumetric phantom ($32 \times 32 \times 32$ voxel grid) with lesion ground truth.
  - Demonstrates 3D volumetric segmentation and Dice Loss computation in ~2 seconds on CPU.

---

## Slide 9: PACS, DVC & Full-Stack Deployment
- **Orthanc DICOM**: Real PACS server integration with privacy-preserving metadata redaction.
- **DVC Pipeline**: `dvc.yaml` tracking stages (`verify_data` $\rightarrow$ `train` $\rightarrow$ `evaluate`) for complete reproducibility.
- **Multi-Container Stack**: Docker Compose orchestrating FastAPI, React, Streamlit, and Orthanc.
- **Quality Gates**: 29 automated tests passing in Pytest.

---

## Slide 10: Limitations, Safety & Key Takeaways
- **Limitations**:
  - $28 \times 28$ downsampling loses fine interstitial detail.
  - Model scores are algorithmic confidence levels, not clinical probabilities.
  - Evaluated on a pediatric cohort; cannot be used for adult diagnostic care.
- **Conclusion**:
  - Delivered a complete, verified, end-to-end medical imaging platform bridging PyTorch deep learning, explainability, DICOM engineering, and modern full-stack web applications.
