# Data Card — PneumoniaMNIST (MedMNIST v2)

## 1. Dataset Overview

- **Dataset Name**: PneumoniaMNIST
- **Benchmark Collection**: MedMNIST v2
- **Source Paper**: Yang et al., *"MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification"*, Nature Scientific Data, 2023.
- **Original Primary Cohort**: Kermany et al., *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, Cell 172(5), 1122-1131, 2018.
- **Dataset License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Local File**: `data/pneumoniamnist.npz`
- **File Size**: 4,170,669 bytes (3.98 MB)
- **MD5 Checksum**: `28209eda62fecd6e6a2d98b1501bb15f`

---

## 2. Dataset Structure & Splits

The dataset provides official, deterministic splits to ensure strict comparability across peer-reviewed benchmarks:

| Split | Number of Images | Normal (Class 0) | Pneumonia (Class 1) | Pneumonia Prevalence |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 4,708 | 1,214 | 3,494 | 74.2% |
| **Validation** | 524 | 135 | 389 | 74.2% |
| **Test** | 624 | 234 | 390 | 62.5% |
| **Total** | **5,856** | **1,583** | **4,273** | **73.0%** |

### Image Specifications
- **Dimensions**: $28 \times 28$ pixels
- **Channels**: 1 (Grayscale)
- **Data Type**: `uint8` with values in $[0, 255]$
- **Preprocessed Projection**: Anterior-Posterior (AP) pediatric chest radiographs centered on the thoracic cavity.

---

## 3. Label Definition & Clinical Origin

- **Class 0 (Normal)**: Radiographs depicting clear lungs without signs of radiological consolidation, infiltration, or effusion.
- **Class 1 (Pneumonia)**: Radiographs depicting bacterial or viral pneumonia manifestations, such as lobar consolidation or diffuse interstitial opacities.
- **Patient Population**: Pediatric patients aged 1 to 5 years treated at Guangzhou Women and Children's Medical Center.

---

## 4. Preprocessing & Normalization Pipeline

1. **Storage Format**: Stored as compressed NumPy `.npz` archive containing:
   - `train_images`, `train_labels`
   - `val_images`, `val_labels`
   - `test_images`, `test_labels`
2. **Tensor Conversion**:
   $$x \in [0, 255] \xrightarrow{\div 255.0} [0.0, 1.0] \xrightarrow{\frac{x - 0.5}{0.5}} [-1.0, 1.0]$$
3. **Invertibility**:
   Unnormalized image for visualization and Grad-CAM blending:
   $$x_{\text{display}} = \text{clip}((x_{\text{tensor}} \times 0.5 + 0.5) \times 255.0, 0, 255) \in \text{uint8}$$
