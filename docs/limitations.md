# Technical & Clinical Limitations

## 1. Educational Scope & Legal Disclaimer

> **DISCLAIMER**: This application is strictly an educational research demonstration.
> It is **NOT** medical software, not certified by regulatory bodies (e.g., CE-IVD, FDA 510(k)), and MUST NOT be used for real patient diagnostic workflows, clinical decision-making, or triage.

---

## 2. Radiographic & Anatomical Limitations

1. **Extreme Downsampling ($28 \times 28$)**:
   - Diagnostic chest radiographs in clinical practice are captured at spatial resolutions between $2048 \times 2048$ and $3072 \times 3072$ with 12 to 16 bits of dynamic grayscale range.
   - Downsampling to $28 \times 28$ pixels causes catastrophic loss of high-frequency anatomical details: fine interstitial markings, air bronchograms, micro-nodules, apical pneumothoraces, and subtle Kerley B lines are invisible at this scale.
2. **Pediatric Cohort Bias**:
   - The primary dataset consists exclusively of pediatric patients aged 1 to 5 years from a single medical center (Guangzhou Women and Children's Medical Center).
   - Pediatric thoracic anatomy differs substantially from adult anatomy (e.g., prominent thymic sail sign, higher cardiothoracic ratio, horizontal ribs). The model cannot generalize to adult or elderly patient populations.
3. **Absence of Co-morbidity Classification**:
   - The model makes a forced binary decision: `Normal` vs. `Pneumonia`.
   - Real clinical radiographs frequently present with cardiomegaly, atelectasis, pneumothorax, effusion, pulmonary edema, tuberculosis, or foreign body aspiration. The model cannot diagnose or differentiate these conditions.

---

## 3. Machine Learning & Grad-CAM Limitations

1. **Model Scores Are Not Probabilities**:
   - Softmax outputs represent normalized mathematical logits, not well-calibrated posterior clinical probabilities. A high score (e.g., 95%) does not indicate 95% clinical certainty.
2. **Grad-CAM Resolution ($3 \times 3$ Feature Map)**:
   - The final convolutional layer `conv3` produces a $3 \times 3$ spatial feature grid.
   - Bilinear upsampling to $28 \times 28$ produces smooth, diffuse heatmaps that highlight general lung regions rather than sharp anatomical borders or segmental lobes.
3. **Artifact and Shortcut Learning Vulnerability**:
   - Deep neural networks frequently exploit non-pathological shortcuts (e.g., radiologic markers, collimator boundaries, patient positioning, rib rotation).
   - High Grad-CAM excitation in a region does not guarantee that the underlying radiological sign is genuine pathology.

---

## 4. Engineering & Deployment Scope

- **No PACS / DICOM Integration**: This MVP operates on standard 8-bit image formats (`PNG`/`JPEG`) and pre-extracted arrays. It does not implement DICOM network listeners (C-STORE, C-FIND, C-MOVE) or Orthanc/WADO protocols.
- **Single-Node Architecture**: Scaled hospital deployment would require distributed message queues (e.g., Celery/RabbitMQ), persistent patient audit databases, and secure zero-trust credential vaults.
