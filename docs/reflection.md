# Engineering Reflection & Internship Retrospective

## 1. What Was Built

During this project, we built a comprehensive, production-grade **AI Medical Imaging Platform** focused on pediatric chest radiograph interpretation and biomedical machine learning engineering:

1. **Robust Core Deep Learning Pipeline**:
   - Custom 3-stage convolutional neural network (`ChestXRayMiniCNN`) tailored for $28 \times 28$ grayscale images.
   - Dynamic class-frequency loss weighting mitigating a 74% majority class imbalance in the training split.
   - Checkpointed based on validation loss, achieving **89.74% accuracy**, **98.21% sensitivity**, and an **ROC-AUC of 0.9418** on the official untouched MedMNIST PneumoniaMNIST test split.
2. **Explainable AI (Grad-CAM)**:
   - Gradient-weighted class activation mapping targeting layer `conv3`.
   - Generates normalized 2D heatmaps and alpha-blended anatomical overlays, offering insight into neural feature excitation.
3. **MONAI Integration**:
   - Deterministic medical-imaging transforms (`EnsureChannelFirst`, `ScaleIntensityRange`, `Resize`, `EnsureType`).
   - Educational 3D volumetric U-Net demonstration on a synthetic 3D phantom evaluating Dice score and loss.
4. **Orthanc PACS & DICOM Ingestion**:
   - Orthanc server integration via Docker Compose with persistent volume storage.
   - Privacy-preserving DICOM ingestion service redacting Protected Health Information (PHI) before tensor conversion.
5. **DVC Pipeline Reproducibility**:
   - Fully defined `dvc.yaml` tracking stages (`verify_data` $\rightarrow$ `train` $\rightarrow$ `evaluate`) and artifact dependencies.
6. **Dual Frontend Architecture**:
   - Primary Modern React dashboard (Vite + React 18) communicating strictly via REST HTTP.
   - Streamlit dashboard preserved as a complete Python-native fallback.
7. **Security & Auditability**:
   - Thread-safe, append-only research audit logging (`artifacts/audit_events.jsonl`).
   - 5 MB upload limits, CORS restrictions, input dimension validation, and prominent clinical disclaimers.
8. **Automated Quality Gates**:
   - 29 unit and integration tests passing in Pytest across ML, API, MONAI, DICOM, DVC, and frontend build outputs.

---

## 2. Technical Challenges & Key Insights

1. **Resolution vs. Computational Overhead**:
   - Clinical chest radiographs are typically $2048 \times 2048$, requiring multi-gigabyte storage and heavy cloud GPU clusters.
   - By adopting the peer-reviewed MedMNIST v2 PneumoniaMNIST benchmark ($28 \times 28$), we preserved realistic clinical challenges (class imbalance, thoracic anatomical features) while enabling full model training in ~25 seconds on a laptop CPU.
2. **Explainability Reliability (Grad-CAM)**:
   - Ensuring that hooks do not leak memory during high-concurrency API calls required explicit hook lifecycle management via `remove_hooks()` and context managers.
   - Downsampling to $3 \times 3$ feature maps in the final conv layer means Grad-CAM heatmaps highlight general lung zones rather than micro-pathology. This limitation was transparently documented in the UI and model card.
3. **Decoupled Service Communication**:
   - Avoiding the temptation to duplicate model inference inside the Streamlit or React UI kept the client bundles lightweight and decoupled tensor compute from rendering.
4. **Graceful PACS Degradation**:
   - In real hospital environments, PACS network connections can fail. We designed `OrthancClient` to fail gracefully: if Orthanc is offline, the primary classification and Grad-CAM endpoints remain 100% operational.

---

## 3. What Worked Well

- **Class-Weighted CrossEntropy**: Boosted test sensitivity to 98.21%, ensuring pneumonia cases are rarely missed as false negatives.
- **FastAPI OpenAPI Schema Validation**: Pydantic models eliminated payload malformation bugs before images reached PyTorch tensors.
- **Vite React Architecture**: Instant HMR and a 657ms production build resulted in a responsive, modern interface.
- **Docker Compose Profiles**: Separating the Orthanc PACS into the `--profile pacs` option allowed lightweight laptops to run the core ML demo without heavy multi-gigabyte container overhead.

---

## 4. What Did Not Work & Remaining Limitations

- **Arbitrary External Radiograph Ingestion**:
  While the pipeline accepts arbitrary image uploads, uploading high-resolution adult radiographs or non-chest images can produce out-of-distribution hallucinations because the model was trained exclusively on pediatric AP projections.
- **DICOM Viewer Limitations**:
  Our React frontend renders extracted 8-bit PNG previews of DICOM slices rather than providing full 16-bit interactive Hounsfield Unit window-leveling (W/L) controls found in clinical diagnostic workstations (e.g. CornerstoneJS/OHIF Viewer).
- **Audit Log Immutability**:
  While the append-only JSONL log prevents accidental overwrites within the application process, it is not cryptographically signed or backed by an immutable ledger.

---

## 5. Future Engineering Directions

1. **OHIF / Cornerstone.js Web DICOM Viewer**: Integrate interactive window-leveling, zoom/pan, and multi-planar reconstruction (MPR) into the React frontend.
2. **Multi-label Co-morbidity Classification**: Extend model heads to identify pleural effusion, cardiomegaly, pneumothorax, and atelectasis simultaneously.
3. **Cloud Remote DVC Backend**: Configure an S3 or Google Cloud Storage remote for centralized team model artifact tracking.
4. **HL7 / FHIR Integration**: Connect inference event outputs to simulated electronic health record (EHR) message brokers.
