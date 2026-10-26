# Requirements Traceability Matrix (RTM)

This document tracks all project requirements, their architectural mapping, test verification mechanisms, and implementation status for the AI Medical Imaging Platform.

---

## Traceability Table

| Req ID | Requirement Description | Implementation Files | Verification / Test | Status |
| :---: | :--- | :--- | :--- | :---: |
| **REQ-01** | **Dataset & Verification**: Safe loading of PneumoniaMNIST (.npz), split isolation (Train: 4708, Val: 524, Test: 624), MD5 verification. | `app/ml/dataset.py`<br>`scripts/verify_dataset.py` | `tests/test_dataset.py`<br>`python scripts/verify_dataset.py` | **PASSED (Verified)** |
| **REQ-02** | **Lightweight CNN Architecture**: Compact CNN for 28x28 grayscale chest X-rays, target conv layer for Grad-CAM, save/load checkpoint. | `app/ml/model.py`<br>`app/ml/train.py`<br>`artifacts/best_model.pt` | `tests/test_model.py` | **PASSED (Verified)** |
| **REQ-03** | **Official Test Evaluation**: Objective metrics (Accuracy, Sensitivity, Specificity, Precision, F1, ROC-AUC, Confusion Matrix). | `app/ml/evaluate.py`<br>`app/ml/metrics.py`<br>`reports/metrics.json` | `tests/test_model.py`<br>`python scripts/run_evaluation.py` | **PASSED (Verified)** |
| **REQ-04** | **Explainable AI (Grad-CAM)**: Gradient-weighted class activation mapping on `conv3`, normalized heatmaps, overlay blending, safe hook cleanup. | `app/ml/gradcam.py`<br>`app/ml/predict.py` | `tests/test_gradcam.py` | **PASSED (Verified)** |
| **REQ-05** | **FastAPI Backend**: Health, readiness, model metadata, sample prediction, upload prediction, Grad-CAM explanation. | `app/api/main.py`<br>`app/api/services.py`<br>`app/api/schemas.py` | `tests/test_api.py` | **PASSED (Verified)** |
| **REQ-06** | **Streamlit Research Dashboard**: Interactive web UI with test sample selector, file uploader, score bars, Grad-CAM overlay, disclaimer. | `app/streamlit_app.py` | `py_compile app/streamlit_app.py`<br>Manual UI verification | **PASSED (Verified)** |
| **REQ-07** | **MONAI Medical Imaging Integration**: MONAI dictionary and array transforms, deterministic preprocessing, shape assertions, unit tests, and optional 3D demo. | `app/ml/monai_transforms.py`<br>`app/ml/monai_3d_demo.py` | `tests/test_monai.py` | **In Progress (Phase 3)** |
| **REQ-08** | **Orthanc DICOM Integration**: Orthanc service via Docker Compose, metadata query client, test DICOM upload/ingestion, patient privacy protection. | `compose.yaml`<br>`app/api/orthanc_client.py`<br>`scripts/test_orthanc.py` | `tests/test_orthanc.py` | **In Progress (Phase 4)** |
| **REQ-09** | **DVC Pipeline Reproducibility**: `dvc.yaml` tracking stages (verify $\rightarrow$ train $\rightarrow$ evaluate), artifact lineage without Git bloat. | `dvc.yaml`<br>`.dvcignore` | `dvc repro` / `dvc dag` | **In Progress (Phase 5)** |
| **REQ-10** | **Modern React Frontend**: Clean Vite+React dashboard communicating with FastAPI over HTTP, loading/error states, Grad-CAM visualizer. | `frontend/`<br>`frontend/src/App.jsx` | `npm run build`<br>`tests/test_frontend_build.py` | **In Progress (Phase 6)** |
| **REQ-11** | **Audit Logging & Security**: Research-demo append-only audit event log (predictions, errors, timestamps, no PHI), payload limits, CORS policy. | `app/services/audit_logger.py`<br>`app/api/main.py` | `tests/test_audit_logger.py` | **In Progress (Phase 7)** |
| **REQ-12** | **Docker Multi-Container Profiles**: Multi-service `compose.yaml` (API, React, Streamlit, Orthanc), non-root users, healthchecks, persistent volumes. | `Dockerfile`<br>`compose.yaml` | `docker compose config`<br>`docker compose build` | **In Progress (Phase 8)** |
| **REQ-13** | **Quality Gates & Pytest Coverage**: Comprehensive automated tests across ML, API, MONAI, DICOM, audit logging, and build pipelines. | `tests/` | `python -m pytest -v` | **In Progress (Phase 9)** |
| **REQ-14** | **Technical Documentation**: Model card, data card, architecture diagrams (Mermaid), testing, deployment, and clinical limitation docs. | `docs/` | Peer review & lint check | **In Progress (Phase 10)** |
| **REQ-15** | **Internship Submission Package**: Final verification report, student reflection document, and presentation slides outline. | `docs/reflection.md`<br>`presentation/`<br>`reports/final_verification.md` | Final acceptance audit | **In Progress (Phase 13)** |

---

## Clinical Safety & Educational Governance

All platform components enforce the mandatory disclaimer:
> *"Educational research demo only. Not intended for clinical use, patient diagnosis, treatment planning, triage, or medical decision-making."*
