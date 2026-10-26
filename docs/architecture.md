# AI Medical Imaging Platform — Architecture

## 1. System Overview

The **AI Medical Imaging Platform** is an enterprise-grade, modular biomedical research system integrating deep learning, explainability (Grad-CAM), MONAI medical image transforms, Orthanc PACS DICOM workflows, DVC reproducibility, and dual frontends (Modern React + Streamlit fallback).

---

## 2. Mermaid Architecture Diagrams

### Diagram 1: Comprehensive System Architecture

```mermaid
flowchart TD
    subgraph DataLayer ["Data & Reproducibility Layer"]
        RAW["pneumoniamnist.npz\n(MedMNIST v2 Official Archive)"]
        DVC["DVC Pipeline (dvc.yaml)\n(verify -> train -> evaluate)"]
        LOADER["PneumoniaDataset & DataLoaders\n(Train: 4708, Val: 524, Test: 624)"]
        MONAI_T["MONAI Preprocessing Engine\n(ScaleIntensity, Resize, EnsureChannelFirst)"]

        RAW --> DVC
        DVC --> LOADER
        LOADER --> MONAI_T
    end

    subgraph MLLayer ["Machine Learning & XAI Core"]
        CNN["ChestXRayMiniCNN\n(3 Conv Stages, BatchNorm, Dropout)"]
        CKPT[("artifacts/best_model.pt\n(PyTorch Checkpoint, Epoch 4)")]
        GRADCAM["GradCAM Explainability Engine\n(Target: conv3, GAP, Heatmap Overlay)"]
        MONAI_3D["MONAI 3D Volumetric U-Net\n(SpatialDims=3, Synthetic Phantom)"]

        MONAI_T --> CNN
        CNN --> CKPT
        CKPT --> GRADCAM
    end

    subgraph PACSLayer ["DICOM & PACS Ingestion"]
        ORTHANC["Orthanc PACS Server\n(DICOM Port: 4242, HTTP: 8042)"]
        DICOM_INGEST["DICOM Ingestion Service\n(pydicom, PHI Redaction, Pixel Extraction)"]
        ORTHANC_CLIENT["OrthancClient REST Bridge"]

        ORTHANC <--> ORTHANC_CLIENT
        DICOM_INGEST --> ORTHANC_CLIENT
    end

    subgraph BackendLayer ["FastAPI Service Layer (:8000)"]
        API["FastAPI Application (app/api/main.py)"]
        SVC["PlatformService & InferenceEngine"]
        AUDIT["AuditLogger (artifacts/audit_events.jsonl)"]

        API --> SVC
        API --> AUDIT
        SVC --> CNN
        SVC --> GRADCAM
        SVC --> DICOM_INGEST
        SVC --> MONAI_3D
    end

    subgraph FrontendLayer ["User Interface Tier"]
        REACT["React Research Dashboard (:3000 / :5173)\n(Vite + React 18)"]
        STREAMLIT["Streamlit Fallback Dashboard (:8501)\n(Rapid Exploration UI)"]

        REACT -.->|REST API /predict, /samples| API
        STREAMLIT -.->|REST API /predict, /samples| API
    end
```

---

### Diagram 2: Image Ingestion and Inference Workflow

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher / Student
    participant UI as React / Streamlit Frontend
    participant API as FastAPI Backend (:8000)
    participant Ingest as Preprocessing & Ingest Service
    participant Model as ChestXRayMiniCNN
    participant GCAM as Grad-CAM Engine
    participant Audit as AuditLogger

    User->>UI: Select Test Sample or Upload Image / DICOM
    UI->>UI: Validate file size (<= 5 MB) and format
    UI->>API: POST /predict (or /ingest/dicom)
    API->>Audit: Log event: PREDICTION_REQUESTED (no PHI)
    API->>Ingest: Extract pixel array, convert to grayscale, resize (28x28)
    Ingest->>Ingest: Normalize intensities to [-1.0, 1.0]
    Ingest->>Model: Forward pass on normalized tensor (1, 1, 28, 28)
    Model-->>API: Raw logits -> Softmax model scores
    API->>GCAM: Backpropagate predicted class score to conv3
    GCAM-->>API: Compute gradients, GAP weights, ReLU heatmap & overlay
    API->>Audit: Log event: PREDICTION_COMPLETED (latency, class, scores)
    API-->>UI: Return JSON (prediction, scores, Grad-CAM base64, latency)
    UI-->>User: Display classification badge, scores & Grad-CAM overlays
```

---

### Diagram 3: Training, Validation, and Evaluation Pipeline

```mermaid
flowchart LR
    subgraph S1 ["Stage 1: Verify"]
        V1["scripts/verify_dataset.py"] --> V2["MD5 Checksum & Shape Check\n(28209eda62fecd6e6a2d98b1501bb15f)"]
    end

    subgraph S2 ["Stage 2: Train"]
        T1["scripts/run_training.py"] --> T2["Weighted CrossEntropyLoss\n(Normal: 1.939, Pneumonia: 0.674)"]
        T2 --> T3["AdamW Optimizer (lr=0.001)"]
        T3 --> T4["Validation Loss Checkpoint Selection"]
        T4 --> T5[("artifacts/best_model.pt\n(Epoch 4, Val Loss: 0.0709)")]
    end

    subgraph S3 ["Stage 3: Evaluate"]
        E1["scripts/run_evaluation.py"] --> E2["Untouched Test Split (624 Samples)"]
        E2 --> E3["Diagnostic Metrics\n(Acc: 89.7%, Sens: 98.2%, Spec: 75.6%, AUC: 0.942)"]
        E3 --> E4["reports/metrics.json\nreports/evaluation_report.md\nreports/confusion_matrix.png"]
    end

    S1 --> S2 --> S3
```

---

## 3. Technology Stack & Design Decisions

| Technology | Role | Justification |
| :--- | :--- | :--- |
| **PyTorch 2.x** | Core Deep Learning | Native autograd hooks for Grad-CAM, device agnostic execution (CUDA/CPU). |
| **MONAI 1.6+** | Medical Preprocessing | Standardized healthcare imaging transforms, deterministic pipeline, 3D volumetric U-Net support. |
| **Orthanc PACS** | DICOM Storage & Ingestion | Standard open-source vendor-neutral DICOM PACS server with REST API. |
| **DVC 3.6+** | ML Reproducibility | Tracks dataset lineage and training stages without polluting Git with multi-megabyte binary checkpoints. |
| **FastAPI** | REST API Backend | High performance asynchronous Python web framework with typed Pydantic validation. |
| **React 18 + Vite** | Primary Frontend | Production-ready, lightning-fast component dashboard with responsive state management. |
| **Streamlit** | Fallback Frontend | Immediate pythonic UI providing seamless exploration if Node is unavailable. |
| **Docker Compose** | Deployment & Packaging | Multi-container profile architecture (`api`, `frontend`, `streamlit`, `orthanc`). |
