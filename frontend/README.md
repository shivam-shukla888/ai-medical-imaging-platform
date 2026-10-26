# React Research Dashboard

Lightweight, modern React research dashboard built with **Vite** for the **AI Chest X-Ray Research Platform**.

## Features
- **Interactive Radiograph Analysis**: Select official PneumoniaMNIST test samples or upload local grayscale radiographs.
- **Explainability**: Live visual rendering of input radiographs, Grad-CAM heatmaps, and alpha-blended overlays.
- **Model Score Visualizer**: Uncalibrated score progression bars with explicit clinical disclaimer.
- **Benchmark Evaluation**: Instant summary of official test split metrics (Accuracy, Sensitivity, Specificity, ROC-AUC).
- **MONAI 3D Integration**: Interactive triggering of 3D volumetric U-Net segmentation demo.
- **Platform Audit Log**: Live view of append-only research interaction logs.

---

## Development Setup

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment (Optional)
By default, the frontend connects to `http://localhost:8000`. To customize:
```bash
# Windows PowerShell
$env:VITE_API_URL="http://localhost:8000"
```

### 3. Start Development Server
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 4. Build for Production
```bash
npm run build
```
Generates production assets in `frontend/dist/`.
