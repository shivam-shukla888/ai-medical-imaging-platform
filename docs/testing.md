# Testing & Verification Guide

This project includes a comprehensive, automated test suite covering dataset integrity, model architectures, metrics computation, Grad-CAM hooks, and REST API endpoints.

---

## 1. Automated Test Suite (PyTest)

Run all tests using the project's Python environment:

```powershell
python -m pytest -v
```

To run with coverage reporting:

```powershell
python -m pytest -v --cov=app --cov-report=term-missing
```

### Test Modules

1. **`tests/test_dataset.py`**:
   - `test_dataset_verification`: Validates `pneumoniamnist.npz` checksum (MD5: `28209eda62fecd6e6a2d98b1501bb15f`), split sizes, and array dimensions.
   - `test_pneumonia_dataset_loader`: Confirms PyTorch `Dataset` produces `(1, 28, 28)` float32 tensors standardized to $[-1.0, 1.0]$.
   - `test_dataloaders_split_separation`: Verifies zero leakage across train, val, and test splits.

2. **`tests/test_model.py`**:
   - `test_model_forward_shape`: Tests that forward pass returns `(B, 2)` without NaN or infinite values across various batch sizes.
   - `test_target_conv_layer`: Validates the existence and dimensions of `conv3` for feature extraction.
   - `test_model_save_and_load_consistency`: Ensures serialized checkpoints reload with identical numerical predictions.
   - `test_metrics_calculation`: Validates accuracy, precision, recall, specificity, F1, ROC-AUC, and confusion matrix arithmetic.

3. **`tests/test_gradcam.py`**:
   - `test_gradcam_heatmap_dimensions_and_values`: Asserts Grad-CAM heatmaps are exactly `28x28`, finite, and normalized to $[0.0, 1.0]$.
   - `test_gradcam_overlay_synthesis`: Verifies alpha-blending generates valid uint8 `(28, 28, 3)` RGB visuals.
   - `test_gradcam_hook_cleanup`: Asserts that forward and backward hooks are cleanly unregistered on cleanup.

4. **`tests/test_api.py`**:
   - `test_api_health`: Confirms `GET /health` returns status `healthy` and uptime.
   - `test_api_ready`: Confirms `GET /ready` returns readiness state and device.
   - `test_api_model_info`: Validates metadata and disclaimer fields.
   - `test_api_list_samples`: Tests retrieval of benchmark samples with thumbnails.
   - `test_api_predict_sample`: Verifies sample prediction with model scores and Grad-CAM overlay.
   - `test_api_predict_uploaded_image`: Simulates multipart image upload and returns valid predictions.
   - `test_api_predict_corrupted_file`: Confirms that invalid files are safely rejected with `400 Bad Request`.

---

## 2. Command-Line Smoke Tests

### Dataset Verification
```powershell
python scripts/verify_dataset.py
```

### Full Retraining Smoke Test
```powershell
python scripts/run_training.py --epochs 2 --batch-size 64
```

### Test Split Evaluation
```powershell
python scripts/run_evaluation.py
```
