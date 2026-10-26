"""
Integration and unit tests for FastAPI endpoints using TestClient.
"""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.api.main import app

client = TestClient(app)


def create_dummy_png_bytes(size=(28, 28), color=128) -> bytes:
    """Creates a valid in-memory PNG byte stream."""
    img = Image.new("L", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_api_health():
    """Verify /health endpoint returns healthy status and uptime."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "timestamp" in data


def test_api_ready():
    """Verify /ready endpoint returns readiness and device."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["model_loaded"] is True
    assert "device" in data


def test_api_model_info():
    """Verify /model-info endpoint returns metadata and disclaimer."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "architecture" in data
    assert "ChestXRayMiniCNN" in data["architecture"]
    assert "Normal" in data["classes"].values()
    assert "Pneumonia" in data["classes"].values()
    assert "disclaimer" in data


def test_api_list_samples():
    """Verify /samples endpoint lists benchmark test images."""
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert data["total_samples"] > 0
    sample0 = data["samples"][0]
    assert "preview_base64" in sample0
    assert sample0["true_label_name"] in ["Normal", "Pneumonia"]


def test_api_predict_sample():
    """Verify /predict/sample/{idx} returns valid prediction and Grad-CAM."""
    response = client.post("/predict/sample/0?explain=true")
    assert response.status_code == 200
    data = response.json()

    # Check prediction block
    pred = data["prediction"]
    assert pred["class_name"] in ["Normal", "Pneumonia"]
    assert "Normal" in pred["model_scores"]
    assert "Pneumonia" in pred["model_scores"]
    assert "score_note" in pred

    # Check explanation block
    exp = data["explanation"]
    assert exp is not None
    assert "overlay_base64" in exp
    assert exp["overlay_base64"].startswith("data:image/png;base64,")

    # Check disclaimer and timing
    assert "disclaimer" in data
    assert data["inference_time_ms"] > 0


def test_api_predict_uploaded_image():
    """Verify /predict endpoint handles real uploaded image file."""
    png_bytes = create_dummy_png_bytes()
    files = {"file": ("test_xray.png", png_bytes, "image/png")}

    response = client.post("/predict?explain=true", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]["class_name"] in ["Normal", "Pneumonia"]
    assert "overlay_base64" in data["explanation"]


def test_api_predict_corrupted_file():
    """Verify /predict endpoint rejects corrupt/non-image bytes with 400 Bad Request."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_CONTENT"
    files = {"file": ("corrupt.png", corrupt_bytes, "image/png")}

    response = client.post("/predict", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
