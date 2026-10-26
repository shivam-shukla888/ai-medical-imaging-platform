"""
FastAPI application for the AI Chest X-Ray Research Platform.
Provides RESTful endpoints for health status, readiness, model metadata,
prediction, and Grad-CAM visual explanation.
"""
from typing import Optional
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.schemas import (
    HealthResponse,
    ReadyResponse,
    ModelInfoResponse,
    PredictResponse,
    SamplesListResponse,
    ErrorResponse,
)
from app.api.services import get_service

START_TIME = time.time()
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB

app = FastAPI(
    title="AI Chest X-Ray Research Platform API",
    description=(
        "Production-ready FastAPI backend for chest radiograph classification and Grad-CAM explainability.\n\n"
        "**DISCLAIMER**: Educational research demo only. Not intended for clinical use, diagnosis, treatment, or medical decision-making."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local Streamlit frontend and standard web origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check process health",
)
def health():
    """Returns application liveness and uptime status."""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
    summary="Check inference readiness",
)
def ready():
    """Verifies that the model weights are loaded and ready for inference."""
    service = get_service()
    is_ready = service.engine.is_ready()
    return ReadyResponse(
        ready=is_ready,
        message="Model is loaded and ready for inference." if is_ready else "Model checkpoint not loaded. Train model first.",
        model_loaded=is_ready,
        device=str(service.engine.device),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Retrieve model metadata and parameters",
)
def model_info():
    """Returns active architecture details, training checkpoint metrics, and class definitions."""
    service = get_service()
    return service.get_model_info()


@app.get(
    "/samples",
    response_model=SamplesListResponse,
    tags=["Dataset"],
    summary="List available test dataset samples",
)
def list_samples():
    """Returns pre-loaded test split samples with preview thumbnails and ground-truth labels."""
    service = get_service()
    samples = service.list_samples()
    return SamplesListResponse(
        total_samples=len(samples),
        samples=samples,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input image or size limit exceeded"},
        503: {"model": ErrorResponse, "description": "Model engine not ready"},
    },
    tags=["Inference"],
    summary="Run classification prediction on an uploaded image",
)
async def predict(
    file: UploadFile = File(..., description="Grayscale or RGB chest radiograph image (PNG/JPEG)"),
    explain: bool = Query(True, description="Whether to include Grad-CAM visual explanation"),
):
    """
    Classifies an uploaded chest X-ray image (Normal vs. Pneumonia) and computes model scores.
    Optionally computes Grad-CAM heatmaps and visual overlay on the input.
    """
    service = get_service()
    if not service.engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model or ensure artifacts/best_model.pt is present.",
        )

    # Read and validate payload size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image file size ({len(contents)} bytes) exceeds the maximum allowed limit of {MAX_UPLOAD_SIZE} bytes (5 MB).",
        )

    try:
        result = service.predict_image_bytes(contents, generate_explanation=explain)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.post(
    "/explain",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Generate Grad-CAM visual explanation for an uploaded image",
)
async def explain(
    file: UploadFile = File(..., description="Grayscale or RGB chest radiograph image (PNG/JPEG)"),
):
    """
    Dedicated endpoint to compute Grad-CAM attention heatmap and visual overlay.
    """
    return await predict(file=file, explain=True)


@app.post(
    "/predict/sample/{sample_idx}",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Run prediction on a benchmark test dataset sample",
)
def predict_sample(
    sample_idx: int,
    explain: bool = Query(True, description="Whether to include Grad-CAM visual explanation"),
):
    """
    Executes prediction and Grad-CAM on a specific sample from the test split by index.
    """
    service = get_service()
    if not service.engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    try:
        return service.predict_sample_by_index(sample_idx, generate_explanation=explain)
    except IndexError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample inference failed: {str(e)}",
        )
