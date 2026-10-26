"""
Pydantic data schemas for request and response validation.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    timestamp: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    ready: bool
    message: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    name: str
    architecture: str
    num_classes: int
    classes: Dict[int, str]
    checkpoint_epoch: Optional[int]
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    device: str
    disclaimer: str


class PredictionDetail(BaseModel):
    class_id: int
    class_name: str
    model_scores: Dict[str, float]
    score_note: str


class GradCAMDetail(BaseModel):
    target_class: Optional[str] = None
    heatmap_min: Optional[float] = None
    heatmap_max: Optional[float] = None
    heatmap_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: PredictionDetail
    explanation: Optional[GradCAMDetail] = None
    image_preview_base64: str
    inference_time_ms: float
    disclaimer: str
    model_metadata: Dict[str, Any] = {}


class SampleItem(BaseModel):
    index: int
    split: str
    true_label_id: int
    true_label_name: str
    preview_base64: str


class SamplesListResponse(BaseModel):
    total_samples: int
    samples: List[SampleItem]


class ErrorResponse(BaseModel):
    error: str
    detail: str
