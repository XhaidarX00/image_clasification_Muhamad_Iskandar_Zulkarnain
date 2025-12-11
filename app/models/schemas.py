"""
Pydantic models for request/response schemas.
"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PredictionResult(BaseModel):
    """Schema for a single prediction result."""
    id: str
    filename: str
    prediction: str  # "Cat" or "Dog"
    confidence: float  # 0-1
    raw_score: float  # Raw model output
    corrected_label: Optional[str] = None
    timestamp: str
    image_path: str


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    success: bool
    result: Optional[PredictionResult] = None
    error: Optional[str] = None


class CorrectionRequest(BaseModel):
    """Request schema for correction endpoint."""
    corrected_label: str


class CorrectionResponse(BaseModel):
    """Response schema for correction endpoint."""
    success: bool
    message: str


class HistoryResponse(BaseModel):
    """Response schema for history endpoint."""
    success: bool
    predictions: List[PredictionResult]
    total: int


class TrainingStatus(BaseModel):
    """Schema for training status."""
    status: str  # "idle", "running", "completed", "failed"
    progress: Optional[float] = None  # 0-100
    message: Optional[str] = None
    last_updated: str


class TrainingResponse(BaseModel):
    """Response schema for training endpoint."""
    success: bool
    message: str
    status: TrainingStatus


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    timestamp: str
