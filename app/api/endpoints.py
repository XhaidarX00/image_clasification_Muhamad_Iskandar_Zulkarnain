"""
API endpoints for the Cat vs Dog classification application.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from datetime import datetime
import logging
from pathlib import Path
import json

from ..models.schemas import (
    PredictionResponse,
    CorrectionRequest,
    CorrectionResponse,
    HistoryResponse,
    TrainingResponse,
    TrainingStatus,
    HealthResponse
)
from ..services.prediction import prediction_service
from ..services.training import training_service
from ..core.ai_model import ai_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=ai_model.model is not None,
        timestamp=datetime.now().isoformat()
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get a prediction.
    
    - Accepts: image files (jpg, jpeg, png, gif, webp)
    - Returns: prediction result with class, confidence, and image path
    """
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    if file.content_type not in allowed_types:
        return PredictionResponse(
            success=False,
            error=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read file content
        contents = await file.read()
        
        # Process and predict
        result = prediction_service.process_image(contents, file.filename)
        
        return PredictionResponse(
            success=True,
            result=result
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return PredictionResponse(
            success=False,
            error=str(e)
        )


@router.put("/correction/{prediction_id}", response_model=CorrectionResponse)
async def correct_prediction(prediction_id: str, request: CorrectionRequest):
    """
    Correct a prediction label.
    
    - Path param: prediction_id
    - Body: corrected_label ("Cat" or "Dog")
    """
    try:
        success = prediction_service.correct_prediction(
            prediction_id, 
            request.corrected_label
        )
        
        if success:
            return CorrectionResponse(
                success=True,
                message=f"Prediction corrected to: {request.corrected_label}"
            )
        else:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
    except ValueError as e:
        return CorrectionResponse(
            success=False,
            message=str(e)
        )
    except Exception as e:
        logger.error(f"Correction failed: {e}")
        return CorrectionResponse(
            success=False,
            message=str(e)
        )


@router.get("/history", response_model=HistoryResponse)
async def get_history():
    """Get all prediction history."""
    try:
        predictions = prediction_service.get_history()
        return HistoryResponse(
            success=True,
            predictions=predictions,
            total=len(predictions)
        )
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return HistoryResponse(
            success=False,
            predictions=[],
            total=0
        )


@router.delete("/history", response_model=CorrectionResponse)
async def clear_history():
    """Clear all prediction history."""
    try:
        prediction_service.clear_history()
        return CorrectionResponse(
            success=True,
            message="All history cleared successfully"
        )
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        return CorrectionResponse(
            success=False,
            message=str(e)
        )


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Start model retraining with uploaded dataset.
    
    - Accepts: ZIP file containing training data
    - The ZIP should contain subdirectories for each class (e.g., cats/, dogs/)
    - Training runs in background
    """
    # Check if already training
    current_status = training_service.get_status()
    if current_status.get('status') == 'running':
        return TrainingResponse(
            success=False,
            message="Training is already in progress",
            status=TrainingStatus(**current_status)
        )
    
    # Validate file type
    if file.content_type not in ['application/zip', 'application/x-zip-compressed']:
        return TrainingResponse(
            success=False,
            message="Invalid file type. Please upload a ZIP file.",
            status=TrainingStatus(**current_status)
        )
    
    try:
        # Read file content
        contents = await file.read()
        
        # Start training in background
        background_tasks.add_task(
            training_service.start_training,
            contents,
            file.filename
        )
        
        return TrainingResponse(
            success=True,
            message="Training started. Check status for progress.",
            status=TrainingStatus(
                status="running",
                progress=0,
                message="Starting training...",
                last_updated=datetime.now().isoformat()
            )
        )
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return TrainingResponse(
            success=False,
            message=str(e),
            status=TrainingStatus(**current_status)
        )


@router.get("/training-status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    status = training_service.get_status()
    return TrainingStatus(**status)


# Presentation endpoints
@router.get("/presentation/materials")
async def get_presentation_materials():
    """Get list of available presentation materials."""
    presentation_dir = Path("static/presentation")
    
    if not presentation_dir.exists():
        return {
            "available": False,
            "message": "No presentation materials available. Please train a model first.",
            "materials": []
        }
    
    materials = []
    
    # Check for training plots
    if (presentation_dir / "training_history.png").exists():
        materials.append({
            "type": "image",
            "name": "Training History",
            "path": "/static/presentation/training_history.png"
        })
    
    # Check for confusion matrix
    if (presentation_dir / "confusion_matrix.png").exists():
        materials.append({
            "type": "image",
            "name": "Confusion Matrix",
            "path": "/static/presentation/confusion_matrix.png"
        })
    
    # Check for JSON files
    if (presentation_dir / "training_metrics.json").exists():
        materials.append({
            "type": "json",
            "name": "Training Metrics",
            "path": "/api/presentation/metrics"
        })
    
    if (presentation_dir / "dataset_info.json").exists():
        materials.append({
            "type": "json",
            "name": "Dataset Info",
            "path": "/api/presentation/dataset-info"
        })
    
    return {
        "available": len(materials) > 0,
        "total": len(materials),
        "materials": materials
    }


@router.get("/presentation/metrics")
async def get_training_metrics():
    """Get training metrics JSON."""
    metrics_path = Path("static/presentation/training_metrics.json")
    
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Training metrics not found. Please train a model first."
        )
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


@router.get("/presentation/dataset-info")
async def get_dataset_info():
    """Get dataset information JSON."""
    dataset_path = Path("static/presentation/dataset_info.json")
    
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset info not found. Please train a model first."
        )
    
    with open(dataset_path, 'r') as f:
        return json.load(f)


@router.get("/presentation/prediction-stats")
async def get_prediction_statistics():
    """Get prediction statistics for presentation."""
    try:
        stats = prediction_service.get_prediction_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get prediction stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "stats": {}
        }
