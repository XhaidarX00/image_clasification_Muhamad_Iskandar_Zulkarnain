"""
Prediction service for image classification.
"""
import uuid
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image
import tensorflow as tf

from ..core.config import IMG_SIZE, THRESHOLD, CLASSES, UPLOADS_DIR
from ..core.ai_model import ai_model
from ..repositories.data_repo import history_repo
from ..models.schemas import PredictionResult

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling image predictions."""
    
    def __init__(self):
        self.img_size = IMG_SIZE
        self.threshold = THRESHOLD
        self.classes = CLASSES
    
    def process_image(self, image_bytes: bytes, filename: str) -> PredictionResult:
        """
        Process an uploaded image and return prediction.
        
        Args:
            image_bytes: Raw image bytes from upload
            filename: Original filename
            
        Returns:
            PredictionResult with prediction details
        """
        # Generate unique ID and save path
        prediction_id = str(uuid.uuid4())
        safe_filename = f"{prediction_id}_{filename}"
        save_path = UPLOADS_DIR / safe_filename
        
        # Ensure uploads directory exists
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save the original image
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"Saved image to {save_path}")
        
        # Load and preprocess image
        img = Image.open(save_path)
        img = img.convert('RGB')  # Ensure RGB
        img = img.resize(self.img_size)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Note: MobileNetV2 preprocessing is already built into the model architecture
        # Do NOT apply preprocessing here to avoid double preprocessing
        
        # Get prediction
        raw_score = ai_model.predict(img_array)
        
        # Determine class based on threshold
        # Score > 0.5 = Dog (index 1), else Cat (index 0)
        if raw_score > self.threshold:
            prediction = self.classes[1]  # Dog
            confidence = raw_score
        else:
            prediction = self.classes[0]  # Cat
            confidence = 1 - raw_score
        
        logger.info(f"Prediction: {prediction} (raw: {raw_score:.4f}, confidence: {confidence:.4f})")
        
        # Create result
        result = PredictionResult(
            id=prediction_id,
            filename=filename,
            prediction=prediction,
            confidence=float(confidence),
            raw_score=float(raw_score),
            corrected_label=None,
            timestamp=datetime.now().isoformat(),
            image_path=f"/static/uploads/{safe_filename}"
        )
        
        # Save to history
        history_repo.add(result.model_dump())
        
        return result
    
    def correct_prediction(self, prediction_id: str, corrected_label: str) -> bool:
        """
        Correct a prediction label.
        
        Args:
            prediction_id: ID of the prediction to correct
            corrected_label: The correct label ("Cat" or "Dog")
            
        Returns:
            True if successful, False if prediction not found
        """
        if corrected_label not in self.classes:
            raise ValueError(f"Invalid label. Must be one of: {self.classes}")
        
        result = history_repo.update(prediction_id, corrected_label)
        return result is not None
    
    def get_history(self):
        """Get all prediction history."""
        return history_repo.get_all()
    
    def get_prediction_stats(self):
        """Get aggregated prediction statistics for presentation."""
        history = history_repo.get_all()
        
        if not history:
            return {
                "total_predictions": 0,
                "cat_predictions": 0,
                "dog_predictions": 0,
                "average_confidence": 0,
                "most_confident": [],
                "least_confident": []
            }
        
        # Count predictions by class
        cat_count = sum(1 for h in history if h.get('prediction') == 'Cat')
        dog_count = sum(1 for h in history if h.get('prediction') == 'Dog')
        
        # Calculate average confidence
        confidences = [h.get('confidence', 0) for h in history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Sort by confidence for top/bottom predictions
        sorted_history = sorted(history, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Get top 5 most confident and bottom 5 least confident
        most_confident = sorted_history[:5]
        least_confident = sorted_history[-5:][::-1]  # Reverse to show lowest first
        
        return {
            "total_predictions": len(history),
            "cat_predictions": cat_count,
            "dog_predictions": dog_count,
            "average_confidence": round(avg_confidence, 4),
            "most_confident": [
                {
                    "id": p.get('id'),
                    "filename": p.get('filename'),
                    "prediction": p.get('prediction'),
                    "confidence": round(p.get('confidence', 0), 4),
                    "image_path": p.get('image_path')
                }
                for p in most_confident
            ],
            "least_confident": [
                {
                    "id": p.get('id'),
                    "filename": p.get('filename'),
                    "prediction": p.get('prediction'),
                    "confidence": round(p.get('confidence', 0), 4),
                    "image_path": p.get('image_path')
                }
                for p in least_confident
            ]
        }
    
    def clear_history(self):
        """Clear all prediction history."""
        history_repo.clear()
        logger.info("All history cleared")


# Global service instance
prediction_service = PredictionService()
