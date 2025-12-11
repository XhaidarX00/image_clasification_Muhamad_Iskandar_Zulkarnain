"""
Data repository for JSON-based persistence.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ..core.config import HISTORY_FILE, CONFIG_FILE, DATA_DIR

logger = logging.getLogger(__name__)


class HistoryRepository:
    """Repository for managing prediction history in JSON file."""
    
    def __init__(self):
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the history JSON file exists."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not HISTORY_FILE.exists():
            self._write_data([])
    
    def _read_data(self) -> List[dict]:
        """Read all history data from file."""
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_data(self, data: List[dict]):
        """Write history data to file."""
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_all(self) -> List[dict]:
        """Get all prediction history."""
        return self._read_data()
    
    def add(self, prediction: dict) -> dict:
        """Add a new prediction to history."""
        data = self._read_data()
        data.insert(0, prediction)  # Add to beginning (newest first)
        self._write_data(data)
        logger.info(f"Added prediction {prediction.get('id')} to history")
        return prediction
    
    def get_by_id(self, prediction_id: str) -> Optional[dict]:
        """Get a specific prediction by ID."""
        data = self._read_data()
        for item in data:
            if item.get('id') == prediction_id:
                return item
        return None
    
    def update(self, prediction_id: str, corrected_label: str) -> Optional[dict]:
        """Update a prediction with corrected label."""
        data = self._read_data()
        for item in data:
            if item.get('id') == prediction_id:
                item['corrected_label'] = corrected_label
                self._write_data(data)
                logger.info(f"Updated prediction {prediction_id} with correction: {corrected_label}")
                return item
        return None
    
    def clear(self):
        """Clear all history and delete uploaded images."""
        # Import here to avoid circular imports
        from ..core.config import UPLOADS_DIR
        
        # Get all predictions to find image paths
        data = self._read_data()
        
        # Delete each uploaded image file
        deleted_count = 0
        for prediction in data:
            image_path = prediction.get('image_path', '')
            if image_path:
                # Extract filename from path like "/static/uploads/filename.jpg"
                filename = image_path.split('/')[-1]
                file_path = UPLOADS_DIR / filename
                
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        # Clear the JSON data
        self._write_data([])
        logger.info(f"History cleared. Deleted {deleted_count} image files.")


class ConfigRepository:
    """Repository for managing application configuration."""
    
    def __init__(self):
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the config JSON file exists."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CONFIG_FILE.exists():
            default_config = {
                "training_status": {
                    "status": "idle",
                    "progress": None,
                    "message": None,
                    "last_updated": datetime.now().isoformat()
                }
            }
            self._write_data(default_config)
    
    def _read_data(self) -> dict:
        """Read config data from file."""
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _write_data(self, data: dict):
        """Write config data to file."""
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get(self) -> dict:
        """Get all configuration."""
        return self._read_data()
    
    def get_training_status(self) -> dict:
        """Get current training status."""
        config = self._read_data()
        return config.get("training_status", {
            "status": "idle",
            "progress": None,
            "message": None,
            "last_updated": datetime.now().isoformat()
        })
    
    def update_training_status(
        self, 
        status: str, 
        progress: Optional[float] = None, 
        message: Optional[str] = None
    ):
        """Update training status."""
        config = self._read_data()
        config["training_status"] = {
            "status": status,
            "progress": progress,
            "message": message,
            "last_updated": datetime.now().isoformat()
        }
        self._write_data(config)
        logger.info(f"Training status updated: {status}")


# Global repository instances
history_repo = HistoryRepository()
config_repo = ConfigRepository()
