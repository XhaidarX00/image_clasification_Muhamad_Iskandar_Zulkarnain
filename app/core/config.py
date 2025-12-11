"""
Configuration settings and path definitions for the application.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_DIR = BASE_DIR / "model"

# Model settings
MODEL_PATH = MODEL_DIR / "final_image_classifier_new.keras"
WEIGHTS_DIR = MODEL_DIR / "weights"
MOBILENET_WEIGHTS_PATH = WEIGHTS_DIR / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
THRESHOLD = 0.5

# Class labels (0 = Cat, 1 = Dog based on notebook)
CLASSES = ["Cat", "Dog"]

# JSON file paths
HISTORY_FILE = DATA_DIR / "history.json"
CONFIG_FILE = DATA_DIR / "config.json"

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(exist_ok=True)
