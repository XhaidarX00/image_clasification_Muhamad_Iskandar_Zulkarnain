"""
AI Model singleton for loading and managing the TensorFlow model.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from pathlib import Path

from .config import MODEL_PATH, IMG_SIZE

logger = logging.getLogger(__name__)


class AIModel:
    """Singleton class for managing the AI model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self.load_model()
    
    def load_model(self):
        """Load the model from disk or create a dummy model if not found."""
        try:
            if MODEL_PATH.exists():
                logger.info(f"Loading model from {MODEL_PATH}")
                self._model = keras.models.load_model(str(MODEL_PATH))
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {MODEL_PATH}. Creating dummy model...")
                self._model = self._create_dummy_model()
                # Save the dummy model
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                self._model.save(str(MODEL_PATH))
                logger.info("Dummy model created and saved")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating dummy model as fallback...")
            self._model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy MobileNetV2-based model matching the training architecture."""
        # Base model - MobileNetV2
        base = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        base.trainable = False
        
        # Build the model architecture matching the training service
        inputs = keras.Input(shape=IMG_SIZE + (3,))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Data augmentation layer (used during training)
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        x = data_augmentation(x)
        
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)  # Added for training stability
        x = layers.Dropout(0.3)(x)  # Increased dropout for regularization
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def reload_model(self):
        """Reload the model from disk (used after retraining)."""
        self._model = None
        self.load_model()
    
    def predict(self, image_array):
        """
        Make a prediction on the preprocessed image array.
        
        Args:
            image_array: Numpy array of shape (1, 160, 160, 3)
            
        Returns:
            float: Prediction score between 0 and 1
        """
        if self._model is None:
            self.load_model()
        
        prediction = self._model.predict(image_array, verbose=0)
        return float(prediction[0][0])
    
    @property
    def model(self):
        """Get the underlying Keras model."""
        return self._model


# Global singleton instance
ai_model = AIModel()
