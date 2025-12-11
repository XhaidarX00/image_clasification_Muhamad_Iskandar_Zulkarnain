"""
Training service for model retraining.
Runs in background tasks with optimized configuration for better accuracy.
"""
import zipfile
import shutil
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..core.config import MODEL_PATH, IMG_SIZE, BATCH_SIZE, MOBILENET_WEIGHTS_PATH
from ..core.ai_model import ai_model
from ..repositories.data_repo import config_repo

logger = logging.getLogger(__name__)

# Training configuration
INITIAL_EPOCHS = 10  # Initial training with frozen base
FINE_TUNE_EPOCHS = 15  # Fine-tuning epochs
FINE_TUNE_AT = 100  # Unfreeze layers from this index onwards (last ~30 layers)


class TrainingService:
    """Service for handling model retraining."""
    
    def __init__(self):
        self.img_size = IMG_SIZE
        self.batch_size = BATCH_SIZE
    
    async def start_training(self, zip_bytes: bytes, filename: str):
        """
        Start the retraining process with uploaded ZIP file.
        
        Args:
            zip_bytes: Raw bytes of the ZIP file
            filename: Original filename
        """
        temp_dir = None
        try:
            # Update status to running
            config_repo.update_training_status(
                status="running",
                progress=0,
                message="Starting training..."
            )
            
            # Create temp directory for extraction
            temp_dir = Path(tempfile.mkdtemp())
            zip_path = temp_dir / filename
            
            # Save ZIP file
            with open(zip_path, 'wb') as f:
                f.write(zip_bytes)
            
            config_repo.update_training_status(
                status="running",
                progress=10,
                message="Extracting dataset..."
            )
            
            # Extract ZIP
            extract_dir = temp_dir / "dataset"
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            
            # Find the data directory
            data_dir = self._find_data_directory(extract_dir)
            if not data_dir:
                raise ValueError("Could not find valid dataset structure in ZIP file")
            
            config_repo.update_training_status(
                status="running",
                progress=20,
                message="Loading dataset..."
            )
            
            # Load datasets
            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                image_size=self.img_size,
                batch_size=self.batch_size,
                label_mode='int',
                validation_split=0.2,
                subset='training',
                seed=42
            )
            
            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                image_size=self.img_size,
                batch_size=self.batch_size,
                label_mode='int',
                validation_split=0.2,
                subset='validation',
                seed=42
            )
            
            # Optimize datasets with caching and shuffling
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            
            config_repo.update_training_status(
                status="running",
                progress=30,
                message="Building model..."
            )
            
            # Build model (same architecture as notebook)
            model = self._build_model()
            
            config_repo.update_training_status(
                status="running",
                progress=40,
                message="Training model..."
            )
            
            # Phase 1: Train with frozen base model
            total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
            
            callbacks_phase1 = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-7
                ),
                TrainingProgressCallback(
                    total_epochs=total_epochs,
                    phase="Phase 1: Feature Extraction",
                    epoch_offset=0
                )
            ]
            
            logger.info("Phase 1: Training with frozen base model...")
            history1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=INITIAL_EPOCHS,
                callbacks=callbacks_phase1
            )
            
            config_repo.update_training_status(
                status="running",
                progress=50,
                message="Phase 2: Fine-tuning base model..."
            )
            
            # Phase 2: Fine-tune the base model
            # Find the MobileNetV2 base model by searching layers
            base_model = None
            for layer in model.layers:
                if 'mobilenetv2' in layer.name.lower() or isinstance(layer, tf.keras.Model):
                    if hasattr(layer, 'layers') and len(layer.layers) > 10:
                        base_model = layer
                        break
            
            if base_model is not None:
                base_model.trainable = True
                # Freeze all layers except the last ones
                for layer in base_model.layers[:FINE_TUNE_AT]:
                    layer.trainable = False
                logger.info(f"Fine-tuning enabled for {base_model.name}, unfreezing from layer {FINE_TUNE_AT}")
            else:
                logger.warning("Could not find base model for fine-tuning, skipping Phase 2")
            
            # Recompile with lower learning rate for fine-tuning
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            callbacks_phase2 = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-8
                ),
                TrainingProgressCallback(
                    total_epochs=total_epochs,
                    phase="Phase 2: Fine-tuning",
                    epoch_offset=INITIAL_EPOCHS
                )
            ]
            
            logger.info("Phase 2: Fine-tuning base model layers...")
            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=FINE_TUNE_EPOCHS,
                callbacks=callbacks_phase2
            )
            
            config_repo.update_training_status(
                status="running",
                progress=90,
                message="Saving model..."
            )
            
            # Save model
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(MODEL_PATH))
            
            # Reload model in singleton
            ai_model.reload_model()
            
            config_repo.update_training_status(
                status="completed",
                progress=100,
                message="Training completed successfully!"
            )
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            config_repo.update_training_status(
                status="failed",
                progress=None,
                message=f"Training failed: {str(e)}"
            )
        finally:
            # Cleanup temp directory
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir: {e}")
    
    def _find_data_directory(self, extract_dir: Path) -> Path:
        """Find the directory containing class subdirectories."""
        # Check if extract_dir itself has class subdirectories
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if len(subdirs) >= 2:
            # Check if subdirs contain images
            for subdir in subdirs:
                images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")) + list(subdir.glob("*.jpeg"))
                if images:
                    return extract_dir
        
        # Check one level deeper (common for ZIPs with root folder)
        for subdir in subdirs:
            inner_subdirs = [d for d in subdir.iterdir() if d.is_dir()]
            if len(inner_subdirs) >= 2:
                for inner_subdir in inner_subdirs:
                    images = list(inner_subdir.glob("*.jpg")) + list(inner_subdir.glob("*.png")) + list(inner_subdir.glob("*.jpeg"))
                    if images:
                        return subdir
        
        return None
    
    def _build_model(self) -> keras.Model:
        """Build the MobileNetV2-based model."""
        # Check if local weights exist
        if MOBILENET_WEIGHTS_PATH.exists():
            weights_path = str(MOBILENET_WEIGHTS_PATH)
            logger.info(f"Using local MobileNetV2 weights: {weights_path}")
        else:
            # Fallback to downloading from internet if local file not found
            weights_path = 'imagenet'
            logger.warning("Local weights not found, downloading from internet...")
        
        # Base model
        base = tf.keras.applications.MobileNetV2(
            input_shape=self.img_size + (3,),
            include_top=False,
            weights=weights_path
        )
        base.trainable = False
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Build model
        inputs = keras.Input(shape=self.img_size + (3,))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
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
    
    def get_status(self):
        """Get current training status."""
        return config_repo.get_training_status()


class TrainingProgressCallback(keras.callbacks.Callback):
    """Callback to update training progress with phase support."""
    
    def __init__(self, total_epochs, phase="Training", epoch_offset=0):
        super().__init__()
        self.total_epochs = total_epochs
        self.phase = phase
        self.epoch_offset = epoch_offset
    
    def on_epoch_end(self, epoch, logs=None):
        # Calculate actual epoch number across all phases
        actual_epoch = self.epoch_offset + epoch + 1
        # Progress from 40% to 90% during training
        progress = 40 + (actual_epoch / self.total_epochs) * 50
        val_acc = logs.get('val_accuracy', logs.get('accuracy', 0))
        config_repo.update_training_status(
            status="running",
            progress=progress,
            message=f"{self.phase} - Epoch {actual_epoch}/{self.total_epochs} - Val Acc: {val_acc:.4f}"
        )


# Global service instance
training_service = TrainingService()
