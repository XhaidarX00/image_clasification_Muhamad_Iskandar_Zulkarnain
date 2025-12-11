"""
Training service for model retraining.
Runs in background tasks with optimized configuration for better accuracy.
"""
import zipfile
import shutil
import tempfile
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from ..core.config import MODEL_PATH, IMG_SIZE, BATCH_SIZE, MOBILENET_WEIGHTS_PATH
from ..core.ai_model import ai_model
from ..repositories.data_repo import config_repo

logger = logging.getLogger(__name__)

# Training configuration
INITIAL_EPOCHS = 10  # Initial training with frozen base
FINE_TUNE_EPOCHS = 15  # Fine-tuning epochs
FINE_TUNE_AT = 100  # Unfreeze layers from this index onwards (last ~30 layers)

# Presentation materials directory
PRESENTATION_DIR = Path("static/presentation")


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
                status="running",
                progress=92,
                message="Generating presentation materials..."
            )
            
            # Combine training history from both phases
            combined_history = self._combine_histories(history1, history2)
            
            # Generate presentation materials
            try:
                self._generate_training_plots(combined_history)
                self._generate_confusion_matrix(model, val_ds)
                self._save_dataset_info(train_ds, val_ds)
                self._save_training_metrics(combined_history, model, val_ds)
                logger.info("Presentation materials generated successfully")
            except Exception as viz_error:
                logger.error(f"Failed to generate visualizations: {viz_error}")
            
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
    
    def _combine_histories(self, history1, history2):
        """Combine training histories from two phases."""
        combined = {}
        for key in history1.history.keys():
            combined[key] = history1.history[key] + history2.history[key]
        return combined
    
    def _generate_training_plots(self, history):
        """Generate and save training history plots."""
        PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 5)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # Accuracy plot
        epochs = range(1, len(history['accuracy']) + 1)
        ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PRESENTATION_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Training plots saved")
    
    def _generate_confusion_matrix(self, model, val_ds):
        """Generate and save confusion matrix."""
        PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get predictions
        y_true = []
        y_pred = []
        
        for images, labels in val_ds:
            predictions = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend((predictions > 0.5).astype(int).flatten())
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Cat', 'Dog'], 
                    yticklabels=['Cat', 'Dog'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(PRESENTATION_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Confusion matrix saved")
    
    def _save_dataset_info(self, train_ds, val_ds):
        """Save dataset information as JSON."""
        PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Count samples
        train_samples = sum([len(labels) for _, labels in train_ds])
        val_samples = sum([len(labels) for _, labels in val_ds])
        
        # Count per class (assuming balanced or getting actual counts)
        train_cat = 0
        train_dog = 0
        val_cat = 0
        val_dog = 0
        
        for _, labels in train_ds:
            train_cat += np.sum(labels.numpy() == 0)
            train_dog += np.sum(labels.numpy() == 1)
        
        for _, labels in val_ds:
            val_cat += np.sum(labels.numpy() == 0)
            val_dog += np.sum(labels.numpy() == 1)
        
        dataset_info = {
            "total_samples": int(train_samples + val_samples),
            "train_samples": int(train_samples),
            "val_samples": int(val_samples),
            "train_cats": int(train_cat),
            "train_dogs": int(train_dog),
            "val_cats": int(val_cat),
            "val_dogs": int(val_dog),
            "split_ratio": "80:20",
            "image_size": list(self.img_size),
            "batch_size": self.batch_size
        }
        
        with open(PRESENTATION_DIR / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        logger.info("Dataset info saved")
    
    def _save_training_metrics(self, history, model, val_ds):
        """Save final training metrics as JSON."""
        PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get final metrics from history
        final_train_acc = float(history['accuracy'][-1])
        final_val_acc = float(history['val_accuracy'][-1])
        final_train_loss = float(history['loss'][-1])
        final_val_loss = float(history['val_loss'][-1])
        
        # Get predictions for precision/recall/f1
        y_true = []
        y_pred = []
        
        for images, labels in val_ds:
            predictions = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend((predictions > 0.5).astype(int).flatten())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            "training": {
                "final_accuracy": round(final_train_acc, 4),
                "final_loss": round(final_train_loss, 4),
                "total_epochs": len(history['accuracy'])
            },
            "validation": {
                "accuracy": round(final_val_acc, 4),
                "loss": round(final_val_loss, 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4)
            },
            "model_info": {
                "architecture": "MobileNetV2",
                "base_weights": "ImageNet",
                "training_strategy": "Two-phase (Feature Extraction + Fine-tuning)",
                "initial_epochs": INITIAL_EPOCHS,
                "fine_tune_epochs": FINE_TUNE_EPOCHS
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(PRESENTATION_DIR / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info("Training metrics saved")


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
