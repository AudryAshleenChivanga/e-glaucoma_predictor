"""
CNN Model for Glaucoma Detection from Retinal Images.
"""
import os
import numpy as np
from typing import Tuple, Optional, Dict, Any
import json
from datetime import datetime

from config import (
    MODEL_PATH, MODEL_INPUT_SIZE, MODEL_CHANNELS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT,
    DATA_DIR
)

# Lazy import TensorFlow to save memory at startup
tf = None
keras = None
layers = None
models = None
optimizers = None
callbacks = None
ImageDataGenerator = None

def _import_tensorflow():
    """Lazily import TensorFlow when needed."""
    global tf, keras, layers, models, optimizers, callbacks, ImageDataGenerator
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        keras = tf.keras
        layers = tf.keras.layers
        models = tf.keras.models
        optimizers = tf.keras.optimizers
        callbacks = tf.keras.callbacks
        from tensorflow.keras.preprocessing.image import ImageDataGenerator as _IDG
        ImageDataGenerator = _IDG


class GlaucomaModel:
    """CNN model for glaucoma detection."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[keras.Model] = None
        self.input_shape = (*MODEL_INPUT_SIZE, MODEL_CHANNELS)
        self.training_history: Dict[str, Any] = {}
        
    def build_model(self):
        """
        Build a CNN architecture for glaucoma classification.
        Uses a custom architecture optimized for retinal image analysis.
        """
        _import_tensorflow()
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        self.model = model
        return model
    
    def load_model(self) -> bool:
        """
        Load a pre-trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        _import_tensorflow()
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
                return True
            else:
                print(f"No model found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, path: Optional[str] = None) -> bool:
        """
        Save the current model to disk.
        
        Args:
            path: Optional custom path. Uses default if not provided.
        
        Returns:
            True if saved successfully, False otherwise
        """
        save_path = path or self.model_path
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Make a prediction on a preprocessed image.
        
        Args:
            image: Preprocessed image array with shape (1, H, W, C)
        
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or build_model() first.")
        
        # Get prediction probability
        probability = float(self.model.predict(image, verbose=0)[0][0])
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH"
            recommendation = "Immediate consultation with an ophthalmologist is strongly recommended."
        elif probability >= 0.4:
            risk_level = "MODERATE"
            recommendation = "Schedule an appointment with an eye specialist for further evaluation."
        else:
            risk_level = "LOW"
            recommendation = "Continue regular eye check-ups. No immediate concern detected."
        
        return {
            "probability": probability,
            "percentage": round(probability * 100, 2),
            "risk_level": risk_level,
            "prediction": "Glaucoma Detected" if probability >= 0.5 else "Normal",
            "recommendation": recommendation,
            "confidence": round(abs(probability - 0.5) * 2 * 100, 2)  # Confidence score
        }
    
    def create_data_generators(self, data_dir: str):
        """
        Create data generators for training and validation.
        
        Args:
            data_dir: Path to the data directory
        
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        _import_tensorflow()
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=VALIDATION_SPLIT
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=MODEL_INPUT_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Validation generator (no augmentation except rescaling)
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=MODEL_INPUT_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def train(
        self,
        data_dir: str = DATA_DIR,
        epochs: int = EPOCHS,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model on the dataset.
        
        Args:
            data_dir: Path to training data directory
            epochs: Number of training epochs
            save_best: Whether to save the best model during training
        
        Returns:
            Dictionary containing training results
        """
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(data_dir)
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        if save_best:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            callback_list.append(
                callbacks.ModelCheckpoint(
                    self.model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            )
        
        # Train the model
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        # Store training history
        self.training_history = {
            "timestamp": datetime.now().isoformat(),
            "epochs_trained": len(history.history['loss']),
            "final_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }
        
        # Save training history
        history_path = self.model_path.replace('.keras', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.training_history
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Count parameters
        trainable = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
        non_trainable = np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights])
        
        return {
            "input_shape": self.input_shape,
            "total_parameters": int(trainable + non_trainable),
            "trainable_parameters": int(trainable),
            "non_trainable_parameters": int(non_trainable),
            "layers": len(self.model.layers),
            "model_path": self.model_path
        }


# Global model instance
glaucoma_model = GlaucomaModel()


def get_model() -> GlaucomaModel:
    """Get the global model instance."""
    return glaucoma_model


def initialize_model() -> bool:
    """
    Initialize the model - load from disk or build new.
    
    Returns:
        True if model is ready, False otherwise
    """
    model = get_model()
    
    # Try to load existing model
    if model.load_model():
        return True
    
    # Build new model if none exists
    print("Building new model...")
    model.build_model()
    return True

