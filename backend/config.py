"""
Configuration settings for the Glaucoma E-Predictor backend.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/glaucoma_model.keras")
MODEL_INPUT_SIZE = (224, 224)
MODEL_CHANNELS = 3

# Training Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.2"))

# Data Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
GLAUCOMA_DIR = os.path.join(DATA_DIR, "glaucoma")
NORMAL_DIR = os.path.join(DATA_DIR, "normal")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Prediction Thresholds
HIGH_RISK_THRESHOLD = 0.7
MODERATE_RISK_THRESHOLD = 0.4

