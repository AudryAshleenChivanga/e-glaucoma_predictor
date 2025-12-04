"""
Glaucoma E-Predictor - FastAPI Backend
A machine learning system for early glaucoma screening from retinal images.
"""
import os
import sys
import asyncio
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

# Optimize TensorFlow memory usage BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import API_HOST, API_PORT, CORS_ORIGINS, DATA_DIR
from preprocessing import (
    load_image_from_bytes,
    preprocess_image,
    validate_image,
    get_image_info
)
from model import get_model, initialize_model, GlaucomaModel


# Response Models
class PredictionResponse(BaseModel):
    success: bool
    probability: float
    percentage: float
    risk_level: str
    prediction: str
    recommendation: str
    confidence: float
    image_info: dict
    timestamp: str


class TrainingResponse(BaseModel):
    success: bool
    message: str
    status: str
    training_id: Optional[str] = None
    details: Optional[dict] = None


class ModelInfoResponse(BaseModel):
    success: bool
    model_loaded: bool
    model_info: Optional[dict] = None
    training_history: Optional[dict] = None
    
    class Config:
        protected_namespaces = ()


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    timestamp: str
    version: str
    
    class Config:
        protected_namespaces = ()


# Training status tracking
training_status = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "message": "Idle",
    "last_training": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    # Startup - Don't load model at startup to save memory
    print("[*] Glaucoma E-Predictor API starting...")
    print("[+] Model will be loaded on first prediction request (lazy loading)")
    yield
    # Shutdown
    print("[*] Shutting down Glaucoma E-Predictor...")


# Create FastAPI application
app = FastAPI(
    title="Glaucoma E-Predictor API",
    description="AI-powered glaucoma detection from retinal images",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - fast health check."""
    return {"status": "healthy", "message": "Glaucoma E-Predictor API"}


@app.get("/health")
async def health_check():
    """Health check endpoint - fast response for Render."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/status", response_model=HealthResponse)
async def full_status():
    """Full status endpoint with model info."""
    model = get_model()
    return HealthResponse(
        status="healthy",
        model_ready=model.model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict glaucoma from a retinal image.
    
    Upload a retinal fundus image to receive a prediction indicating
    the likelihood of glaucoma.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Validate image
        is_valid, error_message = validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Get image info
        image_info = get_image_info(image_bytes)
        
        # Preprocess image
        image = load_image_from_bytes(image_bytes)
        processed_image = preprocess_image(image)
        
        # Get model and make prediction (lazy loading)
        model = get_model()
        if model.model is None:
            print("[*] Loading model for first prediction...")
            model.build_model()
            print("[+] Model loaded successfully!")
        
        prediction = model.predict(processed_image)
        
        return PredictionResponse(
            success=True,
            probability=prediction["probability"],
            percentage=prediction["percentage"],
            risk_level=prediction["risk_level"],
            prediction=prediction["prediction"],
            recommendation=prediction["recommendation"],
            confidence=prediction["confidence"],
            image_info=image_info,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


async def run_training(data_dir: str, epochs: int):
    """Background task for model training."""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["message"] = "Preparing training data..."
        training_status["total_epochs"] = epochs
        
        model = get_model()
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            training_status["message"] = f"Error: Data directory not found at {data_dir}"
            training_status["is_training"] = False
            return
        
        training_status["message"] = "Training in progress..."
        
        # Run training
        history = model.train(data_dir=data_dir, epochs=epochs)
        
        training_status["is_training"] = False
        training_status["progress"] = 100
        training_status["message"] = "Training completed successfully!"
        training_status["last_training"] = {
            "timestamp": datetime.now().isoformat(),
            "final_accuracy": history.get("final_accuracy"),
            "final_val_accuracy": history.get("final_val_accuracy")
        }
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["message"] = f"Training error: {str(e)}"


@app.post("/retrain", response_model=TrainingResponse)
async def retrain_model(
    background_tasks: BackgroundTasks,
    epochs: int = 20,
    data_dir: Optional[str] = None
):
    """
    Retrain the model with new data.
    
    This endpoint triggers background retraining of the model.
    Use /training-status to monitor progress.
    """
    global training_status
    
    if training_status["is_training"]:
        return TrainingResponse(
            success=False,
            message="Training already in progress",
            status="busy",
            details={"current_status": training_status}
        )
    
    # Use provided data_dir or default
    training_data_dir = data_dir or DATA_DIR
    
    # Validate data directory
    if not os.path.exists(training_data_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Data directory not found: {training_data_dir}. Please download the dataset first."
        )
    
    # Generate training ID
    training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Start background training
    background_tasks.add_task(run_training, training_data_dir, epochs)
    
    return TrainingResponse(
        success=True,
        message="Training started successfully",
        status="started",
        training_id=training_id,
        details={
            "epochs": epochs,
            "data_dir": training_data_dir
        }
    )


@app.get("/training-status")
async def get_training_status():
    """Get the current training status."""
    return {
        "success": True,
        **training_status
    }


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the current model."""
    model = get_model()
    
    if model.model is None:
        return ModelInfoResponse(
            success=True,
            model_loaded=False,
            model_info=None,
            training_history=None
        )
    
    return ModelInfoResponse(
        success=True,
        model_loaded=True,
        model_info=model.get_model_summary(),
        training_history=model.training_history if model.training_history else None
    )


@app.post("/initialize-model")
async def init_model():
    """Initialize or reinitialize the model."""
    try:
        model = get_model()
        model.build_model()
        return {
            "success": True,
            "message": "Model initialized successfully",
            "model_info": model.get_model_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")


@app.get("/data-status")
async def check_data_status():
    """Check if training data is available."""
    data_exists = os.path.exists(DATA_DIR)
    
    data_info = {
        "data_available": data_exists,
        "data_path": DATA_DIR
    }
    
    if data_exists:
        # Count images in subdirectories
        try:
            subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
            data_info["classes"] = subdirs
            data_info["image_counts"] = {}
            
            for subdir in subdirs:
                subdir_path = os.path.join(DATA_DIR, subdir)
                images = [f for f in os.listdir(subdir_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                data_info["image_counts"][subdir] = len(images)
            
            data_info["total_images"] = sum(data_info["image_counts"].values())
        except Exception as e:
            data_info["error"] = str(e)
    
    return {
        "success": True,
        **data_info
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

