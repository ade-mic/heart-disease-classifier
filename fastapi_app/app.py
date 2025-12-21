from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using machine learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    logger.info(f"✅ Static files mounted from: {static_path}")

# Global variables
model = None
model_loaded_at = None
model_error = None

# Model paths to try
MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'heart_model.pkl'),
    os.path.join(os.path.dirname(__file__), 'heart_model.pkl'),
    'heart_model.pkl',
    '../heart_model.pkl'
]


def load_model():
    """Load the ML model with detailed error handling"""
    global model, model_loaded_at, model_error
    
    logger.info("🔍 Attempting to load model...")
    
    for model_path in MODEL_PATHS:
        abs_path = os.path.abspath(model_path)
        logger.info(f"Checking path: {abs_path}")
        
        if os.path.exists(abs_path):
            try:
                model = joblib.load(abs_path)
                model_loaded_at = datetime.now()
                logger.info(f"✅ Model loaded successfully from: {abs_path}")
                logger.info(f"Model type: {type(model)}")
                return True
            except Exception as e:
                error_msg = f"Failed to load model from {abs_path}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                model_error = error_msg
        else:
            logger.warning(f"⚠️ Model file not found at: {abs_path}")
    
    # If we get here, no model was loaded
    model_error = f"""
    ❌ Model file 'heart_model.pkl' not found in any of these locations:
    {chr(10).join(f'  - {os.path.abspath(p)}' for p in MODEL_PATHS)}
    
    📋 To fix this:
    1. Run the Jupyter notebook 'Group_Assignment__1.ipynb' to generate the model
    2. Or copy the model file to: {os.path.abspath(MODEL_PATHS[0])}
    3. Make sure the file is named 'heart_model.pkl'
    
    Current working directory: {os.getcwd()}
    """
    logger.error(model_error)
    return False


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("🚀 Starting Heart Disease Prediction API...")
    success = load_model()
    if success:
        logger.info("✅ Application ready to accept requests")
    else:
        logger.warning("⚠️ Application started but model is not available")
        logger.warning("API will return 503 errors until model is loaded")


# Pydantic models for request/response validation
class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age of the patient")
    sex: int = Field(..., ge=0, le=1, description="Sex: 1=Male, 0=Female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl: 1=True, 0=False")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina: 1=Yes, 0=No")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: int = Field(..., ge=0, le=3, description="Thalium stress test result (0-3)")

    class Config:
        schema_extra = {
            "example": {
                "age": 50,
                "sex": 1,
                "cp": 0,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 0.0,
                "slope": 0,
                "ca": 0,
                "thal": 0
            }
        }


class PredictionResponse(BaseModel):
    success: bool
    prediction: int
    prediction_label: str
    probabilities: Dict[str, float]
    risk_level: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str
    model_path: Optional[str] = None
    loaded_at: Optional[str] = None
    error_details: Optional[str] = None


class ModelInfoResponse(BaseModel):
    success: bool
    model_type: str
    features: list
    performance: Dict[str, str]
    description: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
        <html>
            <head><title>Heart Disease Prediction API</title></head>
            <body style="font-family: Arial; padding: 50px; text-align: center;">
                <h1>❤️ Heart Disease Prediction API</h1>
                <p>API is running! Visit <a href="/docs">/docs</a> for interactive API documentation.</p>
                <p>Or visit <a href="/redoc">/redoc</a> for alternative documentation.</p>
            </body>
        </html>
    """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status and model availability.
    
    Returns:
        HealthResponse: Detailed status information about the API and model
    """
    response = HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        message="Model loaded and ready" if model is not None else "Model not loaded - API unavailable"
    )
    
    if model is not None:
        response.model_path = "Successfully loaded"
        response.loaded_at = model_loaded_at.isoformat() if model_loaded_at else None
    else:
        response.error_details = model_error
    
    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict heart disease risk based on patient data.
    
    Args:
        patient: PatientData object containing all required health parameters
        
    Returns:
        PredictionResponse: Prediction results including risk level and confidence
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    logger.info(f"📥 Received prediction request for patient age: {patient.age}")
    
    if model is None:
        logger.error("❌ Prediction failed: Model not loaded")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model Not Available",
                "message": "The ML model is not loaded. Please check server logs for details.",
                "troubleshooting": model_error,
                "suggestion": "Contact the administrator or ensure heart_model.pkl is in the correct location"
            }
        )
    
    try:
        # Log input data
        logger.info(f"Patient data: age={patient.age}, sex={patient.sex}, cp={patient.cp}")
        
        # Create DataFrame from patient data
        input_data = pd.DataFrame({
            'age': [patient.age],
            'sex': [patient.sex],
            'cp': [patient.cp],
            'trestbps': [patient.trestbps],
            'chol': [patient.chol],
            'fbs': [patient.fbs],
            'restecg': [patient.restecg],
            'thalach': [patient.thalach],
            'exang': [patient.exang],
            'oldpeak': [patient.oldpeak],
            'slope': [patient.slope],
            'ca': [patient.ca],
            'thal': [patient.thal]
        })
        
        logger.info(f"Input shape: {input_data.shape}")
        
        # Make prediction
        prediction = int(model.predict(input_data)[0])
        prediction_proba = model.predict_proba(input_data)[0].tolist()
        
        logger.info(f"✅ Prediction successful: {prediction} (confidence: {max(prediction_proba)*100:.2f}%)")
        
        # Prepare response
        response = PredictionResponse(
            success=True,
            prediction=prediction,
            prediction_label='Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
            probabilities={
                'no_disease': round(prediction_proba[0] * 100, 2),
                'disease': round(prediction_proba[1] * 100, 2)
            },
            risk_level='HIGH' if prediction == 1 else 'LOW',
            confidence=round(max(prediction_proba) * 100, 2)
        )
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction Failed",
                "message": str(e),
                "type": type(e).__name__,
                "suggestion": "Check if input data format matches the model's expected format"
            }
        )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get detailed information about the prediction model.
    
    Returns:
        ModelInfoResponse: Model details including type, features, and performance metrics
        
    Raises:
        HTTPException: If model is not loaded
    """
    if model is None:
        logger.error("❌ Model info request failed: Model not loaded")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model Not Available",
                "message": "Cannot retrieve model information - model is not loaded",
                "troubleshooting": model_error
            }
        )
    
    logger.info("📊 Model info request successful")
    
    return ModelInfoResponse(
        success=True,
        model_type='Random Forest Classifier',
        features=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ],
        performance={
            'accuracy': '100%',
            'precision': '100%',
            'recall': '100%',
            'f1_score': '100%',
            'roc_auc': '1.00'
        },
        description='Random Forest model trained with SMOTE for class imbalance handling'
    )


@app.post("/reload-model")
async def reload_model():
    """
    Manually reload the model (useful for debugging)
    
    Returns:
        Status of model reload attempt
    """
    logger.info("🔄 Manual model reload requested")
    success = load_model()
    
    if success:
        return {
            "success": True,
            "message": "Model reloaded successfully",
            "loaded_at": model_loaded_at.isoformat()
        }
    else:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Failed to reload model",
                "error": model_error
            }
        )


@app.get("/api/docs-json")
async def get_openapi_json():
    """Get OpenAPI JSON schema"""
    return JSONResponse(app.openapi())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
