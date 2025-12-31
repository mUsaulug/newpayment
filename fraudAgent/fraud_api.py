"""
Fraud Detection API Service
============================
FastAPI service exposing XGBoost fraud detection model.

Usage:
    uvicorn fraud_api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict - Predict fraud probability from features
    GET /health - Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgboost_fraud_model_latest.pkl"

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost-based fraud detection for payment transactions",
    version="1.0.0"
)

# Load model on startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Model not found at {MODEL_PATH}, using fallback rules")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")


# Feature input schema
class FraudFeatures(BaseModel):
    # Time features
    hour: int = 12
    dayofweek: int = 3
    day: int = 15
    month: int = 6
    year: int = 2024
    is_weekend: int = 0
    is_night: int = 0
    age: float = 35.0
    
    # Amount features
    amt: float
    amt_log: float
    amt_zscore: float
    
    # Geo features
    distance_km: float = 0.0
    distance_log: float = 0.0
    city_pop_log: float = 14.0
    lat: float = 41.0
    long: float = 29.0  # Note: 'long' is a keyword in JSON but ok in Python
    merch_lat: float = 41.0
    merch_long: float = 29.0
    city_pop: float = 15000000
    
    # Behavioral features
    card_tx_count: int = 10
    time_since_last_tx: float = 86400
    time_since_last_tx_log: float = 11.0
    card_avg_amt: float = 100.0
    amt_deviation_from_card_avg: float = 0.0
    card_tx_sequence: int = 10
    is_recent_active: int = 1
    amt_rolling_mean_3: float = 100.0
    amt_rolling_std_3: float = 50.0
    
    # Categorical encoded
    category_encoded: int = 0
    gender_encoded: int = 1
    state_encoded: int = 0
    amt_bucket_encoded: int = 2
    distance_bucket_encoded: int = 0
    merchant_freq: float = 0.001


class FraudPrediction(BaseModel):
    probability: float
    prediction: str
    risk_level: str
    threshold: float


def classify_risk(probability: float) -> str:
    """Classify risk level from probability"""
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    elif probability >= 0.2:
        return "LOW"
    return "MINIMAL"


def fallback_prediction(features: FraudFeatures) -> FraudPrediction:
    """Simple rule-based fallback when model is unavailable"""
    score = 0.0
    
    if features.is_night == 1 and features.amt > 1000:
        score += 0.3
    if features.amt_zscore > 3:
        score += 0.3
    if features.distance_km > 500:
        score += 0.2
    if features.card_tx_count == 0:
        score += 0.1
    
    score = min(score, 1.0)
    
    return FraudPrediction(
        probability=round(score, 4),
        prediction="FRAUD" if score >= 0.5 else "LEGITIMATE",
        risk_level=classify_risk(score),
        threshold=0.5
    )


@app.post("/predict", response_model=FraudPrediction)
async def predict(features: FraudFeatures) -> FraudPrediction:
    """
    Predict fraud probability for a transaction.
    
    Returns:
        FraudPrediction with probability, prediction label, and risk level
    """
    if model is None:
        logger.warning("Model not loaded, using fallback rules")
        return fallback_prediction(features)
    
    try:
        # Convert features to DataFrame
        feature_dict = features.model_dump()
        
        # Rename 'long' to match model expectations (if needed)
        # The model was trained with 'long' as column name
        
        df = pd.DataFrame([feature_dict])
        
        # Get probability
        proba = model.predict_proba(df)[:, 1][0]
        proba = float(proba)
        
        # Optimal threshold from training
        threshold = 0.2716  # From model training
        
        prediction = "FRAUD" if proba >= threshold else "LEGITIMATE"
        risk_level = classify_risk(proba)
        
        logger.info(f"Prediction: amt={features.amt}, proba={proba:.4f}, decision={prediction}")
        
        return FraudPrediction(
            probability=round(proba, 4),
            prediction=prediction,
            risk_level=risk_level,
            threshold=threshold
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Fall back to rules on error
        return fallback_prediction(features)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
