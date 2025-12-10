"""
FastAPI Sentiment Analysis API

Provides endpoints for sentiment prediction using the trained model.
"""

import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None


class TextInput(BaseModel):
    """Single text input for prediction."""
    text: str = Field(..., min_length=1, description="Text to analyze")
    clean: bool = Field(default=True, description="Whether to clean the text before prediction")


class BatchTextInput(BaseModel):
    """Batch text input for prediction."""
    texts: List[str] = Field(..., min_items=1, description="List of texts to analyze")
    clean: bool = Field(default=True, description="Whether to clean texts before prediction")


class PredictionResult(BaseModel):
    """Prediction result for a single text."""
    text: str
    sentiment: str
    confidence: float
    probabilities: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup."""
    global predictor
    
    model_path = os.environ.get(
        "MODEL_PATH", 
        "model_outputs/sentiment_classifier.joblib"
    )
    
    try:
        # Only load model if it exists
        if os.path.exists(model_path):
            from inference import SentimentPredictor
            predictor = SentimentPredictor(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. API will run without prediction capability.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text using a fine-tuned BERT model",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/predict", response_model=PredictionResult)
async def predict_single(input_data: TextInput):
    """
    Predict sentiment for a single text.
    
    Args:
        input_data: Text input with optional cleaning flag
        
    Returns:
        Prediction result with sentiment and confidence
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure a trained model is available."
        )
    
    try:
        results = predictor.predict(input_data.text, clean_texts=input_data.clean)
        return PredictionResult(**results[0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResult])
async def predict_batch(input_data: BatchTextInput):
    """
    Predict sentiment for multiple texts.
    
    Args:
        input_data: Batch of texts with optional cleaning flag
        
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure a trained model is available."
        )
    
    try:
        results = predictor.predict(input_data.texts, clean_texts=input_data.clean)
        return [PredictionResult(**r) for r in results]
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
