import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import SentimentPredictor


app = FastAPI(title="Sentiment Analysis API", version="0.1.0")

_predictor = None


def get_predictor() -> SentimentPredictor:
    """
    Lazily instantiate the predictor so the container can start
    even if the model file is mounted later.
    """
    global _predictor
    if _predictor is None:
        model_path = os.getenv("MODEL_PATH", "model_outputs/sentiment_classifier.joblib")
        try:
            _predictor = SentimentPredictor(model_path=model_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {model_path}. Mount a trained model_outputs volume.",
            ) from exc
    return _predictor


class PredictRequest(BaseModel):
    text: str
    clean_texts: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    predictor = get_predictor()
    result = predictor.predict(request.text, clean_texts=request.clean_texts)[0]
    return result
