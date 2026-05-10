from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import predict  # noqa: E402

app = FastAPI(title="Fraud Detection API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: PredictionRequest):
    try:
        result = predict(request.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
