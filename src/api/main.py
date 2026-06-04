import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow.sklearn
from src.api.pydantic_models import PredictRequest, PredictResponse

app = FastAPI(title="Credit Risk Scoring API")

model = None


def _find_model_path():
    uri = os.getenv("MODEL_URI") or os.getenv("MODEL_PATH")
    if uri:
        return uri
    if os.path.exists("models"):
        entries = sorted(os.listdir("models"))
        if entries:
            return os.path.join("models", entries[0])
    return None


@app.on_event("startup")
def startup_event():
    global model
    model_path = _find_model_path()
    if not model_path:
        model = None
        return
    try:
        model = mlflow.sklearn.load_model(model_path)
    except Exception:
        model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = req.features
    X = pd.DataFrame([features])
    try:
        probs = model.predict_proba(X)[:, 1]
        prob = float(probs[0])
    except Exception:
        prob = None

    try:
        pred = int(model.predict(X)[0])
    except Exception:
        raise HTTPException(status_code=500, detail="Model prediction failed")

    return PredictResponse(probability=prob if prob is not None else 0.0, prediction=pred, details={"features": features})
