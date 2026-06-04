from pydantic import BaseModel
from typing import Dict, Any


class PredictRequest(BaseModel):
    features: Dict[str, float]


class PredictResponse(BaseModel):
    probability: float
    prediction: int
    details: Dict[str, Any] = {}
