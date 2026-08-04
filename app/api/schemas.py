"""
==============================================================
Pydantic Schemas
==============================================================
"""

from typing import Dict

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    image_path: str
    prediction: str
    class_index: int
    confidence: float
    probabilities: Dict[str, float]


class AnalyzeResponse(BaseModel):
    prediction: PredictionResponse
    report: dict
    pdf: str
    original_image: str
    gradcam_image: str
