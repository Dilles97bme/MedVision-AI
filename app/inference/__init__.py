"""
Inference Package
"""

from .loader import get_device, load_model
from .predictor import predict_image

__all__ = [
    "get_device",
    "load_model",
    "predict_image",
]
