"""
==============================================================
MedVision-AI

Module:
predictor.py

Description:
Prediction utilities for EfficientNet-B0 inference.

Responsibilities
----------------
- Load an image
- Apply preprocessing
- Perform model inference
- Return predicted class
- Return confidence score
- Return class probabilities

Author:
Dilleswara Rao Intenaka
==============================================================
"""

from __future__ import annotations

# ==========================================================
# Standard Library Imports
# ==========================================================

from pathlib import Path
from typing import Union

# ==========================================================
# Third-Party Imports
# ==========================================================

import torch
from PIL import Image

from app.preprocessing.transforms import val_transform


# ==========================================================
# Class Labels
# ==========================================================

CLASS_NAMES: dict[int, str] = {
    0: "NORMAL",
    1: "PNEUMONIA",
}


# ==========================================================
# Image Loader
# ==========================================================

def load_image(
    image_source: Union[str, Path, Image.Image],
) -> Image.Image:
    """
    Load an image from either a file path or a PIL image.

    Parameters
    ----------
    image_source : str | Path | PIL.Image.Image
        Input image source.

    Returns
    -------
    PIL.Image.Image
        RGB image.
    """

    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    return Image.open(image_source).convert("RGB")


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(
    image: Image.Image,
) -> torch.Tensor:
    """
    Apply validation transforms to an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor with shape (1, C, H, W).
    """

    tensor = val_transform(image)

    return tensor.unsqueeze(0)


# ==========================================================
# Prediction
# ==========================================================

@torch.no_grad()
def predict_image(
    image_source: Union[str, Path, Image.Image],
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """
    Predict the class of a chest X-ray image.

    Parameters
    ----------
    image_source : str | Path | PIL.Image.Image
        Input image source.

    model : torch.nn.Module
        Trained EfficientNet-B0 model.

    device : torch.device
        Computation device.

    Returns
    -------
    dict
        Prediction results containing

        - image_path
        - prediction
        - class_index
        - confidence
        - probabilities
    """

    # ------------------------------------------------------
    # Load Image
    # ------------------------------------------------------

    image = load_image(image_source)

    # ------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------

    tensor = preprocess_image(image).to(device)

    # ------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------

    outputs = model(tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    # ------------------------------------------------------
    # Predicted Class
    # ------------------------------------------------------

    confidence, prediction = torch.max(
        probabilities,
        dim=1,
    )

    predicted_index = prediction.item()

    # ------------------------------------------------------
    # Class Probabilities
    # ------------------------------------------------------

    probs = probabilities.squeeze(0).cpu().tolist()

    probability_dict = {
        CLASS_NAMES[idx]: float(prob)
        for idx, prob in enumerate(probs)
    }

    # ------------------------------------------------------
    # Image Name
    # ------------------------------------------------------

    if isinstance(image_source, Image.Image):
        image_name = "Uploaded Image"
    else:
        image_name = str(image_source)

    # ------------------------------------------------------
    # Return Prediction
    # ------------------------------------------------------

    return {
        "image_path": image_name,
        "prediction": CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "confidence": float(confidence.item()),
        "probabilities": probability_dict,
    }
