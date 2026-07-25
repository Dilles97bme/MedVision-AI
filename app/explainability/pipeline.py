"""
==============================================================
MedVision-AI

Module:
pipeline.py

Description:
High-level Grad-CAM pipeline.

This module orchestrates the explainability workflow by:

- Preprocessing an input image
- Generating the Grad-CAM heatmap
- Creating the overlay visualization

This pipeline is intended for reuse in:

- Streamlit
- FastAPI
- Jupyter notebooks
- Testing scripts

Author:
Dilleswara Rao Intenaka
==============================================================
"""

from __future__ import annotations

# ==========================================================
# Standard Library Imports
# ==========================================================

from typing import Any

# ==========================================================
# Third-Party Imports
# ==========================================================

import numpy as np
import torch
from PIL import Image

# ==========================================================
# Local Imports
# ==========================================================

from app.inference.predictor import preprocess_image

from .gradcam import GradCAM
from .visualization import overlay_heatmap


# ==========================================================
# Grad-CAM Pipeline
# ==========================================================

def generate_gradcam(
    image: Image.Image,
    model: torch.nn.Module,
    alpha: float = 0.4,
    target_class: int | None = None,
) -> dict[str, Any]:
    """
    Generate a Grad-CAM explanation.

    Parameters
    ----------
    image : PIL.Image.Image
        Original input image.

    model : torch.nn.Module
        Trained CNN model.

    alpha : float, default=0.4
        Overlay transparency.

    target_class : int, optional
        Class index for which Grad-CAM should be generated.
        If None, the predicted class is used.

    Returns
    -------
    dict

        {
            "heatmap": np.ndarray,
            "overlay": np.ndarray,
            "image": np.ndarray,
        }
    """

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = next(model.parameters()).device

    # ------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------

    tensor = preprocess_image(image).to(device)

    # ------------------------------------------------------
    # Original image
    # ------------------------------------------------------

    original = np.array(
        image.convert("RGB")
    )

    # ------------------------------------------------------
    # Target layer
    # ------------------------------------------------------

    target_layer = model.features[-1]

    # ------------------------------------------------------
    # Generate Grad-CAM
    # ------------------------------------------------------

    with GradCAM(
        model=model,
        target_layer=target_layer,
    ) as gradcam:

        heatmap = gradcam.generate(
            image_tensor=tensor,
            target_class=target_class,
        )

    # ------------------------------------------------------
    # Overlay
    # ------------------------------------------------------

    overlay = overlay_heatmap(
        image=original,
        heatmap=heatmap,
        alpha=alpha,
    )

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

   # ------------------------------------------------------
    # Convert NumPy arrays to PIL Images
    # ------------------------------------------------------

    original_pil = Image.fromarray(original)
    overlay_pil = Image.fromarray(overlay)

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    return {
        "image": original_pil,
        "heatmap": heatmap,
        "overlay": overlay_pil,
    }
