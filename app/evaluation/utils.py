"""
==============================================================
MedVision-AI
Evaluation Utility Functions
==============================================================
"""

from __future__ import annotations

import numpy as np
import torch

# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Class names
CLASS_NAMES = (
    "Normal",
    "Pneumonia",
)


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized image tensor into a displayable NumPy image.

    Parameters
    ----------
    image : torch.Tensor
        Tensor with shape (C, H, W).

    Returns
    -------
    np.ndarray
        Image with shape (H, W, C).
    """

    image = image.detach().cpu().numpy()

    image = image.transpose(1, 2, 0)

    image = image * IMAGENET_STD + IMAGENET_MEAN

    image = np.clip(image, 0, 1)

    return image
