"""
==============================================================
MedVision-AI

Module:
utils.py

Description:
Shared utility functions for explainability modules.

This module contains reusable helper functions used by
Grad-CAM and future explainability techniques.

Author:
Dilleswara Rao Intenaka

Project:
MedVision-AI
==============================================================
"""

from __future__ import annotations

import cv2
import numpy as np


__all__ = [
    "normalize_heatmap",
    "resize_heatmap",
    "confidence_to_percentage",
    "validate_heatmap",
    "validate_image",
]


# ==========================================================
# Heatmap Utilities
# ==========================================================

def normalize_heatmap(
    heatmap: np.ndarray,
) -> np.ndarray:
    """
    Normalize a heatmap to the range [0, 1].

    Parameters
    ----------
    heatmap : np.ndarray
        Input heatmap.

    Returns
    -------
    np.ndarray
        Normalized heatmap.
    """

    validate_heatmap(heatmap)

    heatmap = heatmap.astype(np.float32)

    heatmap -= heatmap.min()

    heatmap /= heatmap.max() + 1e-8

    return heatmap


def resize_heatmap(
    heatmap: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Resize a heatmap.

    Parameters
    ----------
    heatmap : np.ndarray
        Input heatmap.

    size : tuple[int, int]
        Desired output size (width, height).

    Returns
    -------
    np.ndarray
        Resized heatmap.
    """

    validate_heatmap(heatmap)

    return cv2.resize(
        heatmap,
        size,
        interpolation=cv2.INTER_LINEAR,
    )


# ==========================================================
# Confidence Utilities
# ==========================================================

def confidence_to_percentage(
    confidence: float,
    decimals: int = 2,
) -> str:
    """
    Convert confidence score to percentage.

    Parameters
    ----------
    confidence : float
        Confidence between 0 and 1.

    decimals : int, default=2
        Number of decimal places.

    Returns
    -------
    str
        Formatted percentage string.
    """

    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    return f"{confidence:.{decimals}%}"


# ==========================================================
# Validation Utilities
# ==========================================================

def validate_heatmap(
    heatmap: np.ndarray,
) -> None:
    """
    Validate a Grad-CAM heatmap.

    Parameters
    ----------
    heatmap : np.ndarray
        Heatmap array.

    Raises
    ------
    TypeError
        If input is not a NumPy array.

    ValueError
        If heatmap is invalid.
    """

    if not isinstance(heatmap, np.ndarray):
        raise TypeError(
            "Heatmap must be a NumPy array."
        )

    if heatmap.size == 0:
        raise ValueError(
            "Heatmap cannot be empty."
        )

    if heatmap.ndim != 2:
        raise ValueError(
            "Heatmap must be a 2D array."
        )


def validate_image(
    image: np.ndarray,
) -> None:
    """
    Validate an input image.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Raises
    ------
    TypeError
        If input is not a NumPy array.

    ValueError
        If image format is invalid.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Image cannot be empty."
        )

    # Grayscale
    if image.ndim == 2:
        return

    # RGB
    if image.ndim == 3 and image.shape[2] == 3:
        return

    raise ValueError(
        "Image must be either grayscale "
        "(H, W) or RGB (H, W, 3)."
    )
