"""
==============================================================
MedVision-AI

Module:
visualization.py

Description:
Visualization utilities for Grad-CAM explanations.

This module provides reusable functions for:

- Applying color maps to Grad-CAM heatmaps
- Overlaying Grad-CAM heatmaps on chest X-ray images
- Displaying Grad-CAM visualizations
- Saving Grad-CAM figures

These utilities are designed for use in notebooks,
Streamlit applications, and FastAPI deployments.

Author:
Dilleswara Rao Intenaka

Project:
MedVision-AI
==============================================================
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    normalize_heatmap,
    resize_heatmap,
    validate_heatmap,
    validate_image,
)

__all__ = [
    "apply_colormap",
    "overlay_heatmap",
    "visualize_gradcam",
    "save_visualization",
]


# ==========================================================
# Apply Color Map
# ==========================================================

def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Apply an OpenCV color map to a normalized heatmap.

    Parameters
    ----------
    heatmap : np.ndarray
        Normalized heatmap with values in [0, 1].

    colormap : int, default=cv2.COLORMAP_JET
        OpenCV colormap.

    Returns
    -------
    np.ndarray
        RGB heatmap.
    """

    validate_heatmap(heatmap)

    heatmap_uint8 = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        colormap,
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    return colored_heatmap


# ==========================================================
# Overlay Heatmap
# ==========================================================

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an image.

    Parameters
    ----------
    image : np.ndarray
        Original image (RGB or grayscale).

    heatmap : np.ndarray
        Grad-CAM heatmap.

    alpha : float, default=0.4
        Heatmap transparency.

    colormap : int
        OpenCV color map.

    Returns
    -------
    np.ndarray
        Overlay image.
    """

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    validate_image(image)

    heatmap = normalize_heatmap(heatmap)

    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )

    if heatmap.shape != image.shape[:2]:
        heatmap = resize_heatmap(
            heatmap,
            (
                image.shape[1],
                image.shape[0],
            ),
        )

    colored_heatmap = apply_colormap(
        heatmap,
        colormap,
    )

    image = image.astype(np.float32)
    colored_heatmap = colored_heatmap.astype(
        np.float32
    )

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    return overlay


# ==========================================================
# Visualize Grad-CAM
# ==========================================================

def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    figsize: tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Display the original image,
    Grad-CAM heatmap,
    and overlay.

    Parameters
    ----------
    image : np.ndarray
        Original image.

    heatmap : np.ndarray
        Grad-CAM heatmap.

    alpha : float
        Overlay transparency.

    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    validate_image(image)

    heatmap = normalize_heatmap(
        heatmap,
    )

    colored_heatmap = apply_colormap(
        heatmap,
    )

    overlay = overlay_heatmap(
        image=image,
        heatmap=heatmap,
        alpha=alpha,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
    )

    fig.suptitle(
        "Grad-CAM Visualization",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].imshow(
        image,
        cmap="gray" if image.ndim == 2 else None,
    )
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(
        colored_heatmap,
    )
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(
        overlay,
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    return fig


# ==========================================================
# Save Visualization
# ==========================================================

def save_visualization(
    figure: plt.Figure,
    filepath: str | Path,
    dpi: int = 300,
    close: bool = False,
) -> None:
    """
    Save a Grad-CAM visualization.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure returned by visualize_gradcam().

    filepath : str or Path
        Output image path.

    dpi : int, default=300
        Image resolution.

    close : bool, default=False
        Close the figure after saving.
    """

    if figure is None:
        raise ValueError(
            "Figure cannot be None."
        )

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight",
    )

    if close:
        plt.close(figure)
