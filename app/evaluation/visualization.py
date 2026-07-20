"""
==============================================================
MedVision-AI
Evaluation Visualization Module
==============================================================

This module provides visualization utilities for evaluating
classification models.

Visualizations
--------------
- Confusion Matrix
- ROC Curve
- Sample Predictions


"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


DEFAULT_CLASS_NAMES = [
    "Normal",
    "Pneumonia",
]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    figsize: tuple[int, int] = (6, 6),
):
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Predicted labels.

    class_names : Sequence[str], default=("Normal","Pneumonia")
        Display names for classes.

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(figsize=figsize)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: tuple[int, int] = (6, 6),
):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_prob : np.ndarray
        Probability of positive class.

    Returns
    -------
    matplotlib.axes.Axes
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob,
    )

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.4f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()

    return ax


def show_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    num_images: int = 8,
):
    """
    Display sample predictions.

    Parameters
    ----------
    images : np.ndarray
        Images in HWC format.

    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Predicted labels.

    probabilities : np.ndarray, optional
        Positive-class probabilities.

    class_names : Sequence[str]
        Class names.

    num_images : int
        Number of images to display.
    """

    num_images = min(num_images, len(images))

    cols = 4
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, rows * 3.5),
    )

    axes = np.array(axes).reshape(-1)

    for i in range(num_images):

        axes[i].imshow(images[i], cmap="gray")

        title = (
            f"True : {class_names[y_true[i]]}\n"
            f"Pred : {class_names[y_pred[i]]}"
        )

        if probabilities is not None:
            title += f"\nConf : {probabilities[i]:.2%}"

        axes[i].set_title(
            title,
            fontsize=9,
        )

        axes[i].axis("off")

    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    return axes
