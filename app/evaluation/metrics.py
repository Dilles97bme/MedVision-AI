"""
==============================================================
MedVision-AI
Evaluation Metrics Module
==============================================================

This module provides reusable functions for evaluating
classification performance of trained deep learning models.

Metrics Supported
-----------------
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score

The primary function is:

    compute_metrics(...)

which returns all evaluation metrics in a dictionary.

"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class labels.

    y_pred : np.ndarray
        Predicted class labels.

    y_prob : np.ndarray, optional
        Predicted probability for the positive class.
        Required only for ROC-AUC computation.

    Returns
    -------
    dict
        Dictionary containing:

        - accuracy
        - precision
        - recall
        - f1_score
        - auc (None if probabilities unavailable)
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    if y_prob is not None:
        metrics["auc"] = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        metrics["auc"] = None

    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    """
    Convert metric dictionary into a printable string.

    Parameters
    ----------
    metrics : dict
        Dictionary returned by ``compute_metrics()``.

    Returns
    -------
    str
        Nicely formatted evaluation summary.
    """

    lines = [
        "=" * 50,
        "Evaluation Metrics",
        "=" * 50,
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall   : {metrics['recall']:.4f}",
        f"F1 Score : {metrics['f1_score']:.4f}",
    ]

    if metrics["auc"] is not None:
        lines.append(f"AUC      : {metrics['auc']:.4f}")

    return "\n".join(lines)
