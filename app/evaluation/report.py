"""
==============================================================
MedVision-AI
Evaluation Report Module
==============================================================

This module provides utilities for generating, printing,
and saving model evaluation reports.

Responsibilities
----------------
- Generate classification report
- Print evaluation summary
- Save prediction results to CSV


"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.metrics import classification_report


DEFAULT_CLASS_NAMES = (
    "Normal",
    "Pneumonia",
)


def generate_classification_report(
    y_true,
    y_pred,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> str:
    """
    Generate a classification report.

    Parameters
    ----------
    y_true
        Ground-truth labels.

    y_pred
        Predicted labels.

    class_names
        Display names for each class.

    Returns
    -------
    str
        Formatted classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )


def print_evaluation_summary(
    metrics: dict[str, Any],
) -> None:
    """
    Print evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary returned by compute_metrics().
    """

    print("=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    if metrics["auc"] is not None:
        print(f"AUC      : {metrics['auc']:.4f}")

    print("=" * 60)


def save_predictions_csv(
    filepath: str | Path,
    y_true,
    y_pred,
    probabilities=None,
) -> Path:
    """
    Save prediction results to CSV.

    Parameters
    ----------
    filepath
        Output CSV path.

    y_true
        Ground-truth labels.

    y_pred
        Predicted labels.

    probabilities
        Positive-class probabilities.

    Returns
    -------
    Path
        Saved CSV path.
    """

    filepath = Path(filepath)

    data = {
        "True Label": y_true,
        "Predicted Label": y_pred,
    }

    if probabilities is not None:
        data["Probability"] = probabilities

    df = pd.DataFrame(data)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        filepath,
        index=False,
    )

    return filepath
