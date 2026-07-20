"""
=====================================================
MedVision-AI

Module:
metrics.py

Description:
Utility functions and helper classes for computing
training and validation metrics.

This module currently provides:
    - Batch classification accuracy calculation
    - Running average tracker (AverageMeter)

Author:
Dilleswara Rao Intenaka
=====================================================
"""

# =====================================
# Standard Library Imports
# =====================================

from __future__ import annotations

# =====================================
# Third-Party Imports
# =====================================

import torch


# =====================================
# Metric Functions
# =====================================

def calculate_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Calculate batch classification accuracy.

    Parameters
    ----------
    outputs : torch.Tensor
        Raw model outputs (logits) of shape (batch_size, num_classes).

    labels : torch.Tensor
        Ground-truth class labels of shape (batch_size,).

    Returns
    -------
    float
        Classification accuracy as a percentage.

    Example
    -------
    >>> outputs = torch.tensor([[2.1, 0.3],
    ...                         [0.2, 1.8]])
    >>> labels = torch.tensor([0, 1])
    >>> calculate_accuracy(outputs, labels)
    100.0
    """

    if outputs.ndim != 2:
        raise ValueError(
            f"'outputs' must have shape (batch_size, num_classes). "
            f"Received shape {tuple(outputs.shape)}."
        )

    if labels.ndim != 1:
        raise ValueError(
            f"'labels' must have shape (batch_size,). "
            f"Received shape {tuple(labels.shape)}."
        )

    if outputs.size(0) != labels.size(0):
        raise ValueError(
            "Batch size mismatch between outputs and labels."
        )

    predictions = outputs.argmax(dim=1)

    correct_predictions = (predictions == labels).sum().item()

    accuracy = (correct_predictions / labels.size(0)) * 100.0

    return accuracy


# =====================================
# Helper Classes
# =====================================

class AverageMeter:
    """
    Track and compute the running average of a metric.

    This class is commonly used to track metrics such as:

    - Training loss
    - Validation loss
    - Accuracy
    - Learning rate

    Attributes
    ----------
    value : float
        Most recent value.

    average : float
        Running average.

    total : float
        Accumulated weighted sum.

    count : int
        Number of observed samples.

    Example
    -------
    >>> meter = AverageMeter()
    >>> meter.update(0.7, n=32)
    >>> meter.update(0.5, n=32)
    >>> meter.average
    0.6
    """

    def __init__(self) -> None:
        """Initialize the metric tracker."""
        self.reset()

    def reset(self) -> None:
        """
        Reset all tracked statistics.
        """

        self.value = 0.0
        self.total = 0.0
        self.count = 0
        self.average = 0.0

    def update(
        self,
        value: float,
        n: int = 1,
    ) -> None:
        """
        Update running statistics.

        Parameters
        ----------
        value : float
            Metric value for the current batch.

        n : int, default=1
            Number of samples represented by the value.
        """

        self.value = float(value)

        self.total += value * n

        self.count += n

        self.average = self.total / self.count

    def __str__(self) -> str:
        """
        String representation of the current average.
        """

        return f"{self.average:.4f}"

    @property
    def avg(self) -> float:
        """
        Alias for the running average.

        This property is provided for compatibility with
        common PyTorch training implementations.
        """
        return self.average
