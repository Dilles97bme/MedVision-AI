"""
==============================================================
MedVision-AI
Model Evaluator Module
==============================================================

This module performs inference on a trained classification model
using a PyTorch DataLoader.

Responsibilities
----------------
- Run model inference
- Collect ground-truth labels
- Collect predicted labels
- Collect prediction probabilities

The returned outputs are intended to be consumed by:

- metrics.py
- report.py
- visualization.py


"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.

    dataloader : DataLoader
        Test DataLoader.

    device : torch.device
        Device used for inference.

    Returns
    -------
    dict
        Dictionary containing:

        labels
            Ground-truth labels.

        predictions
            Predicted class labels.

        probabilities
            Probability of the positive class.

        logits
            Raw model outputs.

        accuracy
            Overall classification accuracy.
    """

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_logits = []

    correct = 0
    total = 0

    for images, labels in tqdm(
        dataloader,
        desc="Evaluating",
        leave=False,
    ):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

        all_logits.extend(
            outputs.cpu().numpy()
        )

    accuracy = correct / total

    return {
        "labels": np.asarray(all_labels),
        "predictions": np.asarray(all_predictions),
        "probabilities": np.asarray(all_probabilities),
        "logits": np.asarray(all_logits),
        "accuracy": accuracy,
    }
