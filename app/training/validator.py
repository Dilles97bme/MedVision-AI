
"""
=====================================================
MedVision-AI

Module:
validator.py

Description:
Validation utilities for one complete epoch.

This module evaluates a trained PyTorch model on a
validation dataset without updating model parameters.

Author:
Dilleswara Rao Intenaka
=====================================================
"""

# =====================================
# Standard Library Imports
# =====================================

from __future__ import annotations

from typing import Tuple

# =====================================
# Third-Party Imports
# =====================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# =====================================
# Local Imports
# =====================================

from app.training.metrics import (
    AverageMeter,
    calculate_accuracy,
)

# =====================================
# Validation Function
# =====================================


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model for one validation epoch.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    dataloader : DataLoader
        Validation DataLoader.

    criterion : nn.Module
        Loss function.

    device : torch.device
        Device used for validation.

    Returns
    -------
    tuple(float, float)
        Average validation loss.

        Average validation accuracy.
    """

    # ---------------------------------
    # Set model to evaluation mode
    # ---------------------------------

    model.eval()

    # ---------------------------------
    # Metric trackers
    # ---------------------------------

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # ---------------------------------
    # Progress Bar
    # ---------------------------------

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    # ---------------------------------
    # Disable Gradient Computation
    # ---------------------------------

    with torch.no_grad():

        # -----------------------------
        # Iterate over mini-batches
        # -----------------------------

        for images, labels in progress_bar:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            batch_size = images.size(0)

            # -------------------------
            # Forward Pass
            # -------------------------

            outputs = model(images)

            loss = criterion(outputs, labels)

            # -------------------------
            # Metrics
            # -------------------------

            accuracy = calculate_accuracy(
                outputs,
                labels,
            )

            loss_meter.update(
                loss.item(),
                batch_size,
            )

            acc_meter.update(
                accuracy,
                batch_size,
            )

            # -------------------------
            # Update Progress Bar
            # -------------------------

            progress_bar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{acc_meter.avg:.2f}%",
            )

    return loss_meter.avg, acc_meter.avg
