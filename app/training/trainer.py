"""
=====================================================
MedVision-AI

Module:
trainer.py

Description:
Training utilities for one complete epoch.

This module contains reusable functions for training
deep learning models using PyTorch.

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
from torch.optim import Optimizer
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
# Training Function
# =====================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    dataloader : DataLoader
        Training DataLoader.

    criterion : nn.Module
        Loss function.

    optimizer : Optimizer
        PyTorch optimizer.

    device : torch.device
        Device used for training.

    Returns
    -------
    tuple(float, float)

        Average training loss.

        Average training accuracy.
    """

    # ---------------------------------
    # Set model to training mode
    # ---------------------------------

    model.train()

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
        desc="Training",
        leave=False,
    )

    # ---------------------------------
    # Iterate over mini-batches
    # ---------------------------------

    for images, labels in progress_bar:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = images.size(0)

        # -----------------------------
        # Reset Gradients
        # -----------------------------

        optimizer.zero_grad(set_to_none=True)

        # -----------------------------
        # Forward Pass
        # -----------------------------

        outputs = model(images)

        loss = criterion(outputs, labels)

        # -----------------------------
        # Backpropagation
        # -----------------------------

        loss.backward()

        # -----------------------------
        # Parameter Update
        # -----------------------------

        optimizer.step()

        # -----------------------------
        # Metrics
        # -----------------------------

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

        # -----------------------------
        # Update Progress Bar
        # -----------------------------

        progress_bar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.2f}%",
        )

    return loss_meter.avg, acc_meter.avg
