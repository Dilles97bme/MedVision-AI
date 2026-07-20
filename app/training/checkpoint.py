"""
=====================================================
MedVision-AI

Module:
checkpoint.py

Description:
Utility functions for saving and loading training
checkpoints.

=====================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


# =====================================================
# Save Checkpoint
# =====================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    val_loss: float,
    filepath: str | Path,
    scheduler: Any | None = None,
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    model : nn.Module
        Trained model.

    optimizer : Optimizer
        Optimizer.

    epoch : int
        Current epoch.

    val_loss : float
        Validation loss.

    filepath : str | Path
        Output checkpoint path.

    scheduler : optional
        Learning-rate scheduler.
    """

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)


# =====================================================
# Load Checkpoint
# =====================================================

def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict:
    """
    Load a checkpoint.

    Parameters
    ----------
    filepath : str | Path
        Path to checkpoint.

    model : nn.Module
        Model instance.

    optimizer : Optimizer, optional
        Optimizer to restore.

    scheduler : optional
        Scheduler to restore.

    Returns
    -------
    dict
        Loaded checkpoint dictionary.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {filepath}"
        )

    checkpoint = torch.load(
        filepath,
        map_location="cpu",
    )

    # Load model weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Load optimizer state (training only)
    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    # Load scheduler state (training only)
    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint
