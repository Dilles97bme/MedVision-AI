"""
==============================================================
MedVision-AI

Module:
loader.py

Description:
Utility functions for loading trained models used during
inference.

Responsibilities
----------------
- Detect computation device
- Build EfficientNet-B0
- Load trained checkpoint
- Return model in evaluation mode

==============================================================
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from app.models.efficientnet import build_efficientnet_b0


# ==========================================================
# Device
# ==========================================================

def get_device() -> torch.device:
    """
    Return the available computation device.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ==========================================================
# Model Loader
# ==========================================================

def load_model(
    checkpoint_path: str | Path,
    num_classes: int = 2,
    freeze_backbone: bool = True,
) -> tuple[nn.Module, torch.device]:
    """
    Load a trained EfficientNet-B0 model.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the trained model checkpoint.

    num_classes : int, default=2
        Number of output classes.

    Returns
    -------
    tuple
        (model, device)
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    device = get_device()

    model = build_efficientnet_b0(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=freeze_backbone,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # ------------------------------------------------------
    # Support both checkpoint formats
    # ------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    model.to(device)

    model.eval()

    return model, device
