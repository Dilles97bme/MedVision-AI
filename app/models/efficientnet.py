"""
=====================================================
MedVision-AI

Module:
efficientnet.py

Description:
Utility functions for building EfficientNet models
used throughout the MedVision-AI project.

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

import torch.nn as nn

from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
)

# =====================================
# Build EfficientNet-B0
# =====================================


def build_efficientnet_b0(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Build an EfficientNet-B0 model for image classification.

    Parameters
    ----------
    num_classes : int, default=2
        Number of output classes.

    pretrained : bool, default=True
        If True, load ImageNet pretrained weights.

    freeze_backbone : bool, default=True
        If True, freeze all feature extraction layers.

    dropout : float, default=0.2
        Dropout probability before the classifier.

    Returns
    -------
    nn.Module
        Configured EfficientNet-B0 model.
    """

    # ---------------------------------
    # Load pretrained weights
    # ---------------------------------

    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.efficientnet_b0(
        weights=weights,
    )

    # ---------------------------------
    # Replace classifier
    # ---------------------------------

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(
            in_features,
            num_classes,
        ),
    )

    # ---------------------------------
    # Freeze feature extractor
    # ---------------------------------

    if freeze_backbone:

        for parameter in model.features.parameters():
            parameter.requires_grad = False

    return model


# =====================================
# Count Trainable Parameters
# =====================================

def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count the number of trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
        Number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =====================================
# Count Total Parameters
# =====================================

def count_total_parameters(
    model: nn.Module,
) -> int:
    """
    Count total model parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
        Total number of parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =====================================
# Model Summary
# =====================================

def print_model_summary(
    model: nn.Module,
) -> None:
    """
    Print a concise summary of the model.
    """

    total = count_total_parameters(model)
    trainable = count_trainable_parameters(model)

    print("=" * 55)
    print("EfficientNet-B0 Summary")
    print("=" * 55)
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print(f"Frozen Parameters    : {total-trainable:,}")
    print("=" * 55)
