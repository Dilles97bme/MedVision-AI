"""
==============================================================
MedVision-AI

Module:
gradcam.py

Description:
Gradient-weighted Class Activation Mapping (Grad-CAM)
implementation for CNN-based medical image classification.

===============================
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.

    target_layer : nn.Module
        Target convolutional layer.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be an instance of torch.nn.Module."
            )

        if not isinstance(target_layer, nn.Module):
            raise TypeError(
                "target_layer must be an instance of torch.nn.Module."
            )

        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = None

        self._register_hooks()

    # ==========================================================
    # Hook Registration
    # ==========================================================

    def _register_hooks(self) -> None:
        """Register forward hook."""

        self._forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(
        self,
        module: nn.Module,
        inputs,
        output: torch.Tensor,
    ) -> None:
        """
        Store activations and register gradient hook.
        """

        self.activations = output

        def save_gradient(
            grad: torch.Tensor,
        ) -> None:
            self.gradients = grad.detach()

        output.register_hook(save_gradient)

    # ==========================================================
    # Grad-CAM
    # ==========================================================

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input tensor of shape (1, C, H, W).

        target_class : int, optional
            Target class index. If None, the predicted
            class is used.

        Returns
        -------
        np.ndarray
            Normalized Grad-CAM heatmap.
        """

        if image_tensor.ndim != 4:
            raise ValueError(
                "Input tensor must have shape (1, C, H, W)."
            )

        if image_tensor.size(0) != 1:
            raise ValueError(
                "Grad-CAM currently supports batch size 1."
            )

        if not image_tensor.is_floating_point():
            raise TypeError(
                "Input tensor must be floating point."
            )

        self.model.eval()

        self.activations = None
        self.gradients = None

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        with torch.enable_grad():

            outputs = self.model(image_tensor)

            if target_class is None:
                target_class = int(outputs.argmax(dim=1).item())

            if not (0 <= target_class < outputs.shape[1]):
                raise ValueError(
                    f"target_class must be between 0 and "
                    f"{outputs.shape[1] - 1}."
                )

            score = outputs[:, target_class]

            self.model.zero_grad()

            score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Failed to capture activations. "
                "Verify the selected target layer."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Failed to capture gradients. "
                "Verify gradients are enabled."
            )

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = torch.sum(
            weights * activations,
            dim=1,
        )

        cam = F.relu(cam)

        cam = cam.squeeze()

        cam -= cam.min()

        cam /= cam.max() + 1e-8

        cam = cam.detach().cpu().numpy()

        height = image_tensor.shape[-2]
        width = image_tensor.shape[-1]

        cam = cv2.resize(
            cam,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        return cam.astype(np.float32)

    # ==========================================================
    # Cleanup
    # ==========================================================

    def remove_hooks(self) -> None:
        """Remove registered hooks."""

        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.remove_hooks()

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass
