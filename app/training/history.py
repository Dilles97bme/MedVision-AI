"""
=====================================================
MedVision-AI

Module:
history.py

Description:
Training history manager for storing, retrieving,
saving, and loading training metrics.

Author:
Dilleswara Rao Intenaka
=====================================================
"""

# =====================================
# Standard Library Imports
# =====================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# =====================================
# Third-Party Imports
# =====================================

import pandas as pd


# =====================================
# Training History
# =====================================

class TrainingHistory:
    """
    Store and manage training history.

    Metrics tracked:
    - Training Loss
    - Validation Loss
    - Training Accuracy
    - Validation Accuracy
    """

    def __init__(self) -> None:

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    # ---------------------------------
    # Update History
    # ---------------------------------

    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
    ) -> None:
        """
        Store metrics for one epoch.
        """

        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["train_accuracy"].append(float(train_accuracy))
        self.history["val_accuracy"].append(float(val_accuracy))

    # ---------------------------------
    # Get History
    # ---------------------------------

    def get_history(self) -> Dict[str, List[float]]:
        """
        Return complete history dictionary.
        """

        return self.history

    # ---------------------------------
    # Convert to DataFrame
    # ---------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert history to a pandas DataFrame.
        """

        return pd.DataFrame(self.history)

    # ---------------------------------
    # Save History
    # ---------------------------------

    def save(
        self,
        filepath: str | Path,
    ) -> None:
        """
        Save history as a JSON file.
        """

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(
                self.history,
                file,
                indent=4,
            )

    # ---------------------------------
    # Load History
    # ---------------------------------

    def load(
        self,
        filepath: str | Path,
    ) -> None:
        """
        Load history from JSON.
        """

        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as file:

            self.history = json.load(file)

    def plot(self) -> None:
        """
        Plot training and validation metrics.
        """

        import matplotlib.pyplot as plt

        df = self.to_dataframe()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss
        axes[0].plot(df["train_loss"], label="Train")
        axes[0].plot(df["val_loss"], label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(df["train_accuracy"], label="Train")
        axes[1].plot(df["val_accuracy"], label="Validation")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
    # ---------------------------------
    # Best Validation Accuracy
    # ---------------------------------

    @property
    def best_validation_accuracy(self) -> float:
        """
        Return the best validation accuracy.

        Returns
        -------
        float
        Highest recorded validation accuracy.
        Returns 0.0 if no history is available.
        """

        if not self.history["val_accuracy"]:
            return 0.0

        return max(self.history["val_accuracy"])

    # ---------------------------------
    # Best Validation Loss
    # ---------------------------------
    @property
    def best_validation_loss(self) -> float:
        """
        Return the minimum validation loss.

        Returns
        -------
        float
            Lowest recorded validation loss.
            Returns infinity if no history is available.
        """

        if not self.history["val_loss"]:
            return float("inf")

        return min(self.history["val_loss"])

    # ---------------------------------
    # Number of Epochs
    # ---------------------------------

    @property
    def epochs(self) -> int:
        """
        Number of completed epochs.
        """

        return len(self.history["train_loss"])

    # ---------------------------------
    # Reset History
    # ---------------------------------

    def reset(self) -> None:
        """
        Clear all stored metrics.
        """

        for key in self.history:
            self.history[key].clear()

    # ---------------------------------
    # String Representation
    # ---------------------------------

    def __repr__(self) -> str:

        return (
            f"TrainingHistory("
            f"epochs={self.epochs}, "
            f"best_val_loss={self.best_validation_loss:.4f}, "
            f"best_val_accuracy={self.best_validation_accuracy:.2f}%)"
        )
