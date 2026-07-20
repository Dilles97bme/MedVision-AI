"""
=====================================================
MedVision-AI

Module:
early_stopping.py

Description:
Early stopping utility to terminate model training
when validation performance stops improving.

Author:
Dilleswara Rao Intenaka
=====================================================
"""

# =====================================
# Standard Library Imports
# =====================================

from __future__ import annotations


# =====================================
# Early Stopping Class
# =====================================

class EarlyStopping:
    """
    Early stopping utility.

    Stops training when the monitored validation loss
    has not improved for a specified number of epochs.

    Parameters
    ----------
    patience : int, default=5
        Number of consecutive epochs without improvement
        before stopping training.

    min_delta : float, default=0.0
        Minimum improvement required to reset the counter.

    Attributes
    ----------
    patience : int
        Maximum allowed epochs without improvement.

    min_delta : float
        Minimum improvement threshold.

    best_loss : float
        Best validation loss observed.

    counter : int
        Number of consecutive epochs without improvement.

    early_stop : bool
        Indicates whether training should stop.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

        self.early_stop = False

    def __call__(
        self,
        val_loss: float,
    ) -> bool:
        """
        Update the early stopping state.

        Parameters
        ----------
        val_loss : float
            Current validation loss.

        Returns
        -------
        bool
            True if training should stop,
            otherwise False.
        """

        # ---------------------------------
        # Validation loss improved
        # ---------------------------------

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss

            self.counter = 0

            self.early_stop = False

        # ---------------------------------
        # No improvement
        # ---------------------------------

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self) -> None:
        """
        Reset the early stopping state.
        """

        self.best_loss = float("inf")

        self.counter = 0

        self.early_stop = False

    def state_dict(self) -> dict:
        """
        Return the current state.
        """

        return {
            "best_loss": self.best_loss,
            "counter": self.counter,
            "early_stop": self.early_stop,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        """
        Restore the saved state.

        Parameters
        ----------
        state : dict
            Previously saved state dictionary.
        """

        self.best_loss = state["best_loss"]
        self.counter = state["counter"]
        self.early_stop = state["early_stop"]
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]

    def __repr__(self) -> str:
        """
        String representation.
        """

        return (
            f"EarlyStopping("
            f"patience={self.patience}, "
            f"min_delta={self.min_delta}, "
            f"best_loss={self.best_loss:.6f}, "
            f"counter={self.counter}, "
            f"early_stop={self.early_stop})"
        )
