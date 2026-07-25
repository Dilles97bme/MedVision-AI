"""
==============================================================
Model Dependencies

Loads the trained model once and shares it across all API routes.
==============================================================
"""

from pathlib import Path

from app.inference.loader import load_model

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")


model, device = load_model(
    checkpoint_path=CHECKPOINT_PATH,
    freeze_backbone=False,
)
