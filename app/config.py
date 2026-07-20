"""
=====================================================
MedVision-AI

Module:
config.py

Description:
Central configuration for the project.
=====================================================
"""

from pathlib import Path
import torch

# --------------------------------------------------
# Dataset
# --------------------------------------------------

DATASET_PATH = Path("dataset")

IMAGE_SIZE = 224

NUM_CLASSES = 2

# --------------------------------------------------
# Training
# --------------------------------------------------

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PIN_MEMORY = True

# --------------------------------------------------
# Early Stopping
# --------------------------------------------------

PATIENCE = 5

# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# Output Directories
# --------------------------------------------------

CHECKPOINT_DIR = Path("checkpoints")

OUTPUT_DIR = Path("outputs")

CHECKPOINT_DIR.mkdir(exist_ok=True)

OUTPUT_DIR.mkdir(exist_ok=True)
