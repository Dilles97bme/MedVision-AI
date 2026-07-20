"""
=====================================================
MedVision-AI

Module:
dataset.py

Description:
Reusable dataset utilities for loading and preparing
the Chest X-ray dataset for PyTorch.

This module provides:

- ChestXRayDataset
- collect_image_paths()
- encode_labels()
- create_datasets()
- get_dataloaders()

Author:
Dilleswara Rao Intenaka
=====================================================
"""

# =====================================
# Standard Library Imports
# =====================================

from pathlib import Path
import torch

# =====================================
# Third-Party Imports
# =====================================

from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# =====================================
# Constants
# =====================================

CLASS_NAMES = (
    "NORMAL",
    "PNEUMONIA",
)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
)

# =====================================
# Dataset Class
# =====================================


class ChestXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for Chest X-ray images.

    Parameters
    ----------
    image_paths : list[Path]
        List of image paths.

    labels : list[int]
        Encoded labels.

    transform : callable, optional
        Image transformation pipeline.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform=None,
    ) -> None:

        if len(image_paths) != len(labels):
            raise ValueError(
                "image_paths and labels must have equal length."
            )

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, int]:
        """
        Retrieve one sample.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[Tensor, int]
            Image tensor and encoded label.
        """

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# =====================================
# Dataset Utility Functions
# =====================================

def collect_image_paths(
    dataset_path: Path,
    split: str,
) -> tuple[list[Path], list[str]]:
    """
    Collect image paths and labels from a dataset split.

    Parameters
    ----------
    dataset_path : Path
        Root dataset directory.

    split : str
        Dataset split.

        Supported values
        ----------------
        train
        val
        test

    Returns
    -------
    tuple[list[Path], list[str]]
        Image paths and corresponding string labels.
    """

    split = split.lower()

    if split not in ("train", "val", "test"):
        raise ValueError(
            f"Invalid split '{split}'. "
            "Choose from train, val or test."
        )

    split_dir = dataset_path / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split not found:\n{split_dir}"
        )

    image_paths: list[Path] = []
    labels: list[str] = []

    for class_name in CLASS_NAMES:

        class_dir = split_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder:\n{class_dir}"
            )

        files = sorted(class_dir.iterdir())

        for file_path in files:

            if (
                file_path.is_file()
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                image_paths.append(file_path)
                labels.append(class_name)

    return image_paths, labels

# =====================================
# Label Encoding
# =====================================


LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


def encode_labels(
    labels: list[str],
) -> list[int]:
    """
    Convert string class labels into integer labels.

    Parameters
    ----------
    labels : list[str]
        List of original class labels.

    Returns
    -------
    list[int]
        Integer-encoded labels.

    Raises
    ------
    ValueError
        If an unknown class label is encountered.

    Examples
    --------
    >>> encode_labels(["NORMAL", "PNEUMONIA"])
    [0, 1]
    """

    encoded_labels: list[int] = []

    for label in labels:

        if label not in LABEL_MAP:
            raise ValueError(
                f"Unknown class label: '{label}'. "
                f"Supported labels are: {list(LABEL_MAP.keys())}"
            )

        encoded_labels.append(LABEL_MAP[label])

    return encoded_labels

# =====================================
# Dataset Creation
# =====================================


def create_datasets(
    dataset_path: Path,
    train_transform,
    val_transform,
    test_transform=None,
) -> tuple[
    ChestXRayDataset,
    ChestXRayDataset,
    ChestXRayDataset,
]:
    """
    Create PyTorch Dataset objects for all dataset splits.

    Parameters
    ----------
    dataset_path : Path
        Root dataset directory.

    train_transform : callable
        Transformations applied to the training set.

    val_transform : callable
        Transformations applied to the validation set.

    test_transform : callable, optional
        Transformations applied to the test set.
        If None, val_transform is used.

    Returns
    -------
    tuple
        train_dataset,
        val_dataset,
        test_dataset
    """

    # ---------------------------------
    # Use validation transform for test
    # if no separate transform is given.
    # ---------------------------------

    if test_transform is None:
        test_transform = val_transform

    # ---------------------------------
    # Training Split
    # ---------------------------------

    train_image_paths, train_labels = collect_image_paths(
        dataset_path=dataset_path,
        split="train",
    )

    train_labels = encode_labels(train_labels)

    train_dataset = ChestXRayDataset(
        image_paths=train_image_paths,
        labels=train_labels,
        transform=train_transform,
    )

    # ---------------------------------
    # Validation Split
    # ---------------------------------

    val_image_paths, val_labels = collect_image_paths(
        dataset_path=dataset_path,
        split="val",
    )

    val_labels = encode_labels(val_labels)

    val_dataset = ChestXRayDataset(
        image_paths=val_image_paths,
        labels=val_labels,
        transform=val_transform,
    )

    # ---------------------------------
    # Test Split
    # ---------------------------------

    test_image_paths, test_labels = collect_image_paths(
        dataset_path=dataset_path,
        split="test",
    )

    test_labels = encode_labels(test_labels)

    test_dataset = ChestXRayDataset(
        image_paths=test_image_paths,
        labels=test_labels,
        transform=test_transform,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )
# =====================================
# DataLoader Creation
# =====================================


def get_dataloaders(
    dataset_path: Path,
    train_transform,
    val_transform,
    test_transform=None,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = torch.cuda.is_available(),
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create PyTorch DataLoader objects.

    Parameters
    ----------
    dataset_path : Path
        Root dataset directory.

    train_transform : callable
        Training image transformations.

    val_transform : callable
        Validation image transformations.

    test_transform : callable, optional
        Test image transformations.
        If None, val_transform is used.

    batch_size : int, default=32
        Number of samples per batch.

    num_workers : int, default=2
        Number of subprocesses for data loading.

    pin_memory : bool, default=True
        Whether to pin memory for faster GPU transfer.

    Returns
    -------
    tuple
        train_loader,
        val_loader,
        test_loader
    """

    train_dataset, val_dataset, test_dataset = create_datasets(
        dataset_path=dataset_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )

# =====================================
# Public API
# =====================================


__all__ = [
    "ChestXRayDataset",
    "collect_image_paths",
    "encode_labels",
    "create_datasets",
    "get_dataloaders",
]
