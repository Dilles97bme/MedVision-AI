"""
transforms.py

Image preprocessing pipelines for MedVision-AI.
"""

from PIL import Image
from torchvision import transforms


# ==========================================
# ImageNet Statistics
# ==========================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================
# Helper Function
# ==========================================

def convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert a PIL image to RGB.

    Parameters
    ----------
    img : PIL.Image.Image

    Returns
    -------
    PIL.Image.Image
    """

    return img.convert("RGB")


# ==========================================
# Training Transform
# ==========================================

train_transform = transforms.Compose(
    [
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ==========================================
# Validation Transform
# ==========================================

val_transform = transforms.Compose(
    [
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ==========================================
# Test Transform
# ==========================================

test_transform = transforms.Compose(
    [
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)
