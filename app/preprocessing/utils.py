"""
utils.py

Helper functions for image preprocessing.
"""

from pathlib import Path
from PIL import Image


def load_image(image_path: Path) -> Image.Image:
    """
    Load an image using PIL.
    """

    return Image.open(image_path)


def verify_image(image_path: Path) -> bool:
    """
    Check whether an image is valid.
    """

    try:
        with Image.open(image_path) as img:
            img.verify()

        return True

    except Exception:
        return False


def get_image_size(image_path: Path):
    """
    Return image width and height.
    """

    with Image.open(image_path) as img:
        return img.size
