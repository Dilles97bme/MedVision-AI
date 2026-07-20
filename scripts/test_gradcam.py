"""
==============================================================
MedVision-AI

Script:
test_gradcam.py

Description:
Test the Grad-CAM explainability pipeline.

Workflow
--------
Load Model
    ↓
Load Image
    ↓
Preprocess Image
    ↓
Generate Grad-CAM
    ↓
Create Visualization
    ↓
Save Visualization

Author:
Dilleswara Rao Intenaka
==============================================================
"""

from pathlib import Path

import numpy as np

from app.explainability.gradcam import GradCAM
from app.explainability.visualization import (
    save_visualization,
    visualize_gradcam,
)
from app.inference.loader import load_model
from app.inference.predictor import (
    load_image,
    preprocess_image,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")

IMAGE_PATH = Path(
    r"dataset/chest_xray/test/PNEUMONIA/person78_bacteria_378.jpeg"
)

OUTPUT_PATH = Path(
    "outputs/gradcam/person78_bacteria_378_gradcam.png"
)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, device = load_model(CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(IMAGE_PATH)

    image_tensor = preprocess_image(image)

    # --------------------------------------------------------
    # Target layer (EfficientNet-B0)
    # --------------------------------------------------------

    target_layer = model.features[-1]

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    with GradCAM(
        model=model,
        target_layer=target_layer,
    ) as gradcam:

        heatmap = gradcam.generate(image_tensor)

    # --------------------------------------------------------
    # Convert PIL image to NumPy
    # --------------------------------------------------------

    image_np = np.array(image)

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------

    figure = visualize_gradcam(
        image=image_np,
        heatmap=heatmap,
    )

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    save_visualization(
        figure=figure,
        filepath=OUTPUT_PATH,
        close=True,
    )

    print(f"Grad-CAM visualization saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
