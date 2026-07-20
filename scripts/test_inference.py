"""
==============================================================
MedVision-AI

Script:
test_inference.py

Description:
Test the complete inference pipeline.

Workflow
--------
Load Model
    ↓
Predict Image
    ↓
Display Prediction Results

Author:
Dilleswara Rao Intenaka
==============================================================
"""

from pathlib import Path

from app.inference.loader import load_model
from app.inference.predictor import predict_image


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")

TEST_IMAGES = [
    Path(r"dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"),
    Path(r"C:/MedVision-AI/dataset/chest_xray/test/PNEUMONIA/person78_bacteria_378.jpeg"),
]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    model, device = load_model(CHECKPOINT_PATH)

    print(f"Device : {device}")

    for image_path in TEST_IMAGES:

        print("\n" + "=" * 60)

        result = predict_image(
            image_path=image_path,
            model=model,
            device=device,
        )

        print(f"Image        : {image_path.name}")
        print(f"Prediction   : {result['prediction']}")
        print(f"Confidence   : {result['confidence']:.2%}")

        print("\nProbabilities")

        for class_name, probability in result["probabilities"].items():
            print(f"  {class_name:<12}: {probability:.2%}")


if __name__ == "__main__":
    main()
