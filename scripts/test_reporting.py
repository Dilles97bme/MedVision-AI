"""
==============================================================
MedVision-AI

Script:
test_reporting.py

Description:
Test prediction report generation and saving.
==============================================================
"""

from pathlib import Path

from app.inference.loader import load_model
from app.inference.predictor import predict_image
from app.reporting.report import (
    generate_report,
    format_report,
    save_report,
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")

IMAGE_PATH = Path(
    r"dataset/chest_xray/test/PNEUMONIA/person78_bacteria_378.jpeg"
)

OUTPUT_PATH = Path("outputs/reports/prediction_report.json")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    model, device = load_model(CHECKPOINT_PATH)

    result = predict_image(
        image_path=IMAGE_PATH,
        model=model,
        device=device,
    )

    report = generate_report(
        image_name=IMAGE_PATH.name,
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )

    print(format_report(report))

    save_report(
        report,
        OUTPUT_PATH,
    )

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
