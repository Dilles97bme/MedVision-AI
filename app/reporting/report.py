"""
==============================================================
MedVision-AI

Module:
report.py

Description:
Generate, format, and save structured prediction reports
for MedVision-AI inference.

Responsibilities
----------------
- Generate structured prediction reports
- Format reports for console or UI display
- Save reports as JSON files

Author:
Dilleswara Rao Intenaka

Project:
MedVision-AI
==============================================================
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# ==========================================================
# Report Generation
# ==========================================================

def generate_report(
    image_name: str,
    prediction: str,
    confidence: float,
    probabilities: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a structured prediction report.

    Parameters
    ----------
    image_name : str
        Name of the input image.

    prediction : str
        Predicted class label.

    confidence : float
        Prediction confidence score.

    probabilities : dict[str, float], optional
        Class probabilities.

    metadata : dict[str, Any], optional
        Additional inference information such as
        model name, device, checkpoint version,
        or inference time.

    Returns
    -------
    dict[str, Any]
        Structured prediction report.
    """

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_name": image_name,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": probabilities or {},
        "metadata": metadata or {},
    }


# ==========================================================
# Report Formatting
# ==========================================================

def format_report(report: dict[str, Any]) -> str:
    """
    Convert a report dictionary into a human-readable string.

    Parameters
    ----------
    report : dict

    Returns
    -------
    str
    """

    lines = [
        "=" * 55,
        "MedVision-AI Prediction Report",
        "=" * 55,
        f"Timestamp   : {report['timestamp']}",
        f"Image       : {report['image_name']}",
        f"Prediction  : {report['prediction']}",
        f"Confidence  : {report['confidence']:.2%}",
    ]

    probabilities = report.get("probabilities", {})

    if probabilities:

        lines.append("")
        lines.append("Class Probabilities")
        lines.append("-" * 25)

        for class_name, probability in probabilities.items():
            lines.append(
                f"{class_name:<12}: {probability:.2%}"
            )

    metadata = report.get("metadata", {})

    if metadata:

        lines.append("")
        lines.append("Metadata")
        lines.append("-" * 25)

        for key, value in metadata.items():
            lines.append(
                f"{key:<15}: {value}"
            )

    return "\n".join(lines)


# ==========================================================
# Save Report
# ==========================================================

def save_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save a prediction report as a JSON file.

    Parameters
    ----------
    report : dict
        Prediction report.

    output_path : str | Path
        Destination JSON file.

    Returns
    -------
    Path
        Path to the saved report.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        mode="w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path
