"""
==============================================================
MedVision-AI

Module:
report.py

Description:
Generate structured prediction reports for Grad-CAM
explanations.

This module provides utilities for creating concise,
human-readable prediction summaries that can be displayed
in notebooks, Streamlit applications, FastAPI responses,
or exported into PDF reports.

Author:
Dilleswara Rao Intenaka

Project:
MedVision-AI
==============================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ==========================================================
# Prediction Report
# ==========================================================

@dataclass(slots=True)
class PredictionReport:
    """
    Structured prediction report.
    """

    predicted_class: str

    confidence: float

    image_name: Optional[str] = None

    heatmap_generated: bool = True

    interpretation: Optional[str] = None

    model_name: Optional[str] = None

    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
    )

    def __str__(self) -> str:
        """
        Return formatted report.
        """

        return format_report(self)


# ==========================================================
# Default Interpretation
# ==========================================================

def default_interpretation(
    predicted_class: str,
) -> str:
    """
    Generate a default clinical interpretation.
    """

    label = predicted_class.lower().strip()

    if label == "pneumonia":

        return (
            "The model identified imaging patterns "
            "consistent with pneumonia. The Grad-CAM "
            "heatmap highlights the regions that "
            "contributed most strongly to this prediction. "
            "This output is intended as a decision-support "
            "tool and should be interpreted alongside "
            "clinical assessment by a qualified healthcare "
            "professional."
        )

    if label == "normal":

        return (
            "The model did not identify imaging patterns "
            "suggestive of pneumonia. The Grad-CAM "
            "visualization highlights the image regions "
            "considered during prediction. This output "
            "should be interpreted together with clinical "
            "findings and radiological expertise."
        )

    return (
        "Grad-CAM visualization generated successfully."
    )


# ==========================================================
# Generate Report
# ==========================================================

def generate_report(
    predicted_class: str,
    confidence: float,
    image_name: str | None = None,
    model_name: str | None = None,
    interpretation: str | None = None,
) -> PredictionReport:
    """
    Create a structured prediction report.

    Parameters
    ----------
    predicted_class : str
        Predicted class label.

    confidence : float
        Prediction confidence between 0 and 1.

    image_name : str, optional
        Image filename.

    model_name : str, optional
        Model name.

    interpretation : str, optional
        Custom interpretation.

    Returns
    -------
    PredictionReport
    """

    if not predicted_class.strip():
        raise ValueError(
            "Predicted class cannot be empty."
        )

    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    if interpretation is None:

        interpretation = default_interpretation(
            predicted_class
        )

    return PredictionReport(
        predicted_class=predicted_class,
        confidence=confidence,
        image_name=image_name,
        heatmap_generated=True,
        interpretation=interpretation,
        model_name=model_name,
    )


# ==========================================================
# Format Report
# ==========================================================

def format_report(
    report: PredictionReport,
) -> str:
    """
    Convert a PredictionReport into readable text.
    """

    lines = []

    lines.append("=" * 60)
    lines.append("MedVision-AI Prediction Report")
    lines.append("=" * 60)

    if report.image_name:

        lines.append(
            f"Image           : {report.image_name}"
        )

    if report.model_name:

        lines.append(
            f"Model           : {report.model_name}"
        )

    lines.append(
        f"Prediction      : {report.predicted_class}"
    )

    lines.append(
        f"Confidence      : {report.confidence:.2%}"
    )

    lines.append(
        "Grad-CAM        : "
        f"{'Generated' if report.heatmap_generated else 'Not Generated'}"
    )

    lines.append(
        f"Timestamp       : {report.timestamp}"
    )

    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 60)
    lines.append(report.interpretation or "")

    return "\n".join(lines)


# ==========================================================
# Dictionary Export
# ==========================================================

def report_to_dict(
    report: PredictionReport,
) -> dict:
    """
    Convert report to dictionary.
    """

    return asdict(report)


# ==========================================================
# JSON Export
# ==========================================================

def report_to_json(
    report: PredictionReport,
    indent: int = 4,
) -> str:
    """
    Convert report to JSON.
    """

    return json.dumps(
        report_to_dict(report),
        indent=indent,
    )


# ==========================================================
# Save Report
# ==========================================================

def save_report(
    report: PredictionReport,
    filepath: str | Path,
) -> None:
    """
    Save prediction report as JSON.

    Parameters
    ----------
    report : PredictionReport
        Prediction report.

    filepath : str or Path
        Output JSON path.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report_to_dict(report),
            file,
            indent=4,
        )
