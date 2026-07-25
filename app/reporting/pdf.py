"""
==============================================================
MedVision-AI

Module:
pdf.py

Description:
Generate professional PDF reports for MedVision-AI predictions.

This module creates downloadable PDF reports containing
prediction results, confidence scores, class probabilities,
metadata, and a medical disclaimer.

Author:
Dilleswara Rao Intenaka

Project:
MedVision-AI
==============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from reportlab.platypus import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Spacer,
    Paragraph,
    Table,
    TableStyle,
)


def generate_pdf_report(
    report: Dict[str, Any],
    output_path: str | Path,
    original_image: str | Path | None = None,
    gradcam_image: str | Path | None = None,
) -> Path:
    """
    Generate a PDF prediction report.

    Parameters
    ----------
    report : dict
        Dictionary returned by generate_report().

    output_path : str | Path
        Destination PDF path.

    Returns
    -------
    Path
        Saved PDF path.
    """

    output_path = Path(output_path)

    doc = SimpleDocTemplate(
        str(output_path),
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]

    normal = styles["BodyText"]

    elements = []

    # --------------------------------------------------
    # Title
    # --------------------------------------------------

    elements.append(
        Paragraph(
            "MedVision-AI Prediction Report",
            title_style,
        )
    )

    elements.append(Spacer(1, 0.30 * inch))

    # --------------------------------------------------
    # Original Chest X-ray
    # --------------------------------------------------

    if original_image is not None:

        elements.append(
            Paragraph(
                "Original Chest X-ray",
                heading_style,
            )
        )

        elements.append(
            Image(
                str(original_image),
                width=4.5 * inch,
                height=4.5 * inch,
            )
        )

        elements.append(
            Spacer(1, 0.25 * inch)
        )

    # --------------------------------------------------
    # Prediction Summary
    # --------------------------------------------------

    elements.append(
        Paragraph(
            "Prediction Summary",
            heading_style,
        )
    )

    summary_data = [
        ["Image", report["image_name"]],
        ["Prediction", report["prediction"]],
        ["Confidence", f"{report['confidence']:.2%}"],
        ["Generated", report["timestamp"]],
    ]

    table = Table(summary_data, colWidths=[2 * inch, 4 * inch])

    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    elements.append(table)

    elements.append(Spacer(1, 0.30 * inch))

    # --------------------------------------------------
    # Probabilities
    # --------------------------------------------------

    elements.append(
        Paragraph(
            "Class Probabilities",
            heading_style,
        )
    )

    probability_data = [["Class", "Probability"]]

    for class_name, probability in report["probabilities"].items():
        probability_data.append(
            [
                class_name,
                f"{probability:.2%}",
            ]
        )

    table = Table(probability_data)

    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    elements.append(table)

    elements.append(Spacer(1, 0.30 * inch))

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    if gradcam_image is not None:

        elements.append(
            Paragraph(
                "Grad-CAM Explanation",
                heading_style,
            )
        )

        elements.append(
            Image(
                str(gradcam_image),
                width=4.5 * inch,
                height=4.5 * inch,
            )
        )

        elements.append(
            Spacer(1, 0.30 * inch)
        )

    # --------------------------------------------------
    # Model Information
    # --------------------------------------------------

    elements.append(
        Paragraph(
            "Model Information",
            heading_style,
        )
    )

    metadata = report.get("metadata", {})

    metadata_data = []

    for key, value in metadata.items():
        metadata_data.append([key, str(value)])

    if metadata_data:

        table = Table(metadata_data)

        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.beige),
                ]
            )
        )

        elements.append(table)

    elements.append(Spacer(1, 0.40 * inch))

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Disclaimer</b>",
            heading_style,
        )
    )

    disclaimer = (
        "This report was automatically generated by "
        "<b>MedVision-AI</b>. "
        "It is intended for research and educational purposes "
        "only and should not be used as a substitute for "
        "professional clinical diagnosis."
    )

    elements.append(
        Paragraph(
            disclaimer,
            normal,
        )
    )

    doc.build(elements)

    return output_path
