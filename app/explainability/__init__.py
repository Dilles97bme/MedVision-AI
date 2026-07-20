"""
==============================================================
MedVision-AI

Explainability Package

Provides Grad-CAM generation, visualization,
and reporting utilities.
==============================================================
"""

from .gradcam import GradCAM
from .visualization import (
    apply_colormap,
    overlay_heatmap,
    visualize_gradcam,
    save_visualization,
)
from .report import (
    PredictionReport,
    default_interpretation,
    generate_report,
    format_report,
    report_to_dict,
    report_to_json,
    save_report,
)

__all__ = [
    "GradCAM",

    "apply_colormap",
    "overlay_heatmap",
    "visualize_gradcam",
    "save_visualization",

    "PredictionReport",
    "default_interpretation",
    "generate_report",
    "format_report",
    "report_to_dict",
    "report_to_json",
    "save_report",
]
