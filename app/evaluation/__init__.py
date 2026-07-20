"""
Evaluation package for MedVision-AI.
"""

from .evaluator import evaluate_model
from .metrics import (
    compute_metrics,
    format_metrics,
)
from .report import (
    generate_classification_report,
    print_evaluation_summary,
    save_predictions_csv,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    show_predictions,
)
from .utils import (
    CLASS_NAMES,
    denormalize_image,
)

__all__ = [
    "evaluate_model",
    "compute_metrics",
    "format_metrics",
    "generate_classification_report",
    "print_evaluation_summary",
    "save_predictions_csv",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "show_predictions",
    "CLASS_NAMES",
    "denormalize_image",
]
