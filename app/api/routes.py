"""
==============================================================
MedVision-AI

API Routes
==============================================================
"""


from io import BytesIO
from pathlib import Path


from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from datetime import datetime

from app.database.operations import (
    insert_prediction,
    get_prediction_history,
    get_prediction_by_id,
    delete_prediction,
)


from app.api.dependencies import device, model
from app.api.schemas import (
    AnalyzeResponse,
    PredictionResponse,
)
from app.explainability.pipeline import generate_gradcam
from app.inference.predictor import predict_image
from app.reporting.pdf import generate_pdf_report
from app.reporting.report import generate_report

router = APIRouter()

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Pneumonia",
)
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia from an uploaded chest X-ray image.
    """

    # Validate image type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Run inference
    result = predict_image(
        image,
        model,
        device,
    )

    return result


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Complete Chest X-ray Analysis",
)
async def analyze(file: UploadFile = File(...)):
    """
    Complete chest X-ray analysis.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # ----------------------------
    # Prediction
    # ----------------------------

    result = predict_image(
        image,
        model,
        device,
    )

    # ----------------------------
    # Grad-CAM
    # ----------------------------

    gradcam = generate_gradcam(
        image=image,
        model=model,
    )

    # ----------------------------
    # Save images
    # ----------------------------

    base_name = Path(file.filename).stem

    original_path = REPORTS_DIR / f"{base_name}_original.png"

    overlay_path = REPORTS_DIR / f"{base_name}_gradcam.png"

    gradcam["image"].save(original_path)

    gradcam["overlay"].save(overlay_path)

    # ----------------------------
    # Report
    # ----------------------------

    report = generate_report(
        image_name=file.filename,
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        metadata={
            "Model": "EfficientNet-B0",
            "Device": str(device),
        },
    )

    pdf_path = REPORTS_DIR / f"{base_name}_report.pdf"

    generate_pdf_report(
        report=report,
        output_path=pdf_path,
        original_image=original_path,
        gradcam_image=overlay_path,
    )

    # ----------------------------
# Save Prediction to Database
# ----------------------------

    probabilities = result["probabilities"]

    insert_prediction(
        image_name=file.filename,
        prediction=result["prediction"],
        confidence=result["confidence"],
        normal_probability=probabilities["NORMAL"],
        pneumonia_probability=probabilities["PNEUMONIA"],
        model_name="EfficientNet-B0",
        report_path=str(pdf_path),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    return {
        "prediction": result,
        "report": report,
        "pdf": pdf_path.name,
    }


@router.get("/reports/{filename}")
def download_report(filename: str):
    """
    Download a generated PDF report.
    """

    pdf_path = REPORTS_DIR / filename

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not found.",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
    )


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/model")
def model_status():
    return {
        "status": "loaded",
        "device": str(device),
        "model": model.__class__.__name__,
    }


@router.get(
    "/history",
    summary="Prediction History",
)
def prediction_history():
    """
    Return all stored prediction history.
    """

    return get_prediction_history()


@router.get(
    "/history/{prediction_id}",
    summary="Prediction Details",
)
def prediction_details(prediction_id: int):

    record = get_prediction_by_id(prediction_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    return record


@router.delete(
    "/history/{prediction_id}",
    summary="Delete Prediction",
)
def remove_prediction(prediction_id: int):

    deleted = delete_prediction(prediction_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    return {
        "message": "Prediction deleted successfully."
    }
