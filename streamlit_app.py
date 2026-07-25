"""
==============================================================
MedVision-AI

Streamlit Application

Phase 1
--------

UI Foundation

Author:
Dilleswara Rao Intenaka
==============================================================
"""
from pathlib import Path
import json


from app.inference.loader import load_model
from app.inference.predictor import predict_image

from app.explainability.pipeline import generate_gradcam
from app.reporting.report import (
    generate_report,
    format_report,)
from app.reporting.pdf import generate_pdf_report

from PIL import Image
import streamlit as st


# ----------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------

CHECKPOINT_PATH = Path(
    "checkpoints/best_model.pth"
)

# ----------------------------------------------------------
# Reports Directory
# ----------------------------------------------------------

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
# ----------------------------------------------------------
# Cached Model Loader
# ----------------------------------------------------------


@st.cache_resource
def get_model():
    """
    Load the trained model only once.
    """

    model, device = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        freeze_backbone=False,
    )

    return model, device
# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------


st.set_page_config(
    page_title="MedVision-AI",
    page_icon="🫁",
    layout="wide",
)


# ----------------------------------------------------------
# Title
# ----------------------------------------------------------

st.title("🫁 MedVision-AI")

st.caption(
    "Explainable AI Platform for Chest X-ray Disease Detection"
)

st.info(
    "📤 Upload a chest X-ray image and click **Predict** to perform AI analysis."
)

st.divider()


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.title("🫁 MedVision-AI")

st.sidebar.markdown("---")

st.sidebar.subheader("Supported Disease")

st.sidebar.write("• Pneumonia")

st.sidebar.subheader("Model")

st.sidebar.write("EfficientNet-B0")

st.sidebar.subheader("Explainability")

st.sidebar.write("Grad-CAM")

st.sidebar.subheader("Report")

st.sidebar.write("JSON Download")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Version 1.0"
)

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

try:

    model, device = get_model()

    st.sidebar.success("✅ Model Loaded")

except Exception as error:

    st.error(f"Failed to load model.\n\n{error}")

    st.stop()
# ----------------------------------------------------------
# Image Upload
# ----------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"],
)


# ----------------------------------------------------------
# Display Image
# ----------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.success(
        "Image uploaded successfully."
    )

else:

    st.info(
        "Please upload a chest X-ray image."
    )


st.divider()


# ----------------------------------------------------------
# Placeholder
# ----------------------------------------------------------

st.subheader("Prediction")

if uploaded_file is not None:

    if st.button(
        "🔍 Predict",
        width="stretch",
    ):

        with st.spinner("Running inference..."):

            result = predict_image(
                image,
                model,
                device,
            )

            gradcam_result = generate_gradcam(
                image=image,
                model=model,
            )

        # ----------------------------------------------------------
        # Save Images for PDF
        # ----------------------------------------------------------

        original_image_path = REPORTS_DIR / "original.png"
        gradcam_image_path = REPORTS_DIR / "gradcam.png"

        # Save original image
        gradcam_result["image"].save(original_image_path)

        # Save Grad-CAM overlay

        gradcam_result["overlay"].save(gradcam_image_path)

        st.success("Prediction completed.")

        st.markdown("### Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                result["prediction"],
            )

        with col2:
            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}",
            )

        confidence = result["confidence"]

        if confidence >= 0.95:
            st.success("🟢 High confidence prediction")

        elif confidence >= 0.80:
            st.warning("🟡 Moderate confidence prediction")

        else:
            st.error("🔴 Low confidence prediction. Clinical review recommended.")

        if result["prediction"] == "PNEUMONIA":
            st.error("🩺 Prediction: PNEUMONIA")

        else:
            st.success("✅ Prediction: NORMAL")

        st.markdown("### Class Probabilities")

        for class_name, probability in result["probabilities"].items():

            st.write(f"**{class_name}**")

            st.progress(probability)

            st.write(f"{probability:.2%}")

        st.divider()

        st.header("Explainability")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                gradcam_result["image"],
                caption="Original Image",
                width="stretch",
            )

        with col2:

            st.image(
                gradcam_result["overlay"],
                caption="Grad-CAM Overlay",
                width="stretch",
            )

        report = generate_report(
            image_name=uploaded_file.name,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                "Model": "EfficientNet-B0",
                "Device": str(device),
            },
        )

        # ----------------------------------------------------------
        # Generate PDF Report
        # ----------------------------------------------------------

        pdf_path = generate_pdf_report(
            report=report,
            output_path=REPORTS_DIR / "prediction_report.pdf",
            original_image=original_image_path,
            gradcam_image=gradcam_image_path,
        )

        st.divider()

        st.header("📄 Prediction Report")

        st.code(
            format_report(report),
            language="text",
        )

        col1, col2 = st.columns(2)

        with col1:

            with open(pdf_path, "rb") as pdf_file:

                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_file,
                    file_name="MedVision_AI_Report.pdf",
                    mime="application/pdf",
                    width="stretch"
                )

        with col2:

            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(report, indent=4),
                file_name=f"{uploaded_file.name}_report.json",
                mime="application/json",
                width="stretch",
            )

        st.divider()

        st.caption(
            "© 2026 MedVision-AI | Developed by Dilleswara Rao Intenaka"
        )

        st.caption(
            "⚠️ For research and educational purposes only. "
            "Not intended for clinical diagnosis."
        )
