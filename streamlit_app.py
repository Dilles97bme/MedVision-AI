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

from PIL import Image
import streamlit as st


# ----------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------

CHECKPOINT_PATH = Path(
    "checkpoints/best_model.pth"
)


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

st.markdown(
    """
Explainable AI platform for chest X-ray disease detection.

Upload a chest X-ray image to begin the analysis.
"""
)

st.divider()


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.title("Navigation")

st.sidebar.markdown(
    """
### Home

MedVision-AI demonstrates an explainable AI pipeline for
detecting pneumonia from chest X-ray images.

---

### Supported Disease

- Pneumonia

---

### Explainability

- Grad-CAM Visualization

---

### Model

- EfficientNet-B0
"""
)

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

    st.subheader("Uploaded Image")

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
        use_container_width=True,
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

        st.success("Prediction completed.")

        st.markdown("### Prediction Result")

        st.metric(
            label="Predicted Class",
            value=result["prediction"],
        )

        st.metric(
            label="Confidence",
            value=f"{result['confidence']:.2%}",
        )

        if result["prediction"] == "PNEUMONIA":
            st.error("🩺 Prediction: PNEUMONIA")
        else:
            st.success("✅ Prediction: NORMAL")

        st.markdown("### Class Probabilities")

        for class_name, probability in result["probabilities"].items():

            st.write(class_name)

            st.progress(probability)

            st.caption(f"{probability:.2%}")

        st.divider()

        st.header("Explainability")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                gradcam_result["image"],
                caption="Original Image",
                use_container_width=True,
            )

        with col2:

            st.image(
                gradcam_result["overlay"],
                caption="Grad-CAM Overlay",
                use_container_width=True,
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

        st.divider()

        st.header("📄 Prediction Report")

        st.text(format_report(report))

        st.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name=f"{uploaded_file.name}_report.json",
            mime="application/json",
        )
