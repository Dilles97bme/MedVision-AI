"""
==============================================================
MedVision-AI

Streamlit Application

Phase 1
--------

UI Foundation


==============================================================
"""
import json
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
import os


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


API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/api/v1",
)
st.sidebar.success("✅ Connected to FastAPI")

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

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            try:
                response = requests.post(
                    f"{API_URL}/analyze",
                    files=files,
                    timeout=120,
                )

                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                st.error(f"Cannot connect to FastAPI.\n\n{e}")
                st.stop()

            data = response.json()

        result = data["prediction"]
        report = data["report"]
        pdf_name = data["pdf"]
        original_image = data["original_image"]
        gradcam_image = data["gradcam_image"]

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

        st.divider()

        st.header("Explainability")

        col1, col2 = st.columns(2)

        with col1:

            original_response = requests.get(
                f"{API_URL}/images/{original_image}"
            )

            if original_response.ok:
                st.image(
                    Image.open(
                        BytesIO(original_response.content)
                    ),
                    caption="Original Image",
                    width="stretch",
                )
            else:
                st.error("Could not load original image.")

        with col2:

            gradcam_response = requests.get(
                f"{API_URL}/images/{gradcam_image}"
            )

            if gradcam_response.ok:
                st.image(
                    Image.open(
                        BytesIO(gradcam_response.content)
                    ),
                    caption="Grad-CAM Overlay",
                    width="stretch",
                )
            else:
                st.error("Could not load Grad-CAM image.")

        st.divider()

        st.header("📄 Prediction Report")

        col1, col2 = st.columns(2)

        with col1:
            pdf_response = requests.get(
                f"{API_URL}/reports/{pdf_name}"
            )

            if pdf_response.ok:
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_response.content,
                    file_name=pdf_name,
                    mime="application/pdf",
                    width="stretch",
                )
            else:
                st.error("Unable to download PDF report.")

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
