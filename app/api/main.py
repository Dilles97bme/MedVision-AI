"""
==============================================================
MedVision-AI

FastAPI Application


"""

from fastapi import FastAPI
from app.api.routes import router


tags_metadata = [
    {
        "name": "Inference",
        "description": "Chest X-ray inference endpoints",
    },
]

app = FastAPI(
    title="🫁 MedVision-AI API",
    description="""
    REST API for MedVision-AI.

    Features

    • Chest X-ray Pneumonia Prediction
    • Explainable AI using Grad-CAM
    • JSON Report Generation
    • PDF Report Generation
    • Complete Chest X-ray Analysis
    """,
    version="1.0.0",
    contact={
        "name": "Dilleswara Rao Intenaka",
        "email": "dilles97bme@mail.com",
    },
    openapi_tags=tags_metadata,
)

app.include_router(
    router,
    prefix="/api/v1",
    tags=["Inference"],

)


@app.get("/")
def root():
    """
    Root endpoint.
    """
    return {
        "name": "MedVision-AI API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }
