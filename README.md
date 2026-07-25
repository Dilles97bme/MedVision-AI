# 🫁 MedVision-AI

> **Explainable AI Platform for Chest X-ray Analysis**

An end-to-end explainable AI platform for automated pneumonia detection from chest X-ray images using EfficientNet-B0, Grad-CAM visualization, and an interactive Streamlit web application.

---

## 📖 Overview

MedVision-AI is a modular deep learning application that demonstrates Explainable Artificial Intelligence (XAI) for chest X-ray analysis. The platform integrates image preprocessing, EfficientNet-B0-based classification, Grad-CAM explainability, structured report generation, and an interactive Streamlit interface into a unified workflow.

The primary objective of the project is to demonstrate how modern deep learning models can be combined with explainability techniques to create transparent, interpretable, and user-friendly AI systems for medical imaging. Designed with a modular software architecture, the project supports future extensions such as REST APIs, Docker deployment, and multi-disease classification, making it suitable for research, education, and portfolio demonstration.

## ✨ Features

| Feature | Description |
|----------|-------------|
| 🫁 Chest X-ray Classification | Automated pneumonia detection using EfficientNet-B0 |
| 🔍 Explainable AI | Grad-CAM visualization highlighting important image regions |
| 📊 Confidence Scores | Displays prediction confidence and class probabilities |
| 📄 PDF Report Generation | Generates downloadable professional prediction reports |
| 📁 JSON Report Export | Structured prediction results in JSON format |
| 🖥️ Interactive Web Interface | User-friendly Streamlit application |
| 🧩 Modular Architecture | Separate modules for preprocessing, inference, explainability, and reporting |
| ⚡ GPU Support | CUDA acceleration for inference when available |
| 🔄 Reusable Pipeline | Designed for integration with Streamlit, FastAPI, and future deployment |


## 🏗️ Project Architecture

```text
                 Chest X-ray Image
                         │
                         ▼
                 Image Preprocessing
                         │
                         ▼
               EfficientNet-B0 Model
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   Prediction & Confidence        Grad-CAM Generation
          │                             │
          └──────────────┬──────────────┘
                         ▼
               Report Generation
          (JSON + Professional PDF)
                         │
                         ▼
            Interactive Streamlit UI
```


### Workflow

1. **Upload:** The user uploads a chest X-ray image through the Streamlit interface.
2. **Preprocessing:** The image is resized and transformed into the format expected by the trained model.
3. **Inference:** EfficientNet-B0 predicts the presence or absence of pneumonia and computes confidence scores.
4. **Explainability:** Grad-CAM generates a heatmap highlighting image regions that contributed most to the prediction.
5. **Reporting:** Prediction results, confidence scores, and metadata are compiled into JSON and PDF reports.
6. **Visualization:** The original image, Grad-CAM overlay, prediction summary, and downloadable reports are displayed through the web interface.

## 📂 Project Structure

```text
MedVision-AI/
│
├── app/                            # Core application modules
│   ├── api/                        # API components (future extension)
│   ├── database/                   # Database utilities
│   ├── evaluation/                 # Model evaluation metrics
│   ├── explainability/             # Grad-CAM implementation
│   ├── inference/                  # Model loading and prediction
│   ├── models/                     # EfficientNet-B0 architecture
│   ├── preprocessing/              # Image preprocessing pipeline
│   ├── reporting/                  # JSON & PDF report generation
│   ├── training/                   # Model training pipeline
│   ├── utils/                      # Common utility functions
│   ├── config.py                   # Global configuration
│   └── __init__.py
│
├── checkpoints/                    # Trained model and training history
│   ├── best_model.pth
│   └── training_history.json
│
├── configs/                        # YAML configuration files
│   ├── app_config.yaml
│   ├── config.yaml
│   └── model_config.yaml
│
├── dataset/                        # Chest X-ray dataset
│   └── chest_xray/
│
├── docs/                           # Project documentation
│
├── frontend/                       # Frontend assets (future extension)
│
├── notebooks/                      # Development & experimentation notebooks
│   ├── Dataset Understanding
│   ├── Dataset Analysis
│   ├── Image Preprocessing
│   ├── Dataloader Pipeline
│   ├── EfficientNet Implementation
│   ├── Training Pipeline
│   ├── Model Evaluation
│   ├── Grad-CAM Explainability
│   └── Batch Explainability Validation
│
├── outputs/                        # Generated outputs and visualizations
│   ├── predictions/
│   ├── reports/
│   ├── explainability/
│   ├── gradcam/
│   ├── heatmaps/
│   ├── visualizations/
│   └── batch_reports/
│
├── reports/                        # Generated PDF reports
│
├── scripts/                        # Testing scripts
│   ├── test_inference.py
│   ├── test_gradcam.py
│   └── test_reporting.py
│
├── streamlit_app.py                # Main Streamlit application
├── requirements.txt                # Project dependencies
├── README.md
└── .gitignore
```

## 📊 Dataset

MedVision-AI is trained and evaluated using the **Chest X-ray Images (Pneumonia)** dataset, a publicly available benchmark dataset hosted on Kaggle. The dataset contains pediatric chest X-ray images categorized into **Normal** and **Pneumonia** classes and is widely used for research in automated pneumonia detection.

### Dataset Source

- **Name:** Chest X-ray Images (Pneumonia)
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Dataset Statistics

| Category | Images |
|----------|-------:|
| Normal | 1,583 |
| Pneumonia | 4,273 |
| **Total** | **5,856** |

### Dataset Split

| Split | Normal | Pneumonia | Total |
|------|-------:|----------:|------:|
| Training | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Testing | 234 | 390 | 624 |

> **Note:** The original validation set provided with the dataset contains only 16 images. For robust model development, users may consider creating a larger validation split from the training data.

### Directory Structure

```text
dataset/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

### Data Characteristics

- **Imaging Modality:** Chest X-ray (Posterior-Anterior and Anterior-Posterior views)
- **Classification Task:** Binary Classification
- **Classes:**
  - ✅ Normal
  - 🩺 Pneumonia
- **Image Format:** JPEG (.jpeg)
- **Framework Compatibility:** PyTorch `ImageFolder`

### Dataset Availability

The dataset is **not included** in this repository due to its size and Kaggle licensing requirements. Please download it directly from the official Kaggle page and place it in the following directory:

```text
dataset/chest_xray/
```

After downloading, ensure the folder structure matches the layout shown above before training or running inference.