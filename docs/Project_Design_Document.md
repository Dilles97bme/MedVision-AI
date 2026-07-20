# MedVision AI
# Project Design Document (PDD)

---

# Document Information

| Field | Description |
|-------|-------------|
| **Project Name** | MedVision AI |
| **Project Type** | Explainable AI-Based Clinical Decision Support System |
| **Project Version** | 1.0 |
| **Document Type** | Project Design Document (PDD) |
| **Prepared By** | Intenaka Dilleswara Rao |
| **Organization** | IIT Kharagpur |
| **Department** | Medical Imaging and Informatics |
| **Project Status** | Design Phase |
| **Date** | July 2026 |

---

# Revision History

| Version | Date | Author | Description |
|----------|------|--------|-------------|
| 1.0 | July 2026 | Intenaka Dilleswara Rao | Initial Project Design Document |

---

# Executive Summary

MedVision AI is an end-to-end Explainable Artificial Intelligence (XAI) software platform designed to assist healthcare professionals in the automated screening of pneumonia from chest X-ray images. The system combines deep learning, explainable AI techniques, modern backend technologies, and an interactive web interface into a unified software product.

The platform accepts chest radiographs uploaded by the user, performs standardized preprocessing, predicts whether the image belongs to a normal or pneumonia case using a transfer learning model, generates a Grad-CAM heatmap to explain the prediction, and produces a downloadable clinical report. All prediction records are stored locally to enable auditing and historical analysis.

Unlike traditional academic projects that terminate after model training, MedVision AI demonstrates the complete lifecycle of an AI-powered healthcare application—from requirement analysis and software architecture to deployment and documentation. The project emphasizes modular software engineering, explainable AI, reproducibility, and user-centered design.

The current version focuses exclusively on binary pneumonia classification using chest X-ray images. It is intended for educational, research, and prototype clinical decision-support purposes and is **not intended to replace professional medical diagnosis or treatment**.

---

# 1. Project Overview

## 1.1 Purpose

The purpose of MedVision AI is to develop an explainable AI-assisted clinical decision support platform capable of automatically detecting pneumonia from chest X-ray images while providing interpretable visual explanations of the model's predictions.

The software aims to demonstrate how deep learning models can be integrated into a complete healthcare application that supports clinicians rather than replacing them.

---

## 1.2 Project Description

MedVision AI integrates medical image analysis, explainable artificial intelligence, backend API development, frontend visualization, database management, and report generation into a single modular software platform.

The system enables healthcare professionals to upload chest X-ray images through an intuitive web interface, receive AI-assisted predictions within seconds, visualize the anatomical regions influencing the prediction using Grad-CAM, and download a structured report containing the prediction results.

The application follows a modular architecture, allowing individual software components—including preprocessing, inference, explainability, database management, and report generation—to be developed, tested, and maintained independently.

---

## 1.3 Product Vision

The vision of MedVision AI is to bridge the gap between artificial intelligence research and practical healthcare software by developing a transparent, interpretable, and deployable AI platform that assists clinicians in the early detection of pneumonia.

Rather than functioning as an autonomous diagnostic system, MedVision AI is designed to serve as an intelligent assistant that enhances clinical efficiency, improves decision support, and promotes trust in AI through explainability.

The long-term vision is to evolve the platform into a scalable medical imaging framework capable of supporting multiple thoracic diseases, DICOM image formats, cloud deployment, and hospital information system integration.

---

# 2. Background

Pneumonia is one of the leading infectious diseases affecting the respiratory system and remains a significant cause of morbidity and mortality worldwide. According to the World Health Organization (WHO), pneumonia disproportionately affects children, elderly individuals, and patients with weakened immune systems.

Chest X-ray imaging is the most commonly used diagnostic modality for detecting pulmonary infections because it is inexpensive, widely available, and non-invasive. However, interpreting chest radiographs requires considerable clinical expertise and experience.

Large healthcare institutions process hundreds to thousands of chest X-rays every day. This growing diagnostic workload increases reporting time, contributes to physician fatigue, and may affect diagnostic consistency.

Recent advances in artificial intelligence, particularly Convolutional Neural Networks (CNNs), have demonstrated remarkable performance in automatically identifying pathological patterns from medical images. However, many deep learning systems operate as "black-box" models that provide predictions without explaining their reasoning.

Explainable Artificial Intelligence (XAI) techniques such as Gradient-weighted Class Activation Mapping (Grad-CAM) improve transparency by highlighting the image regions that contribute most to the model's decision. Such visual explanations improve clinician confidence and support responsible AI adoption in healthcare.

MedVision AI combines deep learning with explainability techniques to demonstrate how AI models can assist healthcare professionals while maintaining transparency and interpretability.

---

# 3. Problem Statement

Healthcare organizations generate an enormous number of chest radiographs every day. Manual interpretation of these images requires experienced radiologists and is often limited by increasing workloads, reporting delays, and human fatigue.

Although deep learning models have demonstrated excellent performance in pneumonia detection, many existing AI systems provide only prediction probabilities without explaining the underlying reasoning. This lack of transparency reduces clinician trust and limits real-world adoption.

Furthermore, most publicly available AI implementations remain limited to research notebooks and lack the software engineering components required for practical deployment, including web interfaces, APIs, databases, reporting systems, and deployment support.

Therefore, there is a need for an explainable AI-assisted clinical decision support platform capable of:

- Automatically detecting pneumonia from chest X-ray images.
- Providing interpretable visual explanations of predictions.
- Maintaining prediction history.
- Generating structured clinical reports.
- Demonstrating complete end-to-end software deployment.

MedVision AI addresses these challenges by combining explainable deep learning with modern software engineering practices into a unified healthcare application.

---

# 4. Motivation

As a postgraduate student specializing in Medical Imaging and Informatics at IIT Kharagpur, I have worked extensively on machine learning, explainable AI, and healthcare data analysis.

Through my previous research involving EEG signal classification and interpretable machine learning models, I recognized that achieving high predictive accuracy alone is insufficient for healthcare applications. Medical AI systems must also be transparent, reliable, reproducible, and deployable.

This project was motivated by the desire to bridge the gap between academic machine learning research and real-world AI software development.

Rather than developing another standalone deep learning model, MedVision AI aims to demonstrate the complete engineering lifecycle of an AI healthcare application—including system design, backend development, frontend integration, explainability, database management, report generation, testing, containerization, and deployment.

The project also serves as a portfolio demonstrating competencies in:

- Medical image analysis
- Deep learning
- Explainable AI
- Software engineering
- Backend development
- Frontend development
- AI deployment
- Healthcare application development

---

# 5. Project Objectives

## 5.1 Primary Objective

Develop an end-to-end explainable AI software platform capable of automatically detecting pneumonia from chest X-ray images and assisting clinicians through interpretable predictions delivered via an interactive web application.

---

## 5.2 Business Objectives

- Reduce the time required for preliminary chest X-ray screening.
- Demonstrate how AI can support clinicians rather than replace them.
- Provide an intuitive interface suitable for educational and research environments.
- Showcase an end-to-end deployable AI healthcare solution.

---

## 5.3 Technical Objectives

- Develop a robust medical image preprocessing pipeline.
- Train an EfficientNet-B0 transfer learning model for pneumonia classification.
- Implement Grad-CAM for visual explainability.
- Build RESTful APIs using FastAPI.
- Develop an interactive Streamlit frontend.
- Store prediction history using SQLite.
- Generate downloadable PDF reports.
- Package the application using Docker.
- Maintain modular, reusable, and scalable software architecture.

---

## 5.4 Research Objectives

- Evaluate the effectiveness of transfer learning for pneumonia detection.
- Assess model performance using clinically relevant evaluation metrics.
- Investigate Grad-CAM as an explainability technique for chest X-ray interpretation.
- Demonstrate how explainable AI can improve transparency in medical imaging applications.

---

# 6. Project Scope

## 6.1 In Scope

The first version of MedVision AI will include the following capabilities:

- Binary classification of chest X-ray images.
- Pneumonia versus Normal prediction.
- Automatic image preprocessing.
- Deep learning inference using EfficientNet-B0.
- Prediction confidence visualization.
- Grad-CAM heatmap generation.
- Interactive Streamlit web interface.
- FastAPI backend services.
- SQLite prediction history database.
- PDF report generation.
- Docker-based deployment.
- GitHub documentation.

---

## 6.2 Out of Scope

The following features are intentionally excluded from Version 1:

- Multi-disease chest X-ray diagnosis.
- DICOM image support.
- PACS integration.
- Electronic Health Record (EHR) integration.
- Multi-user authentication.
- Cloud-based deployment.
- Mobile application.
- Real-time hospital integration.
- Clinical approval or regulatory certification.
- Autonomous medical diagnosis.

---

## 6.3 Product Boundaries

MedVision AI is designed as an AI-assisted clinical decision support system.

The software **does not replace radiologists or physicians**. Instead, it provides supporting information that may assist healthcare professionals during image interpretation.

The platform is intended for educational, research, and prototype software development purposes.

Clinical decisions should always be made by qualified healthcare professionals.

---

# 7. Stakeholders

Stakeholders are individuals, groups, or organizations that have a direct or indirect interest in the development, deployment, and usage of MedVision AI. Understanding stakeholder expectations ensures that the software is designed to meet both technical and user requirements.

## 7.1 Primary Stakeholders

### Project Developer

The project developer is responsible for designing, developing, testing, documenting, and maintaining the complete software system. This includes model development, backend implementation, frontend development, database integration, and deployment.

**Responsibilities**

- System design
- AI model development
- Software implementation
- Testing
- Documentation
- Deployment
- Maintenance

---

### Healthcare Professionals

Healthcare professionals are the primary users of the application. They use the software as an AI-assisted decision support tool during chest X-ray interpretation.

Examples include:

- Radiologists
- Physicians
- Pulmonologists
- General Practitioners

Their expectations include:

- Fast predictions
- Reliable performance
- Easy-to-understand explanations
- Simple user interface

---

### Medical Researchers

Researchers may use MedVision AI to evaluate explainable AI techniques, compare model performance, and investigate AI applications in medical imaging.

Expected benefits include:

- Reproducible experiments
- Explainable predictions
- Performance evaluation
- Easy extensibility

---

### Medical Students

Students can use the software as an educational platform to understand:

- Pneumonia detection
- Medical imaging
- Explainable AI
- Deep learning applications

---

## 7.2 Secondary Stakeholders

### Healthcare Technology Companies

Organizations developing AI-powered healthcare solutions may use this project as a prototype demonstrating modern AI software architecture.

Examples include:

- Siemens Healthineers
- GE HealthCare
- Philips Healthcare
- Qure.ai
- Niramai

---

### Academic Institutions

Universities and research laboratories may use the project for teaching and research purposes.

---

### Open Source Community

Developers can contribute improvements, fix bugs, or extend the software to support additional diseases or imaging modalities.

---

# 8. End Users

The primary end users of MedVision AI are professionals and learners who require AI-assisted interpretation of chest X-ray images.

## 8.1 Radiologists

Radiologists can upload chest X-rays to receive preliminary AI-assisted predictions along with Grad-CAM visual explanations that highlight suspicious regions.

---

## 8.2 Physicians

General physicians may use the platform as a secondary opinion during pneumonia screening, particularly in resource-constrained environments.

---

## 8.3 Medical Students

Students can explore how AI models analyze chest X-rays and learn the principles of explainable AI.

---

## 8.4 Researchers

Researchers may use the application to evaluate transfer learning models, compare explainability methods, and reproduce experiments.

---

## 8.5 AI Developers

Developers interested in healthcare AI can study the modular software architecture and extend the platform with additional features.

---

# 9. Assumptions

The following assumptions have been made during the design and development of MedVision AI.

## Data Assumptions

- The uploaded images are chest X-ray images.
- Images belong to either the Normal or Pneumonia class.
- Dataset labels are assumed to be correct.
- Images are free from personally identifiable patient information.
- Images are of sufficient quality for model inference.

---

## User Assumptions

- Users possess basic computer literacy.
- Users understand that the software is an AI-assisted support system.
- Users do not interpret AI predictions as final medical diagnoses.

---

## System Assumptions

- Internet access is not required for local execution.
- The application runs on Windows, Linux, or macOS.
- Python and required dependencies are installed.
- Sufficient memory is available for model inference.

---

## Development Assumptions

- Transfer learning provides better performance than training a CNN from scratch for this dataset.
- EfficientNet-B0 provides a good balance between accuracy and computational efficiency.
- Grad-CAM generates meaningful visual explanations for pneumonia classification.

---

# 10. Constraints

The project is intentionally developed within the following constraints.

## Technical Constraints

- Binary classification only.
- JPEG and PNG image formats only.
- Local SQLite database.
- Single-image prediction.
- Local deployment.

---

## Dataset Constraints

- Only the Kaggle Chest X-ray Pneumonia dataset is used.
- Dataset size is limited compared to large clinical repositories.
- Dataset represents only two classes.
- Labels originate from the public dataset.

---

## Computational Constraints

- Development performed on consumer-grade hardware.
- GPU availability may be limited.
- Model inference should remain computationally efficient.

---

## Project Constraints

- Developed as an educational and research prototype.
- Limited development time.
- No clinical validation performed.
- No regulatory approval.

---

# 11. Functional Requirements

Functional requirements describe what the software must do.

## FR-1 Image Upload

The system shall allow users to upload chest X-ray images using the web interface.

---

## FR-2 Image Validation

The system shall verify supported file formats before processing.

Supported formats:

- JPG
- JPEG
- PNG

---

## FR-3 Image Preprocessing

The system shall automatically preprocess uploaded images by:

- Resizing
- Normalization
- Tensor conversion

---

## FR-4 Disease Prediction

The system shall classify uploaded images into one of the following categories:

- Normal
- Pneumonia

---

## FR-5 Confidence Score

The system shall display prediction confidence for each class.

---

## FR-6 Explainability

The system shall generate Grad-CAM heatmaps highlighting regions influencing the prediction.

---

## FR-7 Prediction History

The system shall store prediction history in a SQLite database.

Stored information includes:

- Image name
- Prediction
- Confidence
- Date
- Time

---

## FR-8 Clinical Report

The system shall generate downloadable PDF reports containing:

- Uploaded image
- Prediction
- Confidence score
- Grad-CAM visualization
- Timestamp
- Model version
- Medical disclaimer

---

## FR-9 REST API

The backend shall expose REST APIs for:

- Prediction
- History retrieval
- Health check
- Report generation

---

## FR-10 User Interface

The system shall provide an intuitive web interface for interacting with the AI model.

---

# 12. Non-functional Requirements

Non-functional requirements define how well the software performs.

## Performance

- Prediction should complete within a few seconds on supported hardware.
- The interface should remain responsive during inference.

---

## Reliability

- The application should handle invalid inputs gracefully.
- Prediction history should be stored without corruption.

---

## Usability

- Interface should be intuitive.
- Users should require minimal technical knowledge.

---

## Maintainability

- Source code should follow modular architecture.
- Components should be independently maintainable.

---

## Scalability

The architecture should support future integration of:

- Additional diseases
- Larger datasets
- Cloud deployment
- Authentication
- DICOM support

---

## Security

- No patient-identifiable information should be stored.
- Input validation should prevent malformed uploads.
- Local database access should remain restricted to the application.

---

## Portability

The software should execute consistently across:

- Windows
- Linux
- macOS

using Docker or a Python virtual environment.

---

## Reproducibility

The software should allow experiments to be reproduced using:

- Fixed random seeds
- Version-controlled code
- Saved model checkpoints
- Configuration files

---

# 13. Success Criteria

The project will be considered successful if the following objectives are achieved.

## AI Performance

- EfficientNet-B0 successfully trains using the selected dataset.
- Model demonstrates satisfactory classification performance on the test set.
- Grad-CAM produces meaningful visual explanations.

---

## Software Performance

- Users can upload chest X-ray images without errors.
- Predictions are generated successfully.
- Prediction history is stored correctly.
- PDF reports are generated successfully.
- REST APIs function correctly.
- Streamlit frontend communicates with the backend.

---

## Engineering Quality

- Modular project architecture.
- Well-documented source code.
- Version-controlled development using Git.
- Dockerized deployment.
- Comprehensive project documentation.

---

## User Experience

Users should be able to:

- Upload an image
- Receive predictions
- Understand Grad-CAM explanations
- Download reports
- View prediction history

without requiring programming knowledge.

---

# 14. Dataset Description

## 14.1 Dataset Overview

The MedVision AI platform uses the **Chest X-ray Images (Pneumonia)** dataset published on Kaggle by Paul Mooney. The dataset is designed for binary image classification and contains pediatric chest radiographs categorized into two diagnostic classes:

- Normal
- Pneumonia

The dataset is widely used in academic research and educational projects because it provides a well-organized directory structure and sufficient data for demonstrating deep learning techniques in medical image analysis.

---

## 14.2 Why This Dataset?

The dataset was selected based on the following considerations:

- Publicly available without licensing restrictions for research and educational use.
- Well-organized directory structure that simplifies data loading.
- Suitable dataset size for transfer learning.
- Binary classification simplifies software development while demonstrating a complete AI pipeline.
- Frequently used in medical imaging research, enabling comparison with existing studies.
- Appropriate for developing explainable AI applications using Grad-CAM.

Although larger datasets such as NIH ChestX-ray14 or CheXpert contain more disease categories, they require significantly more computational resources and complex multi-label learning strategies.

For Version 1 of MedVision AI, a binary classification problem provides an ideal balance between model complexity, development time, and software engineering objectives.

---

## 14.3 Dataset Structure

```
dataset/

├── train/
│      ├── NORMAL/
│      └── PNEUMONIA/
│
├── val/
│      ├── NORMAL/
│      └── PNEUMONIA/
│
└── test/
       ├── NORMAL/
       └── PNEUMONIA/
```

The dataset is already divided into:

- Training Set
- Validation Set
- Testing Set

This predefined structure simplifies reproducible experimentation.

---

## 14.4 Dataset Characteristics

| Property | Description |
|----------|-------------|
| Imaging Modality | Chest X-ray |
| Classification Type | Binary Classification |
| Classes | Normal, Pneumonia |
| Image Format | JPEG |
| Color Space | Grayscale |
| Task | Image Classification |

---

## 14.5 Dataset Limitations

Although suitable for prototype development, the dataset has several limitations.

- Binary disease classification only.
- Limited demographic diversity.
- Pediatric population only.
- Images collected from a limited number of institutions.
- Not representative of real-world hospital data.
- No DICOM metadata.
- No clinical information accompanies the images.

These limitations should be considered before applying the model in clinical environments.

---

# 15. Technology Stack

MedVision AI follows a modular software architecture in which each technology has a specific responsibility.

| Layer | Technology |
|--------|------------|
| Programming Language | Python 3.11+ |
| Deep Learning Framework | PyTorch |
| Transfer Learning Model | EfficientNet-B0 |
| Medical Image Processing | Pillow (PIL), OpenCV |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning Utilities | Scikit-learn |
| Explainable AI | Grad-CAM |
| Backend Framework | FastAPI |
| API Server | Uvicorn |
| Frontend | Streamlit |
| Database | SQLite |
| ORM (Optional) | SQLAlchemy |
| PDF Generation | ReportLab |
| Version Control | Git |
| Repository Hosting | GitHub |
| Deployment | Docker |
| IDE | Visual Studio Code |

---

# 16. Technology Selection Justification

Every technology selected for MedVision AI was chosen after considering performance, simplicity, scalability, and suitability for AI application development.

---

## Python

Python serves as the primary programming language because of its extensive ecosystem for artificial intelligence, machine learning, medical imaging, and web development.

Advantages include:

- Readable syntax
- Rich AI libraries
- Excellent community support
- Cross-platform compatibility

---

## PyTorch

PyTorch was selected because it provides:

- Dynamic computational graphs
- Excellent debugging support
- Native transfer learning utilities
- Large research community
- Seamless GPU acceleration

It is widely adopted in both academic research and industrial AI development.

---

## EfficientNet-B0

EfficientNet-B0 was selected as the backbone model due to its excellent balance between accuracy and computational efficiency.

Reasons include:

- Pretrained ImageNet weights
- Fewer parameters than larger CNNs
- Faster inference
- Lower GPU memory requirements
- Strong benchmark performance

Transfer learning significantly reduces training time while improving generalization on limited datasets.

---

## Why EfficientNet Instead of ResNet?

ResNet is a proven and reliable architecture; however, EfficientNet offers several practical advantages for this project.

| EfficientNet-B0 | ResNet50 |
|-----------------|----------|
| Smaller model size | Larger model |
| Lower computational cost | Higher computation |
| Faster inference | Slower inference |
| Better parameter efficiency | More parameters |
| Suitable for deployment | Better for large-scale training |

For a lightweight, deployable prototype, EfficientNet-B0 is the more appropriate choice.

---

## FastAPI

FastAPI was selected over Flask because it provides:

- Higher performance
- Automatic API documentation (Swagger UI)
- Built-in data validation using Pydantic
- Native asynchronous support
- Modern Python type hints

These features make it ideal for AI inference services.

---

## Streamlit

Streamlit enables rapid development of interactive AI applications without requiring extensive frontend expertise.

Benefits include:

- Simple Python interface
- Fast prototyping
- Interactive widgets
- Easy deployment
- Excellent integration with AI workflows

---

## SQLite

SQLite was selected because:

- No separate database server is required.
- Lightweight.
- Easy to deploy.
- Sufficient for single-user applications.
- Minimal configuration.

For larger deployments, SQLite can later be replaced by PostgreSQL or MySQL.

---

## Docker

Docker ensures that the application runs consistently across different operating systems by packaging all dependencies into a portable container.

Benefits include:

- Environment consistency
- Easy deployment
- Dependency isolation
- Reproducibility

---

# 17. High-Level System Architecture

The MedVision AI platform follows a modular client-server architecture.

```
                        User

                          │

                          ▼

               Streamlit Frontend

                          │

                          ▼

                  FastAPI Backend

                          │

          ┌───────────────┼───────────────┐

          ▼               ▼               ▼

 Image Preprocessing   AI Inference   Database Service

          │               │               │

          ▼               ▼               ▼

     EfficientNet-B0   Grad-CAM      SQLite Database

                          │

                          ▼

                  PDF Report Generator

                          │

                          ▼

                     Prediction Result
```

The architecture separates presentation, business logic, AI inference, explainability, and data storage, improving maintainability and scalability.

---

# 18. Software Module Design

The software is divided into independent modules, each with a clearly defined responsibility.

---

## Frontend Module

Responsibilities

- Image upload
- Result visualization
- Display Grad-CAM
- Download reports
- View history

Technology

- Streamlit

---

## Backend Module

Responsibilities

- API endpoints
- Model loading
- Request validation
- Prediction orchestration

Technology

- FastAPI

---

## Preprocessing Module

Responsibilities

- Image loading
- Resize
- Normalization
- Tensor conversion

---

## AI Inference Module

Responsibilities

- Load EfficientNet model
- Perform inference
- Calculate confidence scores

---

## Explainability Module

Responsibilities

- Generate Grad-CAM heatmaps
- Overlay activation maps
- Save explanations

---

## Database Module

Responsibilities

- Store prediction history
- Retrieve previous results

Technology

- SQLite

---

## Reporting Module

Responsibilities

- Generate PDF reports
- Embed prediction results
- Include Grad-CAM visualization

---

# 19. Folder Structure

```
MedVision-AI/

│

├── app/
│   ├── api/
│   ├── preprocessing/
│   ├── models/
│   ├── explainability/
│   ├── database/
│   ├── reporting/
│   └── utils/
│
├── frontend/
│
├── dataset/
│
├── configs/
│
├── docs/
│
├── notebooks/
│
├── outputs/
│
├── tests/
│
├── models/
│
├── requirements.txt
│
├── Dockerfile
│
└── README.md
```

Each directory is responsible for one aspect of the application, supporting modular development and easier maintenance.

---

# 20. Data Flow Diagram (DFD)

The following workflow illustrates the movement of data through the MedVision AI system.

```
User

↓

Upload Chest X-ray

↓

Image Validation

↓

Preprocessing

↓

EfficientNet-B0 Inference

↓

Prediction

↓

Grad-CAM Generation

↓

Confidence Score

↓

Store Prediction

↓

Generate PDF Report

↓

Display Results to User
```

The workflow demonstrates how raw medical images are transformed into explainable diagnostic outputs while maintaining prediction history and report generation.