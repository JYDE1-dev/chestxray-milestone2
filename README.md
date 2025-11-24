# Milestone 2: Model Refinement & Deployment

## Chest X-Ray Multi-Label Anomaly Detection (MobileNetV2 + Grad-CAM + FastAPI + Streamlit)
* Author: Alayande Olajide
* Milestone: 2 – Model Refinement & Deployment
* Client: Paavanan Vellan
* Date: 23-11-25


## Overview

Milestone 2 focuses on refining the chest X-ray anomaly detection model, enabling incremental training, building full evaluation metrics, integrating explainability (Grad-CAM), and deploying the model to a cloud-hosted API for real-time inference.

This milestone also includes a standalone Streamlit demo for easy visual testing of the model.

## Milestone 2 Deliverables

### 1. Model Refinement

Multi-label MobileNetV2 model trained on NIH ChestX-ray subset (8,000 images).

Class weighting applied to handle dataset imbalance.

Fine-tuning of deeper layers for improved learning.

Fully modular training pipeline to allow incremental updates in later milestones.

### 2. Evaluation

Evaluation script produces:

Micro F1

Macro F1

Weighted F1

Per-class confusion matrices

Prediction CSV

Classification report

Outputs saved in:

milestone2_classification_report.csv

milestone2_confusion_matrix.png

milestone2_predictions.csv

### 3. Explainability (Grad-CAM)

Grad-CAM overlays generated per disease label above threshold.

Shared utility module: gradcam_utils.py

Heatmaps saved inside: static/gradcam/

### 4. Deployment

Two deployment components:

FastAPI Backend

Endpoint: /predict

Returns disease probabilities, predicted labels, and optional Grad-CAM heatmaps.

Cloud-ready (Render deployment).

Streamlit Demo UI

Upload image and receive:

Predictions (sorted)

Detected labels

Grad-CAM explanations


## How to Run Locally
### 1. Activate Virtual Environment
cd work_Project
.\workvenv\Scripts\activate

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run FastAPI
uvicorn milestone2_api:app --reload


Open:

http://127.0.0.1:8000/docs

### 4. Run Streamlit UI
streamlit run milestone2_streamlit.py
