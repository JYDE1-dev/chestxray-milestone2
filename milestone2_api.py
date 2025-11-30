import io
import os
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

from gradcam_utils import (
    IMG_SIZE,
    NIH_LABELS,
    generate_gradcam_overlays
)

# ==========================
# CONFIG + MODEL LOAD
# ==========================
MODEL_PATH = "milestone2_mobilenetv2.h5"
model = load_model(MODEL_PATH)   # Works with TF 2.13

app = FastAPI(
    title="Milestone 2 Chest X-ray API",
    version="1.0.0",
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# GradCAM output directory
GRADCAM_DIR = "static/gradcam"
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Expose static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==========================
# ROOT ROUTE
# ==========================
@app.get("/")
def root():
    return {"message": "Milestone 2 API (TF 2.13) Running successfully!"}


# ==========================
# PREDICT ROUTE
# ==========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_heatmaps: bool = Form(False)
):
    try:
        # Read uploaded image
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)

        # Preprocess
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Inference
        preds = model.predict(arr)[0]

        # Multi-label thresholding
        final_labels = [
            {"label": NIH_LABELS[idx], "score": float(score)}
            for idx, score in enumerate(preds)
            if score >= threshold
        ]

        if len(final_labels) == 0:
            final_labels = [{"label": "No significant abnormality", "score": 0.0}]

        # ==========================
        # HEATMAP GENERATION
        # ==========================
        heatmap_files = []

        if return_heatmaps:
            # Save uploaded image temporarily
            temp_filename = f"{uuid.uuid4()}.png"
            temp_path = os.path.join(UPLOAD_DIR, temp_filename)
            img.save(temp_path)

            overlays = generate_gradcam_overlays(
                model,
                temp_path,
                preds,
                labels=NIH_LABELS,
                threshold=threshold,
                output_dir=GRADCAM_DIR
            )

            # Format heatmap URLs for JSON response
            for lbl, path in overlays:
                heatmap_files.append({
                    "label": lbl,
                    "url": f"/static/gradcam/{os.path.basename(path)}"
                })

        # Response
        return JSONResponse(
            content={
                "status": "success",
                "predictions": final_labels,
                "heatmaps": heatmap_files
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
