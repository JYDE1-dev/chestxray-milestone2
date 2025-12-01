import io
import os
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

from gradcam_utils import (
    IMG_SIZE,
    NIH_LABELS,
    generate_gradcam_overlays
)

# -------------------------------
# Load model (H5 works with TF 2.13)
# -------------------------------
MODEL_PATH = "milestone2_mobilenetv2.h5"
model = load_model(MODEL_PATH)

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="Milestone 2 Chest X-ray API",
    version="1.1.0",
    description="Enhanced Swagger UI with heatmap thumbnails & viewer links"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
GRADCAM_DIR = "static/gradcam"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Static files mount
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------
# Viewer: Full-screen heatmap page
# ---------------------------------
@app.get("/viewer")
def viewer(file: str):
    file_url = f"/static/gradcam/{file}"
    html = f"""
    <html>
    <body style="background:#111; text-align:center; padding:30px;">
        <h2 style="color:white;">Heatmap Viewer</h2>
        <img src="{file_url}" style="max-width:95%; border:3px solid #0f0;">
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/")
def root():
    return {"message": "Milestone 2 API (Enhanced Swagger UI) Running!"}


# ---------------------------------
# Predict Route
# ---------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_heatmaps: bool = Form(False)
):
    try:
        # Read file and preprocess
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]

        # Predictions
        final_labels = [
            {"label": NIH_LABELS[i], "score": float(p)}
            for i, p in enumerate(preds) if p >= threshold
        ]

        if not final_labels:
            final_labels = [{"label": "No significant abnormality", "score": 0.0}]

        # ---------------------------
        # Heatmaps
        # ---------------------------
        heatmap_list = []
        if return_heatmaps:
            temp_id = str(uuid.uuid4()) + ".png"
            temp_path = os.path.join(UPLOAD_DIR, temp_id)
            img.save(temp_path)

            overlays = generate_gradcam_overlays(
                model=model,
                image_path=temp_path,
                pred_probs=preds,
                labels=NIH_LABELS,
                threshold=threshold,
                output_dir=GRADCAM_DIR
            )

            for label, saved_path in overlays:
                filename = os.path.basename(saved_path)

                heatmap_list.append({
                    "label": label,
                    "image_url": f"/static/gradcam/{filename}",
                    "viewer": f"/viewer?file={filename}",
                    "thumbnail_html": f"<img src='/static/gradcam/{filename}' width='250'>"
                })

        return JSONResponse(content={
            "status": "success",
            "predictions": final_labels,
            "heatmaps": heatmap_list
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })
