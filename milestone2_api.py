# milestone2_api.py

import os
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image

from gradcam_utils import (
    NIH_LABELS,
    IMG_SIZE,
    generate_gradcam_overlays,
)

# ============================================
# CONFIG
# ============================================
MODEL_PATH = "milestone2_mobilenetv2.h5"
UPLOAD_DIR = "uploads"
HEATMAP_DIR = "static/gradcam"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

# ============================================
# APP INIT
# ============================================
app = FastAPI(title="Milestone 2 Chest X-ray API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading SavedModel using TFSMLayer...")
model = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
print("Model loaded successfully.")


def preprocess_pil_image(pil_img):
    """Convert PIL image to model-ready numpy array (1, 224, 224, 3)."""
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get("/")
def root():
    return {"message": "Milestone 2 Chest X-Ray API is running."}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.3),
    return_heatmaps: bool = Form(False),
):
    """
    Upload a chest X-ray image and get multi-label predictions.
    Optionally returns Grad-CAM heatmaps.
    """
    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    # Preprocess
    pil_img = Image.open(saved_path)
    arr = preprocess_pil_image(pil_img)

    # Predict
    preds = model.predict(arr)[0]  # shape (14,)
    prob_dict = {label: float(preds[i]) for i, label in enumerate(NIH_LABELS)}

    predicted_labels = [
        label for i, label in enumerate(NIH_LABELS) if preds[i] >= threshold
    ]

    response = {
        "probabilities": prob_dict,
        "predicted_labels": predicted_labels,
        "threshold": threshold,
        "heatmaps": [],
        "image_path": saved_path,
    }

    if return_heatmaps:
        overlays = generate_gradcam_overlays(
            model,
            saved_path,
            preds,
            labels=NIH_LABELS,
            threshold=threshold,
            output_dir=HEATMAP_DIR,
        )
        # return relative paths for client
        heatmap_info = [
            {"label": lbl, "path": path} for (lbl, path) in overlays
        ]
        response["heatmaps"] = heatmap_info

    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("milestone2_api:app", host="0.0.0.0", port=8000, reload=True)


