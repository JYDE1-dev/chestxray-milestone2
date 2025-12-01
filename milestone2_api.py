import io
import os
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
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

# ------------------------------
# Load model
# ------------------------------
MODEL_PATH = "milestone2_mobilenetv2.h5"
model = load_model(MODEL_PATH)  # TF 2.13 compatible

# ------------------------------
# FastAPI setup
# ------------------------------
app = FastAPI(title="Milestone 2 Chest X-ray API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
GRADCAM_DIR = "static/gradcam"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------------------------
# Helper — generate full URL dynamically
# -----------------------------------------
def make_url(request: Request, path: str):
    """Create absolute URL for Render."""
    return f"{request.url.scheme}://{request.url.netloc}/{path}"


# -----------------------------------------
# Root endpoint
# -----------------------------------------
@app.get("/")
def root():
    return {"message": "Milestone 2 API v2 is running successfully!"}


# -----------------------------------------
# Viewer Route (HTML preview for heatmaps)
# -----------------------------------------
@app.get("/viewer")
def view_image(file: str):
    file_path = f"static/gradcam/{file}"
    if not os.path.exists(file_path):
        return HTMLResponse("<h2>File not found</h2>", status_code=404)

    return HTMLResponse(f"""
        <html>
        <body style="text-align:center; font-family: Arial">
            <h2>Grad-CAM Heatmap Viewer</h2>
            <img src="/static/gradcam/{file}" style="max-width:90%; border:2px solid black;">
        </body>
        </html>
    """)


# -----------------------------------------
# Prediction Route
# -----------------------------------------
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_heatmaps: bool = Form(False)
):
    try:
        # Load the incoming image
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Run inference
        preds = model.predict(arr)[0]

        # -------------------------
        # Predictions
        # -------------------------
        final_labels = []
        for idx, score in enumerate(preds):
            if score >= threshold:
                final_labels.append({
                    "label": NIH_LABELS[idx],
                    "score": float(score)
                })

        if len(final_labels) == 0:
            final_labels = [{"label": "No significant abnormality", "score": 0.0}]

        # -------------------------
        # Grad-CAM overlays
        # -------------------------
        heatmap_outputs = []

        if return_heatmaps:
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

            for label, path in overlays:
                filename = os.path.basename(path)

                heatmap_outputs.append({
                    "label": label,
                    "image_url": make_url(request, f"static/gradcam/{filename}"),
                    "viewer": make_url(request, f"viewer?file={filename}"),
                    "html_preview": f"<img src='{make_url(request, f'static/gradcam/{filename}')}' width='300'>"
                })

        return {
            "status": "success",
            "predictions": final_labels,
            "heatmaps": heatmap_outputs
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
