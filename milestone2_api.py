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

# ===============================
# IMPORTS FOR GRADCAM
# ===============================
from gradcam_utils import (
    IMG_SIZE,
    NIH_LABELS,
    generate_gradcam_overlays
)

# ===============================
# RENDER-WRITABLE STATIC PATHS
# ===============================
STATIC_RUNTIME_ROOT = "/opt/render/project/src/runtime_static"
GRADCAM_DIR = f"{STATIC_RUNTIME_ROOT}/gradcam"
UPLOAD_DIR = f"{STATIC_RUNTIME_ROOT}/uploads"

os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===============================
# LOAD MODEL (.h5)
# ===============================
MODEL_PATH = "milestone2_mobilenetv2.h5"
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ===============================
# FASTAPI SETUP
# ===============================
app = FastAPI(
    title="Milestone 2 Chest X-ray API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve runtime static content
app.mount("/static", StaticFiles(directory=STATIC_RUNTIME_ROOT), name="static")


@app.get("/")
def root():
    return {
        "message": "Milestone 2 API (Render version) running successfully!",
        "static_path": "/static/gradcam/"
    }


# ===============================
# PREDICT ENDPOINT
# ===============================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_heatmaps: bool = Form(False)
):
    try:
        # -----------------------
        # READ IMAGE
        # -----------------------
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # -----------------------
        # MODEL PREDICTION
        # -----------------------
        preds = model.predict(arr)[0]  # shape = (14,)

        final_labels = []
        for idx, score in enumerate(preds):
            if score >= threshold:
                final_labels.append({
                    "label": NIH_LABELS[idx],
                    "score": float(score)
                })

        if len(final_labels) == 0:
            final_labels = [{"label": "No significant abnormality", "score": 0.0}]

        # -----------------------
        # GENERATE GRADCAM
        # -----------------------
        heatmap_files = []

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

            # Convert internal paths → publicly accessible URLs
            for label, internal_path in overlays:
                filename = os.path.basename(internal_path)
                public_url = f"/static/gradcam/{filename}"
                heatmap_files.append({
                    "label": label,
                    "url": public_url
                })

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
