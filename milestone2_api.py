import io
import os
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ========================
# IMPORT GRADCAM UTILS
# ========================
from gradcam_utils import (
    IMG_SIZE,
    NIH_LABELS,
    generate_gradcam_overlays
)

CLASS_NAMES = NIH_LABELS

# ========================
# LOAD .keras MODEL
# ========================
MODEL_PATH = "milestone2_mobilenetv2.keras"

print("Loading .keras model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ========================
# FASTAPI APP
# ========================
app = FastAPI(
    title="Milestone 2 Chest X-ray API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for uploads & heatmaps
UPLOAD_DIR = "uploads"
GRADCAM_DIR = "static/gradcam"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Milestone 2 API is running successfully!"}


# =======================================
# PREDICT ENDPOINT
# =======================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_heatmaps: bool = Form(False)
):
    try:
        # Save incoming file
        file_bytes = await file.read()
        temp_filename = f"{uuid.uuid4()}.png"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Prepare image
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Predict
        preds = model.predict(arr)[0].tolist()

        # Thresholding
        final_labels = []
        for i, p in enumerate(preds):
            if p >= threshold:
                final_labels.append({"label": CLASS_NAMES[i], "score": float(p)})

        if not final_labels:
            final_labels = [{"label": "No significant abnormality", "score": 0.0}]

        # Grad-CAM
        heatmap_files = []
        if return_heatmaps:
            overlays = generate_gradcam_overlays(
                model=model,
                image_path=temp_path,
                pred_probs=preds,
                labels=CLASS_NAMES,
                threshold=threshold,
                output_dir=GRADCAM_DIR
            )

            for lbl, path in overlays:
                heatmap_files.append({
                    "label": lbl,
                    "file": path.replace("\\", "/")
                })

        return JSONResponse({
            "status": "success",
            "predictions": final_labels,
            "heatmaps": heatmap_files
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
