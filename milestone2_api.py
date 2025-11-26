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
    CLASS_NAMES,
    generate_gradcam_overlays
)

# ========================
# LOAD SAVEDMODEL USING TFSMLayer
# ========================
MODEL_PATH = "milestone2_mobilenetv2_savedmodel"

print("Loading SavedModel using TFSMLayer...")
model = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
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

# Directory for uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory for GradCAM results
GRADCAM_DIR = "static/gradcam"
os.makedirs(GRADCAM_DIR, exist_ok=True)


# =======================================
# ROOT ROUTE
# =======================================
@app.get("/")
def root():
    return {"message": "Milestone 2 API is running successfully!"}


# =======================================
# PREDICT ROUTE
# =======================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_heatmaps: bool = Form(False)
):
    try:
        # -----------------------------
        # READ FILE
        # -----------------------------
        file_bytes = await file.read()
        img = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB").resize(IMG_SIZE)

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # -----------------------------
        # RUN INFERENCE (NO .predict)
        # -----------------------------
        output_dict = model(arr, training=False)

        # Take output from the dict
        output_tensor = list(output_dict.values())[0]
        preds = output_tensor.numpy()[0]  # shape = (num_classes,)

        # -----------------------------
        # THRESHOLD MULTILABEL LOGIC
        # -----------------------------
        final_labels = []
        for idx, score in enumerate(preds):
            if score >= threshold:
                final_labels.append({"label": CLASS_NAMES[idx], "score": float(score)})

        if len(final_labels) == 0:
            final_labels = [{"label": "No significant abnormality", "score": 0.0}]

        # -----------------------------
        # OPTIONAL: GRAD-CAM OVERLAYS
        # -----------------------------
        heatmap_files = []
        if return_heatmaps:
            heatmap_files = generate_gradcam_overlays(
                model=model,
                image_array=arr,
                class_names=CLASS_NAMES,
                save_dir=GRADCAM_DIR
            )

        return JSONResponse(
            content={
                "status": "success",
                "predictions": final_labels,
                "heatmaps": heatmap_files,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

