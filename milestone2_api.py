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
    NIH_LABELS,       # Replace CLASS_NAMES -> NIH_LABELS (your utils uses this)
    generate_gradcam_overlays
)

CLASS_NAMES = NIH_LABELS  # Alias for clarity

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
        # SAVE UPLOADED FILE
        # -----------------------------
        unique_name = f"{uuid.uuid4().hex}.png"
        temp_path = os.path.join(UPLOAD_DIR, unique_name)

        file_bytes = await file.read()
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # -----------------------------
        # IMAGE PREPROCESSING
        # -----------------------------
        img = Image.open(temp_path).convert("RGB").resize(IMG_SIZE)
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
        # OPTIONAL: GENERATE GRAD-CAM
        # -----------------------------
        heatmap_files = []
        if return_heatmaps:
            overlays = generate_gradcam_overlays(
                model,
                temp_path,             # REQUIRED image_path
                preds.tolist(),        # prediction vector
                labels=CLASS_NAMES,
                threshold=threshold,
                output_dir=GRADCAM_DIR
            )

            # overlays = list of (label, filepath)
            heatmap_files = [
                {"label": label, "url": f"/static/gradcam/{os.path.basename(path)}"}
                for label, path in overlays
            ]

        # -----------------------------
        # RETURN SUCCESS
        # -----------------------------
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
            content={
                "status": "error",
                "message": str(e)
            }
        )
