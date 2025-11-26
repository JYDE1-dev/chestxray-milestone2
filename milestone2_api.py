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

# ================================
# IMPORT GRADCAM UTILITIES
# ================================
from gradcam_utils import (
    IMG_SIZE,
    NIH_LABELS as CLASS_NAMES,
    generate_gradcam_overlays
)

# ================================
# MODEL PATH (.keras model)
# ================================
MODEL_PATH = "milestone2_mobilenetv2.keras"

print("Loading .keras model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ================================
# FASTAPI INITIALIZATION
# ================================
app = FastAPI(
    title="Milestone 2 Chest X-ray API",
    version="1.0.0",
)

# Allow all CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# FOLDERS
# ================================
UPLOAD_DIR = "uploads"
GRADCAM_DIR = "static/gradcam"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)


# ================================
# ROOT ROUTE
# ================================
@app.get("/")
def root():
    return {"message": "Milestone 2 API is running successfully!"}


# ================================
# PREDICT ROUTE
# ================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_heatmaps: bool = Form(False)
):
    try:
        # ---------------------------------
        # Save uploaded file temporarily
        # ---------------------------------
        file_bytes = await file.read()
        ext = file.filename.split('.')[-1]
        temp_name = f"{uuid.uuid4()}.{ext}"
        temp_path = os.path.join(UPLOAD_DIR, temp_name)

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # ---------------------------------
        # Preprocess image
        # ---------------------------------
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # ---------------------------------
        # Run inference
        # ---------------------------------
        preds = model.predict(arr)[0]  # shape = (num_classes,)

        # ---------------------------------
        # Apply threshold
        # ---------------------------------
        final_labels = []
        for idx, score in enumerate(preds):
            if score >= threshold:
                final_labels.append({
                    "label": CLASS_NAMES[idx],
                    "score": float(score)
                })

        if len(final_labels) == 0:
            final_labels = [{"label": "No significant abnormality", "score": 0.0}]

        # ---------------------------------
        # Optional: GRAD-CAM
        # ---------------------------------
        heatmap_files = []

        if return_heatmaps:
            overlays = generate_gradcam_overlays(
                model=model,
                image_path=temp_path,
                pred_probs=preds.tolist(),
                labels=CLASS_NAMES,
                threshold=threshold,
                output_dir=GRADCAM_DIR
            )

            # Convert to public URLs
            for label, path in overlays:
                heatmap_files.append({
                    "label": label,
                    "file": f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}/{path.replace(os.sep, '/')}"
                })

        # ---------------------------------
        # RETURN JSON RESPONSE
        # ---------------------------------
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
