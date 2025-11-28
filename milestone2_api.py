import io
import os
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

from gradcam_utils import (
    IMG_SIZE,
    NIH_LABELS,
    generate_gradcam_overlays
)

MODEL_PATH = "milestone2_mobilenetv2.h5"
model = load_model(MODEL_PATH)   # works with TF 2.13

app = FastAPI(
    title="Milestone 2 Chest X-ray API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Milestone 2 API (TF 2.13) Running successfully!"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_heatmaps: bool = Form(False)
):
    try:
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]

        final_labels = []
        for idx, score in enumerate(preds):
            if score >= threshold:
                final_labels.append({"label": NIH_LABELS[idx], "score": float(score)})

        if len(final_labels) == 0:
            final_labels = [{"label": "No significant abnormality", "score": 0.0}]

        heatmap_files = []
        if return_heatmaps:
            temp_filename = f"{uuid.uuid4()}.png"
            temp_path = os.path.join(UPLOAD_DIR, temp_filename)
            img.save(temp_path)

            heatmap_files = generate_gradcam_overlays(
                model,
                temp_path,
                preds,
                labels=NIH_LABELS,
                threshold=threshold,
                output_dir="static/gradcam"
            )

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
