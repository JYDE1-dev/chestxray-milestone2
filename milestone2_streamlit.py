# milestone2_streamlit.py

import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from gradcam_utils import (
    NIH_LABELS,
    IMG_SIZE,
    generate_gradcam_overlays,
)

MODEL_PATH = "milestone2_mobilenetv2.h5"
HEATMAP_DIR = "static/gradcam"
os.makedirs(HEATMAP_DIR, exist_ok=True)


@st.cache_resource
def load_m2_model():
    return load_model(MODEL_PATH)


def preprocess_pil_image(pil_img):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    st.title("Milestone 2 — Chest X-Ray Anomaly Detection")
    st.write("Multi-label MobileNetV2 model with Grad-CAM visualization.")

    model = load_m2_model()

    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
    threshold = st.slider("Prediction threshold", 0.05, 0.9, 0.3, 0.05)

    if uploaded_file is not None:
        # Show original image
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Prediction"):
            arr = preprocess_pil_image(pil_img)
            preds = model.predict(arr)[0]  # shape (14,)

            prob_dict = {label: float(preds[i]) for i, label in enumerate(NIH_LABELS)}
            pred_labels = [lbl for i, lbl in enumerate(NIH_LABELS) if preds[i] >= threshold]

            st.subheader("Predicted Probabilities")
            st.table(
                [{"Label": lbl, "Probability": f"{prob_dict[lbl]:.3f}"} for lbl in NIH_LABELS]
            )

            st.subheader("Detected Labels (above threshold)")
            if pred_labels:
                st.write(", ".join(pred_labels))
            else:
                st.write("No label above threshold (model predicts 'No Finding').")

            # Save temp image so Grad-CAM util can read from disk
            tmp_path = os.path.join("uploads", "streamlit_temp.png")
            os.makedirs("uploads", exist_ok=True)
            pil_img.save(tmp_path)

            st.subheader("Grad-CAM Heatmaps")
            overlays = generate_gradcam_overlays(
                model,
                tmp_path,
                preds,
                labels=NIH_LABELS,
                threshold=threshold,
                output_dir=HEATMAP_DIR,
            )

            if not overlays:
                st.write("No labels above threshold → no Grad-CAM generated.")
            else:
                for lbl, path in overlays:
                    st.markdown(f"**{lbl}**")
                    st.image(path, use_column_width=True)


if __name__ == "__main__":
    main()
