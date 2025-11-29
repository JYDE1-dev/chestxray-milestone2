# gradcam_utils.py

import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# NIH 14 labels (same order as training)
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
CLASS_NAMES = NIH_LABELS
IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    """Load and preprocess an image for the model (MobileNetV2)."""
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


def _find_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


class GradCAM:
    def __init__(self, model, target_layer_name=None):
        self.model = model
        if target_layer_name is None:
            target_layer_name = _find_last_conv_layer(model)
        self.target_layer_name = target_layer_name

        self.grad_model = Model(
            [self.model.inputs],
            [self.model.get_layer(self.target_layer_name).output, self.model.output]
        )

    def compute_heatmap(self, img_array, class_idx):
        """
        Compute Grad-CAM heatmap for a single image and a single class index.
        img_array: shape (1, H, W, 3)
        """
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(img_array)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out.numpy()[0]  # (H, W, C)
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[0]):
            conv_out[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_out, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-8
        heatmap /= max_val

        return heatmap  # (H, W), values [0,1]

    def overlay_heatmap(self, heatmap, original_image_path, alpha=0.4):
        """
        Overlay heatmap onto the original image.
        Returns BGR image suitable for cv2.imwrite.
        """
        img = cv2.imread(original_image_path)
        img = cv2.resize(img, IMG_SIZE)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

        return overlay


def generate_gradcam_overlays(
    model,
    image_path,
    pred_probs,
    labels=NIH_LABELS,
    threshold=0.3,
    output_dir="static/gradcam"
):
    """
    Generate Grad-CAM overlay images for all labels with prob >= threshold.

    Returns:
        List of (label, saved_path) tuples.
    """
    os.makedirs(output_dir, exist_ok=True)
    cam = GradCAM(model)

    img_array = preprocess_image(image_path)
    overlays = []

    for i, p in enumerate(pred_probs):
        if p >= threshold:
            label = labels[i]
            heatmap = cam.compute_heatmap(img_array, i)
            overlay = cam.overlay_heatmap(heatmap, image_path)

            # Save file
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_name = f"{base_name}_gradcam_{label}.png"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, overlay)

            overlays.append((label, save_path))

    return overlays

