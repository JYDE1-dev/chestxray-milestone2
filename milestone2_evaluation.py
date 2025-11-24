import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    f1_score,
    multilabel_confusion_matrix
)

# ==========================================================
# CONFIG
# ==========================================================
ROOT = r"D:\DOCS\WEB PROG\Upwork_Project\smash_downloads\archive"

TEST_CSV = os.path.join(ROOT, "test.csv")

# ⚠ IMPORTANT: NEW MODEL NAME
MODEL_PATH = "milestone2_mobilenetv2.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

NIH_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

print("\n🔍 Milestone 2 — Evaluation Script\n")

# ==========================================================
# LOAD TEST DATA
# ==========================================================
test_df = pd.read_csv(TEST_CSV)

# Fix paths
test_df["file_path"] = test_df["file_path"].str.replace("\\", "/", regex=False)
test_df["full_path"] = test_df["file_path"].apply(lambda x: os.path.join(ROOT, x))

# Convert labels → list
def parse_labels(lbl):
    if lbl == "No Finding" or lbl.strip() == "":
        return []
    return lbl.split("|")

test_df["label_list"] = test_df["label_original"].apply(parse_labels)

# Create 14 binary columns
for lbl in NIH_LABELS:
    test_df[lbl] = 0

for i, row in test_df.iterrows():
    for disease in row["label_list"]:
        if disease in NIH_LABELS:
            test_df.at[i, disease] = 1

y_true = test_df[NIH_LABELS].values

# ==========================================================
# BUILD TEST GENERATOR
# ==========================================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_gen = ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
    test_df,
    x_col="full_path",
    y_col=NIH_LABELS,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode="raw",
    color_mode="rgb"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print("Generating predictions...")
y_pred = model.predict(test_gen, verbose=1)

# ==========================================================
# THRESHOLD
# ==========================================================
THRESHOLD = 0.3
y_pred_bin = (y_pred >= THRESHOLD).astype(int)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================
print("\n📌 Classification Report:\n")

report = classification_report(
    y_true,
    y_pred_bin,
    target_names=NIH_LABELS,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(report).to_csv("milestone2_classification_report.csv")
print("✔ Saved → milestone2_classification_report.csv")

# F1 scores
micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred_bin, average="weighted", zero_division=0)

print(f"\nMicro F1: {micro_f1:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}\n")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================
cm = multilabel_confusion_matrix(y_true, y_pred_bin)

fig, axes = plt.subplots(7, 2, figsize=(12, 28))
axes = axes.ravel()

for i, label in enumerate(NIH_LABELS):
    tn, fp, fn, tp = cm[i].ravel()
    matrix = np.array([[tn, fp], [fn, tp]])

    ax = axes[i]
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(label)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

plt.tight_layout()
plt.savefig("milestone2_confusion_matrix.png")
plt.close()
print("✔ Saved → milestone2_confusion_matrix.png")

# ==========================================================
# SAVE RAW PREDICTIONS
# ==========================================================
pred_df = pd.DataFrame(y_pred, columns=[f"{l}_prob" for l in NIH_LABELS])
final_df = pd.concat([test_df.reset_index(drop=True), pred_df], axis=1)
final_df.to_csv("milestone2_predictions.csv", index=False)

print("✔ Saved → milestone2_predictions.csv")

print("\n🎉 Evaluation Complete!")
