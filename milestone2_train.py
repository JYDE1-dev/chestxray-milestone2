import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =======================================================
# CONFIG
# =======================================================
ROOT = r"D:\DOCS\WEB PROG\Upwork_Project\smash_downloads\archive"
TRAIN_CSV = os.path.join(ROOT, "train.csv")

MODEL_PATH = "milestone2_mobilenetv2.h5"
BINARIZER_SAVE = "label_binarizer.npy"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

FREEZE_EPOCHS = 12
FINETUNE_EPOCHS = 8

# NIH 14 labels
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

print("\n⚡ Milestone 2 Training — MobileNetV2 (Fast + Multi-Label)\n")

# =======================================================
# LOAD CSV + CLEAN PATHS
# =======================================================
df = pd.read_csv(TRAIN_CSV)
df["file_path"] = df["file_path"].str.replace("\\", "/", regex=False)
df["full_path"] = df["file_path"].apply(lambda x: os.path.join(ROOT, x))
df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

# =======================================================
# LIMIT DATASET FOR SPEED (8k IMAGES)
# =======================================================
df = df.sample(8000, random_state=42).reset_index(drop=True)
print("Using reduced dataset:", len(df), "images")

# =======================================================
# MULTI-LABEL ENCODING
# =======================================================
def parse_labels(lbl):
    if lbl == "No Finding" or lbl.strip() == "":
        return []
    return lbl.split("|")

df["label_list"] = df["label_original"].apply(parse_labels)

mlb = MultiLabelBinarizer(classes=NIH_LABELS)
y = mlb.fit_transform(df["label_list"])

np.save(BINARIZER_SAVE, mlb.classes_)

# =======================================================
# TRAIN–VAL SPLIT
# =======================================================
train_df, val_df, y_train, y_val = train_test_split(
    df, y, test_size=0.2, random_state=42, shuffle=True
)

# Add binary columns
for lbl in NIH_LABELS:
    train_df[lbl] = 0
    val_df[lbl] = 0

train_df[NIH_LABELS] = y_train
val_df[NIH_LABELS] = y_val

# =======================================================
# CLASS WEIGHTS
# =======================================================
pos_counts = y_train.sum(axis=0)
neg_counts = y_train.shape[0] - pos_counts
class_weights = {i: (neg_counts[i] / (pos_counts[i] + 1e-6)) for i in range(14)}

print("\nComputed class weights:", class_weights)

# =======================================================
# IMAGE GENERATORS
# =======================================================
train_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=5,
    zoom_range=0.1,
    horizontal_flip=True
).flow_from_dataframe(
    train_df,
    x_col="full_path",
    y_col=NIH_LABELS,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
    color_mode="rgb"
)

val_gen = ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
    val_df,
    x_col="full_path",
    y_col=NIH_LABELS,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
    color_mode="rgb"
)

# =======================================================
# BUILD MODEL — MobileNetV2
# =======================================================
base = MobileNetV2(
    include_top=False,
    input_tensor=Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    weights="imagenet"
)

# Freeze base layers
for layer in base.layers:
    layer.trainable = False

# Classifier head
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
output = Dense(14, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy"
)

model.summary()

# =======================================================
# CALLBACKS
# =======================================================
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1
)

early = EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

# =======================================================
# PHASE 1 — TRAIN FROZEN
# =======================================================
print("\n🔵 Phase 1 — Training Frozen MobileNetV2...\n")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FREEZE_EPOCHS,
    callbacks=[checkpoint, lr_reduce, early],
    class_weight=class_weights,
    verbose=1
)

# =======================================================
# PHASE 2 — FINE-TUNE DEEPER LAYERS
# =======================================================
print("\n🟠 Phase 2 — Fine-Tuning MobileNetV2...\n")

for layer in base.layers[-80:]:  
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy"
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINETUNE_EPOCHS,
    callbacks=[checkpoint, lr_reduce, early],
    class_weight=class_weights,
    verbose=1
)

print("\n🎉 Training Finished! Best model saved as:", MODEL_PATH)
