import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

# -----------------------------
# PATHS
# -----------------------------
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# 🔥 STRONG AUGMENTATION (FIXED)
# -----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(preprocess_input)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

test_data = val_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

NUM_CLASSES = len(train_data.class_indices)
print("🔥 Classes:", train_data.class_indices)

# -----------------------------
# SAVE CLASS MAPPING
# -----------------------------
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# MODEL
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# 🔥 FIX 1 — freeze kam kar
for layer in base_model.layers[:-60]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# -----------------------------
# TRAIN PHASE 1 (🔥 LONGER)
# -----------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,   # 🔥 increased
    callbacks=callbacks,
    class_weight=class_weights
)

# -----------------------------
# 🔥 FINE TUNING (CORRECT)
# -----------------------------
print("🔥 Fine-tuning started...")

# 🔥 only deeper layers train
for layer in base_model.layers[-60:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks
)

# -----------------------------
# TEST
# -----------------------------
loss, acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("breed_classifier.h5")

print("🚀 TRAINING COMPLETE")