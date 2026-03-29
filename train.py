import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# DATA GENERATOR (OPTIMIZED)
# -----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
val_data = val_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
test_data = val_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

NUM_CLASSES = len(train_data.class_indices)
print("🔥 Classes:", train_data.class_indices)

# -----------------------------
# CLASS WEIGHTS (ONLY IF NEEDED)
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# MODEL (EfficientNetB0)
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

# 🔥 Strong classifier head
x = BatchNormalization()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# -----------------------------
# TRAIN PHASE 1
# -----------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

# -----------------------------
# FINE TUNING
# -----------------------------
print("🔥 Fine-tuning started...")

for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
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

model.save("breed_classifier.h5")

print("🚀 TRAINING COMPLETE")

