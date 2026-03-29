import os
import shutil
from sklearn.model_selection import train_test_split

base_path = "dataset/all"
output_base = "dataset"

# delete old splits
for folder in ["train", "val", "test"]:
    path = os.path.join(output_base, folder)
    if os.path.exists(path):
        shutil.rmtree(path)

image_paths = []
labels = []

# breed only
for breed in os.listdir(base_path):
    breed_path = os.path.join(base_path, breed)

    if not os.path.isdir(breed_path):
        continue

    for img in os.listdir(breed_path):
        full_path = os.path.join(breed_path, img)

        if not os.path.isfile(full_path):
            continue

        image_paths.append(full_path)
        labels.append(breed)

print("Total Images:", len(image_paths))
print("Total Classes:", len(set(labels)))

# split
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels,
    test_size=0.3,
    stratify=labels,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=1/3,
    stratify=y_temp,
    random_state=42
)

splits = {
    "train": (X_train, y_train),
    "val": (X_val, y_val),
    "test": (X_test, y_test)
}

# copy
for split in splits:
    X, y = splits[split]

    for img_path, label in zip(X, y):
        target_dir = os.path.join(output_base, split, label)
        os.makedirs(target_dir, exist_ok=True)

        shutil.copy(img_path, os.path.join(target_dir, os.path.basename(img_path)))

print("🔥 DATASET READY")