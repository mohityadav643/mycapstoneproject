from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Class names (same order as training)
class_names = [
    "Gir", "Sahiwal", "Kankrej", "Ongole", "Red_Sindhi", "Rathi", "Nagori"
]

@app.route("/")
def home():
    return "API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    img = Image.open(file.stream).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    best = np.argmax(preds)

    return jsonify({
        "breed": class_names[best],
        "confidence": float(preds[best] * 100)
    })

if __name__ == "__main__":
    app.run(debug=True)