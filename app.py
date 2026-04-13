import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import base64
import json

from io import BytesIO
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input
from openai import OpenAI   # ✅ NEW


# 🔥 CHATBOT CLIENT (FINAL FIXED - NOTHING REMOVED)
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Cattle Breed Detection"
    }
)

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Breed Detection", layout="wide")

# --------------------------
# BACKGROUND + HERO STYLE 🔥
# --------------------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = get_base64("bg.jpg")

st.markdown(f"""
<style>
.stApp {{
    background:
    linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.7)),
    url("data:image/jpg;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.hero-title {{
    font-size: 65px;
    font-weight: 800;
    text-align: center;
    color: white;
}}

.hero-title span {{
    color: #22c55e;
}}

.hero-sub {{
    text-align: center;
    color: #d1d5db;
    font-size: 20px;
    margin-bottom: 30px;
}}

.glass {{
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
}}

.stButton>button {{
    background: linear-gradient(45deg, #22c55e, #16a34a);
    color: white;
    border-radius: 12px;
    height: 45px;
    font-size: 16px;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL
# --------------------------
model = tf.keras.models.load_model("breed_classifier.h5")
class_names = sorted(os.listdir("dataset/train"))

# --------------------------
# LOAD MAPPING
# --------------------------
mapping = {}
if os.path.exists("mapping.json"):
    with open("mapping.json") as f:
        mapping = json.load(f)

# --------------------------
# LOAD METRICS
# --------------------------
metrics = {}
if os.path.exists("metrics.json"):
    with open("metrics.json") as f:
        metrics = json.load(f)

# --------------------------
# 🔥 ALL BREEDS INFO
# --------------------------
breed_info = {
    "Amritmahal": "Amritmahal cattle originate from Karnataka and are known for strength and endurance.",
    "Ayrshire": "Ayrshire is a dairy breed from Scotland known for high milk yield.",
    "Bargur": "Bargur cattle are agile and used in hilly regions.",
    "bhadwari": "Bhadawari buffalo is known for high fat milk.",
    "Chhattisgarhi": "Hardy cattle from central India used in agriculture.",
    "Dangi": "Dangi cattle adapt well to heavy rainfall areas.",
    "Deoni": "Dual-purpose breed for milk and draught.",
    "Gir": "One of the best dairy breeds with high milk yield.",
    "Hallikar": "Strong draught cattle from Karnataka.",
    "Jaffarabadi": "Large buffalo breed with high milk yield.",
    "Kangayam": "Strong and disease-resistant draught cattle.",
    "Kankrej": "Dual-purpose breed known for strength.",
    "malvi": "Hardy cattle from Madhya Pradesh.",
    "murrah": "High milk yielding buffalo breed.",
    "nagori": "Fast and strong draught cattle.",
    "nagpuri": "Adaptable buffalo breed with moderate milk.",
    "Ongole": "Muscular and disease-resistant breed from Andhra Pradesh.",
    "Rathi": "Good dairy breed from Rajasthan.",
    "Red Sindhi": "High milk producing breed suitable for tropical climates.",
    "Sahiwal": "Top dairy breed with heat tolerance.",
    "surti": "Buffalo breed known for moderate milk."
}

# 🌍 NEW CODE: Breed Location Mapping
breed_location = {
    "Amritmahal": {"lat": 12.97, "lon": 77.59, "place": "Karnataka"},
    "Ayrshire": {"lat": 55.46, "lon": -4.63, "place": "Scotland"},
    "Bargur": {"lat": 11.55, "lon": 77.45, "place": "Tamil Nadu"},
    "bhadwari": {"lat": 26.50, "lon": 78.63, "place": "UP / MP"},
    "Chhattisgarhi": {"lat": 21.25, "lon": 81.63, "place": "Chhattisgarh"},
    "Dangi": {"lat": 20.59, "lon": 73.78, "place": "Maharashtra"},
    "Deoni": {"lat": 18.28, "lon": 76.62, "place": "Maharashtra"},
    "Gir": {"lat": 21.12, "lon": 70.82, "place": "Gujarat"},
    "Hallikar": {"lat": 13.34, "lon": 77.10, "place": "Karnataka"},
    "Jaffarabadi": {"lat": 21.35, "lon": 72.13, "place": "Gujarat"},
    "Kangayam": {"lat": 11.00, "lon": 77.56, "place": "Tamil Nadu"},
    "Kankrej": {"lat": 23.59, "lon": 72.37, "place": "Gujarat"},
    "malvi": {"lat": 23.25, "lon": 77.41, "place": "Madhya Pradesh"},
    "murrah": {"lat": 28.70, "lon": 76.99, "place": "Haryana"},
    "nagori": {"lat": 27.20, "lon": 73.73, "place": "Rajasthan"},
    "nagpuri": {"lat": 21.15, "lon": 79.09, "place": "Maharashtra"},
    "Ongole": {"lat": 15.50, "lon": 80.05, "place": "Andhra Pradesh"},
    "Rathi": {"lat": 28.02, "lon": 73.31, "place": "Rajasthan"},
    "Red Sindhi": {"lat": 25.40, "lon": 68.35, "place": "Sindh"},
    "Sahiwal": {"lat": 30.67, "lon": 73.11, "place": "Punjab"},
    "surti": {"lat": 22.30, "lon": 73.20, "place": "Gujarat"}
}

# --------------------------
# HERO SECTION 🔥
# --------------------------
st.markdown('<div class="hero-title">Cattle Breed <span>Detection</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload an image or use URL to detect cattle breed instantly</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 🔥 TABS
tab1, tab2 = st.tabs(["🐄 Breed Detection", "🤖 Chatbot"])

# =========================
# 🐄 TAB 1
# =========================
with tab1:

    st.markdown("## 🧠 About This Project")
    st.markdown("""
    <div class="glass">
    AI-based cattle breed detection using deep learning for smart agriculture.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])
        url = st.text_input("🌐 Or paste Image URL")

        url_img = None

        if url:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                url_img = Image.open(BytesIO(response.content)).convert("RGB")
                url_img = url_img.copy()
                st.image(url_img, width=450)

            except:
                st.error("❌ Invalid URL or cannot load image")

        if file is not None:
            img = Image.open(file).convert("RGB")
            st.image(img, width=450)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.markdown("### 📌 Guidelines")
        st.write("• Use clear cattle image")
        st.write("• Full body preferred")
        st.write("• Avoid blurry photos")

        st.markdown('</div>', unsafe_allow_html=True)

    def smart_predict(img, filename):

        img_resized = img.resize((224,224))
        img_array = np.array(img_resized).astype(np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        top3_idx = preds.argsort()[-3:][::-1]

        results = []
        for i in top3_idx:
            confidence = preds[i]*100
            results.append((class_names[i], confidence))

        return results

    if file is not None or url_img is not None:

        st.markdown("## 🔍 Results")

        final_img = img if file is not None else url_img

        result = smart_predict(final_img, "input_image")

        for breed, confidence in result:
            st.write(f"👉 {breed} — {confidence:.2f}% Confidence")
            st.progress(int(confidence))

        final_breed = result[0][0]
        st.success(f"🏆 Final Prediction: {final_breed}")

        if final_breed in breed_info:
            st.markdown("## 📘 Breed Details")
            st.info(breed_info[final_breed])

        # 🌍 NEW CODE: MAP DISPLAY
        if final_breed in breed_location:
            loc = breed_location[final_breed]

            st.markdown("## 🗺️ Breed Origin Map")
            st.write(f"📍 Found in: {loc['place']}")

            map_data = {
                "lat": [loc["lat"]],
                "lon": [loc["lon"]]
            }

            st.map(map_data)
        else:
            st.warning("Location data not available")

    st.markdown("## ⚙️ System Features")

    c1, c2, c3 = st.columns(3)
    c1.success("📷 Image Upload Detection")
    c2.success("🌐 URL Based Prediction")
    c3.success("🤖 Deep Learning Model")

    st.markdown("## 📂 Dataset Overview")
    st.markdown(f"""
    <div class="glass">
    Total Classes: {len(class_names)} <br>
    Image Size: 224x224 <br>
    Model: EfficientNetB0
    </div>
    """, unsafe_allow_html=True)

    if metrics:
        st.markdown("## 📊 Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        col4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

# =========================
# 🤖 TAB 2
# =========================
with tab2:

    st.markdown("## 🤖 Smart Cattle Assistant")

    user_input = st.text_input("Ask anything about cattle...")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="meta-llama/llama-3-8b-instruct",
                    messages=[
                        {"role": "system", "content": "You are an expert in cattle breeds."},
                        {"role": "user", "content": user_input}
                    ]
                )
                st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(str(e))

# --------------------------
# FOOTER
# --------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<center style='color:gray;'>Capstone Project - Mohit Yadav</center>", unsafe_allow_html=True)