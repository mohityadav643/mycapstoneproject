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

# 🔥 CHATBOT CLIENT
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
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

# --------------------------
# HERO SECTION 🔥
# --------------------------
st.markdown('<div class="hero-title">Cattle Breed <span>Detection</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload an image or use URL to detect cattle breed instantly</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 🔥 TABS (NEW)
tab1, tab2 = st.tabs(["🐄 Breed Detection", "🤖 Chatbot"])

# =========================
# 🐄 TAB 1 (YOUR FULL CODE)
# =========================
with tab1:

    # --------------------------
    # 🧠 ABOUT
    # --------------------------
    st.markdown("## 🧠 About This Project")
    st.markdown("""
    <div class="glass">
    AI-based cattle breed detection using deep learning for smart agriculture.
    </div>
    """, unsafe_allow_html=True)

    # --------------------------
    # MAIN LAYOUT
    # --------------------------
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

    # --------------------------
    # 🔥 SMART PREDICTION FUNCTION
    # --------------------------
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

    # --------------------------
    # PREDICTION
    # --------------------------
    if file is not None or url_img is not None:

        st.markdown("## 🔍 Results")

        if file is not None:
            final_img = img
        else:
            final_img = url_img

        result = smart_predict(final_img, "input_image")

        for breed, confidence in result:
            st.write(f"👉 {breed} — {confidence:.2f}% Confidence")
            st.progress(int(confidence))

        final_breed = result[0][0]
        st.success(f"🏆 Final Prediction: {final_breed}")

        if final_breed in breed_info:
            st.markdown("## 📘 Breed Details")
            st.info(breed_info[final_breed])

    # --------------------------
    # ⚙️ FEATURES
    # --------------------------
    st.markdown("## ⚙️ System Features")

    c1, c2, c3 = st.columns(3)
    c1.success("📷 Image Upload Detection")
    c2.success("🌐 URL Based Prediction")
    c3.success("🤖 Deep Learning Model")

    # --------------------------
    # 📂 DATASET INFO
    # --------------------------
    st.markdown("## 📂 Dataset Overview")
    st.markdown(f"""
    <div class="glass">
    Total Classes: {len(class_names)} <br>
    Image Size: 224x224 <br>
    Model: EfficientNetB0
    </div>
    """, unsafe_allow_html=True)

    # --------------------------
    # 📊 MODEL PERFORMANCE
    # --------------------------
    if metrics:
        st.markdown("## 📊 Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        col4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

# =========================
# 🤖 TAB 2 (CHATBOT)
# =========================
with tab2:

    st.markdown("## 🤖 Smart Cattle Assistant")

    user_input = st.text_input("Ask anything about cattle...")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-3.5-turbo",
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