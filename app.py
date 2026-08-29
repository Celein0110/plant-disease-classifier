import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- CONFIG ----------
MODEL_PATH = "plant_disease_model.keras"   # rename your model file to this, or update this path
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

# ---------- PAGE ----------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.write(
    "Upload a photo of a plant leaf and this app will predict the plant species "
    "and whether it's healthy or affected by a disease, using a MobileNetV2 model "
    "fine-tuned on the PlantVillage dataset (38 classes, 96.48% test accuracy)."
)


# ---------- LOAD MODEL (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ---------- PREPROCESSING ----------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    # Same preprocessing used during training: Rescaling(1./127.5, offset=-1)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def format_label(raw_label: str):
    plant, condition = raw_label.split("___")
    plant = plant.replace("_", " ")
    condition = condition.replace("_", " ")
    return plant, condition


# ---------- UI ----------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing leaf..."):
        processed = preprocess_image(image)
        predictions = model.predict(processed)[0]
        top_index = int(np.argmax(predictions))
        confidence = float(predictions[top_index]) * 100
        plant, condition = format_label(CLASS_NAMES[top_index])

    st.subheader("Result")
    st.write(f"**Plant:** {plant}")
    if condition.lower() == "healthy":
        st.success(f"**Status:** Healthy ✅ (confidence: {confidence:.2f}%)")
    else:
        st.error(f"**Status:** {condition} ⚠️ (confidence: {confidence:.2f}%)")

    # Show top-3 predictions for transparency
    st.subheader("Top 3 predictions")
    top3_indices = np.argsort(predictions)[-3:][::-1]
    for idx in top3_indices:
        p, c = format_label(CLASS_NAMES[idx])
        st.write(f"- {p} — {c}: {predictions[idx] * 100:.2f}%")
else:
    st.info("Upload a leaf image above to get a prediction.")
