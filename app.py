from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------- CONFIG ----------
MODEL_PATH = Path("plant_disease_model.keras")
IMG_SIZE = (224, 224)

# NOTE: the model already contains a Rescaling(1/127.5, offset=-1) layer,
# so images are fed in as raw 0-255 floats. Do not normalise here.

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

st.set_page_config(
    page_title="Leaf Diagnosis",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- STYLE: lock the page to one viewport ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --ink:      #0E1F14;
  --moss:     #2F5D3A;
  --leaf:     #3FA34D;
  --alert:    #C2571E;
  --paper:    #EDF1E6;
  --rule:     #C6D0BA;
  --muted:    #63705C;
}

/* Kill every scroll surface */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  height: 100vh !important;
  max-height: 100vh !important;
  overflow: hidden !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
[data-testid="stMainBlockContainer"] {
  padding: 1.1rem 2.2rem 0.6rem 2.2rem !important;
  max-width: 1250px;
  height: 100vh;
  overflow: hidden !important;
}
[data-testid="stVerticalBlock"] { gap: 0.55rem !important; }

[data-testid="stAppViewContainer"] { background: var(--paper); }
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }

/* Masthead */
.masthead {
  display: flex; align-items: baseline; gap: 0.9rem;
  border-bottom: 2px solid var(--ink); padding-bottom: 0.5rem; margin-bottom: 0.2rem;
}
.masthead h1 {
  font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: var(--ink);
}
.masthead .sub {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.09em;
}

/* Compact uploader */
[data-testid="stFileUploaderDropzone"] {
  min-height: 0 !important; padding: 0.55rem 0.9rem !important;
  background: #FFFFFF; border: 1.5px dashed var(--rule); border-radius: 4px;
}
[data-testid="stFileUploaderDropzone"] small { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0.82rem; }
[data-testid="stFileUploader"] label { font-size: 0.72rem !important; color: var(--muted) !important;
  font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.08em; }

/* Specimen frame */
[data-testid="stImage"] img {
  max-height: 46vh; width: auto; object-fit: contain;
  border: 1px solid var(--rule); border-radius: 4px; background: #fff; padding: 6px;
}
[data-testid="stImageCaption"] {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem !important;
  color: var(--muted) !important; letter-spacing: 0.05em;
}

/* Verdict block */
.eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 0.15rem;
}
.species { font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; margin: 0 0 0.7rem 0; }
.verdict {
  font-size: 1.45rem; font-weight: 500; line-height: 1.15; margin: 0 0 0.15rem 0;
  padding-left: 0.75rem; border-left: 5px solid var(--alert); color: var(--alert);
}
.verdict.ok { border-left-color: var(--leaf); color: var(--moss); }

/* Signature: the assay readout */
.assay { margin-top: 1.1rem; border-top: 1px solid var(--rule); padding-top: 0.7rem; }
.assay-row { margin-bottom: 0.55rem; }
.assay-head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: var(--ink); margin-bottom: 0.2rem;
}
.assay-head .pct { font-weight: 600; font-variant-numeric: tabular-nums; }
.assay-row.dim .assay-head { color: var(--muted); font-size: 0.72rem; }
.track { height: 6px; background: #DCE3D3; border-radius: 3px; overflow: hidden; }
.fill { height: 100%; background: var(--moss); }
.assay-row.dim .fill { background: var(--rule); }

.note {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--muted);
  line-height: 1.45; margin-top: 0.85rem; border-left: 2px solid var(--rule); padding-left: 0.6rem;
}

/* Empty state */
.empty { border: 1px dashed var(--rule); border-radius: 4px; padding: 1.6rem 1.4rem; background: #FFFFFF80; }
.empty h3 { font-size: 1rem; font-weight: 500; margin: 0 0 0.5rem 0; }
.empty p { font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: var(--muted);
  line-height: 1.6; margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- MODEL ----------
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(image, dtype=np.float32), axis=0)


def split_label(raw: str):
    plant, condition = raw.split("___")
    plant = plant.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    condition = "Healthy" if condition.lower() == "healthy" else condition[0].upper() + condition[1:]
    return plant, condition


def bar(plant: str, condition: str, pct: float, dim: bool = False) -> str:
    cls = "assay-row dim" if dim else "assay-row"
    return (
        f'<div class="{cls}"><div class="assay-head">'
        f"<span>{plant} — {condition}</span><span class='pct'>{pct:.2f}%</span></div>"
        f'<div class="track"><div class="fill" style="width:{max(pct, 1.2):.2f}%"></div></div></div>'
    )


# ---------- MASTHEAD ----------
st.markdown(
    '<div class="masthead"><h1>Leaf Diagnosis</h1>'
    '<span class="sub">MobileNetV2 · 38 classes · PlantVillage</span></div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model not found. Place `{MODEL_PATH.name}` next to app.py and reload.")
    st.stop()

model = load_model()

left, right = st.columns([1, 1.05], gap="large")

with left:
    uploaded = st.file_uploader(
        "Specimen image", type=["jpg", "jpeg", "png"], label_visibility="visible"
    )
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption=uploaded.name, use_container_width=False)

with right:
    if uploaded is None:
        st.markdown(
            '<div class="empty"><h3>Upload a leaf to run a diagnosis.</h3>'
            "<p>Use a single leaf filling most of the frame, lit evenly, "
            "on a plain background.<br><br>Trained on apple, blueberry, cherry, corn, grape, "
            "orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry "
            "and tomato.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        preds = model.predict(preprocess(image), verbose=0)[0]
        order = np.argsort(preds)[::-1][:3]
        top = int(order[0])
        confidence = float(preds[top]) * 100
        plant, condition = split_label(CLASS_NAMES[top])
        healthy = condition == "Healthy"

        st.markdown(
            f'<div class="eyebrow">Plant</div><div class="species">{plant}</div>'
            f'<div class="eyebrow">Diagnosis</div>'
            f'<div class="verdict{" ok" if healthy else ""}">'
            f'{"No disease detected" if healthy else condition}</div>',
            unsafe_allow_html=True,
        )

        rows = "".join(
            bar(*split_label(CLASS_NAMES[int(i)]), float(preds[int(i)]) * 100, dim=(rank > 0))
            for rank, i in enumerate(order)
        )
        st.markdown(
            f'<div class="assay"><div class="eyebrow">Confidence — top 3</div>{rows}</div>',
            unsafe_allow_html=True,
        )

        if confidence < 60:
            st.markdown(
                '<div class="note">Low confidence. The leaf may belong to a species '
                "outside the 38 trained classes, or the framing may be unclear.</div>",
                unsafe_allow_html=True,
            )
