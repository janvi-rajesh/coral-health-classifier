import os
import time

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
from tensorflow.keras.applications.resnet import preprocess_input as resnet_pre

# ------------------------------------------------------------------
# CONFIG - change the paths to match your saved model files
# ------------------------------------------------------------------
MODELS = {
    "DenseNet121": {"path": "DenseNet121_model.h5", "pre": densenet_pre, "color": "#3b82f6"},
    "ResNet152": {"path": "ResNet.h5", "pre": resnet_pre, "color": "#f97316"},
}
# Alphabetical order = flow_from_directory order used in training
CLASSES = ["Bleached Coral", "Dead Coral", "Healthy Coral"]
CLASS_COLORS = {"Bleached Coral": "#facc15", "Dead Coral": "#ef4444", "Healthy Coral": "#22c55e"}
IMG_SIZE = (224, 224)
REPO_ID = "janvi-rajesh/coral-health-models"  # HF model repo holding the .h5 files


def get_path(filename):
    """Use the local file if present (laptop), else download from the HF model repo (Space)."""
    if os.path.exists(filename):
        return filename
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


st.set_page_config(page_title="Coral Health Classifier", page_icon="🪸", layout="wide")

# ------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(180deg, #06121a 0%, #0a1f2b 100%); color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #050d13; border-right: 1px solid #12303a; }
header[data-testid="stHeader"] { background: transparent; }
.app-title { text-align: center; font-size: 3rem; font-weight: 700; margin: 0;
  background: linear-gradient(90deg, #2dd4bf, #22d3ee); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.app-sub { text-align: center; color: #64808c; font-size: .85rem; margin-bottom: 1.5rem; }
.side-h { color: #94a3b8; font-size: .7rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; margin: 1rem 0 .4rem; }
.legend-item { display: flex; align-items: center; gap: .5rem; font-weight: 600; font-size: .85rem; margin: .3rem 0; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.chip { display: inline-block; background: #0d2a33; border: 1px solid #1b4a57; color: #9fd8e0;
  border-radius: 999px; padding: 2px 12px; font-size: .72rem; margin: 0 6px 6px 0; }
.card { background: #0b2630; border: 1px solid #14444f; border-radius: 12px; padding: 1rem 1.2rem; height: 100%; }
.card .lbl { color: #5f8794; font-size: .65rem; letter-spacing: .1em; text-transform: uppercase; }
.card .big { font-size: 1.9rem; font-weight: 700; color: #e6fffb; margin-top: .2rem; }
.card .cls { font-size: 1.05rem; font-weight: 600; margin-top: .3rem; }
.model-h { font-weight: 600; font-size: 1.05rem; margin: .5rem 0; display: flex; align-items: center; gap: .5rem; }
.bar-row { display: flex; align-items: center; gap: .5rem; font-size: .7rem; color: #9fb7c0; margin-top: .35rem; }
.bar-bg { flex: 1; background: #0a1c24; border-radius: 6px; height: 6px; overflow: hidden; }
.bar-fg { height: 6px; border-radius: 6px; }
.banner { border-radius: 10px; padding: .8rem 1rem; margin-top: 1rem; font-weight: 600; }
.banner.ok { background: #0d3b2a; border: 1px solid #1f7a54; color: #6ee7b7; }
.banner.no { background: #3b2a0d; border: 1px solid #7a5a1f; color: #fcd34d; }
hr { border-color: #12303a; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return tf.keras.models.load_model(get_path(path), compile=False)


def predict(name, img: Image.Image):
    cfg = MODELS[name]
    model = load_model(cfg["path"])
    x = np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    x = cfg["pre"](np.expand_dims(x, 0))  # model-specific preprocessing (important!)
    t0 = time.time()
    probs = model.predict(x, verbose=0)[0]
    return probs, time.time() - t0


def dot(color):
    return f'<span class="dot" style="background:{color}"></span>'


def bars(probs):
    rows = ""
    for c, p in zip(CLASSES, probs):
        rows += (
            f'<div class="bar-row"><span style="width:90px">{c.split()[0]}</span>'
            f'<div class="bar-bg"><div class="bar-fg" style="width:{p*100:.1f}%;background:{CLASS_COLORS[c]}"></div></div>'
            f'<span>{p*100:.1f}%</span></div>'
        )
    return rows


def result_card(probs, secs):
    i = int(np.argmax(probs))
    c = CLASSES[i]
    return (
        f'<div class="card"><div class="lbl">Predicted class</div>'
        f'<div class="cls">{dot(CLASS_COLORS[c])} {c}</div>'
        f'<div class="big">{probs[i]*100:.2f}%</div>'
        f'<div class="lbl">⏱ {secs:.3f}s inference</div>{bars(probs)}</div>'
    ), c


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙ Configuration")
    st.markdown('<div class="side-h">Select Mode</div>', unsafe_allow_html=True)
    mode = st.radio("mode", ["Single Model", "Compare Both Models"], label_visibility="collapsed")

    if mode == "Single Model":
        st.markdown('<div class="side-h">Select Model</div>', unsafe_allow_html=True)
        choice = st.selectbox("model", list(MODELS), label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="side-h">Classes</div>', unsafe_allow_html=True)
    for c in CLASSES:
        st.markdown(f'<div class="legend-item" style="color:{CLASS_COLORS[c]}">{dot(CLASS_COLORS[c])}{c}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="side-h">Models</div>', unsafe_allow_html=True)
    for n, cfg in MODELS.items():
        st.markdown(f'<div class="legend-item" style="font-weight:400;color:#94a3b8">✅ {cfg["path"]}</div>',
                    unsafe_allow_html=True)

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
st.markdown('<h1 class="app-title">🪸 Coral Health Classifier</h1>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Coral reef health detection using DenseNet121 &amp; ResNet152</div>',
            unsafe_allow_html=True)

up = st.file_uploader("Upload a coral image", type=["jpg", "jpeg", "png"])

if up is None:
    st.markdown(
        '<div style="text-align:center;margin-top:3rem;color:#5f8794">'
        '<div style="font-size:3rem">🪸</div>Upload a coral image to begin classification<br>'
        '<small>Detects: Healthy Coral · Bleached Coral · Dead Coral</small></div>',
        unsafe_allow_html=True,
    )
    st.stop()

img = Image.open(up)
c1, c2 = st.columns([1, 2])
with c1:
    st.image(img, caption="Uploaded image", use_container_width=True)
with c2:
    st.markdown("#### 🖼 Image Info")
    st.markdown(
        f'<span class="chip">📄 {up.name}</span><span class="chip">📐 {img.width}×{img.height} px</span>'
        f'<span class="chip">🎨 {img.mode}</span><span class="chip">💾 {up.size/1024:.0f} KB</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

try:
    if mode == "Single Model":
        st.markdown(f"#### 🔍 Prediction — {choice}")
        probs, secs = predict(choice, img)
        i = int(np.argmax(probs))
        c = CLASSES[i]
        a, b, d = st.columns(3)
        a.markdown(f'<div class="card"><div class="lbl">Predicted class</div>'
                   f'<div class="cls" style="margin-top:.8rem">{dot(CLASS_COLORS[c])} {c}</div></div>',
                   unsafe_allow_html=True)
        b.markdown(f'<div class="card"><div class="lbl">Confidence</div><div class="big">{probs[i]*100:.2f}%</div></div>',
                   unsafe_allow_html=True)
        d.markdown(f'<div class="card"><div class="lbl">Inference time</div><div class="big">{secs:.3f}s</div></div>',
                   unsafe_allow_html=True)
        st.markdown(f'<div class="card" style="margin-top:1rem"><div class="lbl">Class probabilities</div>{bars(probs)}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("#### ⚖ Model Comparison — DenseNet121 vs ResNet152")
        cols = st.columns(2)
        preds = []
        for col, (name, cfg) in zip(cols, MODELS.items()):
            probs, secs = predict(name, img)
            html, c = result_card(probs, secs)
            preds.append(c)
            col.markdown(f'<div class="model-h">{dot(cfg["color"])} {name}</div>{html}', unsafe_allow_html=True)

        if preds[0] == preds[1]:
            st.markdown(f'<div class="banner ok">✅ Both models agree — {dot(CLASS_COLORS[preds[0]])} {preds[0]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="banner no">⚠️ Models disagree — DenseNet121: {preds[0]} | ResNet152: {preds[1]}</div>',
                        unsafe_allow_html=True)
except (OSError, ValueError) as e:
    st.error(f"Could not load/run model: {e}\n\nCheck the file paths in MODELS at the top of app.py.")
