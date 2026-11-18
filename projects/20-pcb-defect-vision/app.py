"""PCB defect inspection demo: scan a board image and show where the model sees defects.

streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from board_scan import available_model, scan  # noqa: E402

ROOT = Path(__file__).resolve().parent
PCB = ROOT / "downloads" / "DeepPCB" / "PCBData"

st.set_page_config(page_title="PCB defect inspection", layout="wide")
st.title("PCB defect inspection: patch classifier scanned across a board")
st.caption("Trained on DeepPCB (MIT licence). Its defects are artificially added to real board images; "
           "see README for limits. Demo only, not a validated inspection system.")


@st.cache_resource
def model():
    return available_model()


name, m = model()
st.sidebar.write(f"Model: **{name}**")
threshold = st.sidebar.slider("Defect probability threshold", 0.05, 0.95, 0.5, 0.05)
upload = st.sidebar.file_uploader("Board image (greyscale, e.g. DeepPCB *_test.jpg)", type=["jpg", "png"])
examples = sorted(PCB.glob("group*/*/*_test.jpg"))[:200] if PCB.exists() else []
pick = st.sidebar.selectbox("...or a DeepPCB board", [str(p.relative_to(PCB)) for p in examples]) if examples else None

src = upload or (PCB / pick if pick else None)
if src is None:
    st.info("Upload a board image, or run src/prepare_patches.py to download DeepPCB examples.")
    st.stop()
img = np.array(Image.open(src).convert("L"))
res = scan(img, m, name)
hits = [w for w in res["windows"] if w["defect_prob"] >= threshold]

c1, c2 = st.columns(2)
c1.image(img, caption="board", use_container_width=True)
rgb = np.stack([img] * 3, -1).astype(np.float32)
rgb[..., 0] = np.clip(rgb[..., 0] + 255 * res["heatmap"], 0, 255)
c2.image(rgb.astype(np.uint8), caption="red = defect probability (mean over overlapping 64 px windows)",
         use_container_width=True)
st.metric("Windows over threshold", f"{len(hits)} / {len(res['windows'])}")
if hits:
    st.dataframe(sorted(hits, key=lambda w: -w["defect_prob"])[:30], use_container_width=True)
