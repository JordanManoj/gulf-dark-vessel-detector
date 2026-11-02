"""Dark Vessel Detector: interactive demo (Streamlit).

Pick a Sentinel-1 test chip and compare the CFAR baseline with the fine-tuned YOLOv8s against the
labelled ships. Both detectors run live, so the sliders show the precision/recall trade-off.

Run from the repo root:
  streamlit run "demo app/app.py"
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cfar import CfarParams, cfar_detect  # noqa: E402
from src.eval_chips import load_gt  # noqa: E402
from src.evaluate import match_table, prf  # noqa: E402

DATA = ROOT / "data" / "processed" / "ossdd"
WEIGHTS = ROOT / "runs" / "yolov8s" / "weights" / "best.pt"
GREEN, ORANGE, CYAN = "#2ecc71", "#f39c12", "#00e5ff"

st.set_page_config(page_title="Dark Vessel Detector", layout="wide")


@st.cache_resource
def load_model():
    from ultralytics import YOLO

    return YOLO(str(WEIGHTS))


@st.cache_data
def load_test():
    chips, land, gt = load_gt(DATA, "test")
    counts = gt.groupby("scene_id").size()
    chips["ships"] = chips.scene_id.map(counts).fillna(0).astype(int)
    chips["type"] = np.where(chips.scene_id.map(land), "land", "open water")
    return chips, gt


def run_cfar(key: str, pfa: float) -> pd.DataFrame:
    a = np.load(DATA / "test_arrays" / f"{key}.npz")
    valid = np.isfinite(a["vv"]) & np.isfinite(a["vh"])
    det = cfar_detect(np.nan_to_num(a["vh"], nan=-50.0), valid, CfarParams(pfa=pfa))
    det["scene_id"] = key
    return det


def run_yolo(key: str, conf: float) -> pd.DataFrame:
    r = load_model().predict(str(DATA / "images" / "test" / f"{key}.png"), imgsz=640, conf=conf, verbose=False, max_det=600)[0]
    xywh = r.boxes.xywh.cpu().numpy()
    return pd.DataFrame({"scene_id": key, "row": xywh[:, 1], "col": xywh[:, 0], "score": r.boxes.conf.cpu().numpy()})


def panel(key, det, gt, title):
    pm, gm = match_table(det, gt[gt.scene_id == key])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(plt.imread(DATA / "images" / "test" / f"{key}.png"), interpolation="nearest")
    for _, g in gm[~gm.matched].iterrows():
        ax.add_patch(Circle((g.col, g.row), 14, fill=False, ec=CYAN, lw=1.5, ls="--"))
    for _, p in pm.iterrows():
        ax.add_patch(Rectangle((p.col - 8, p.row - 8), 16, 16, fill=False, ec=GREEN if p.matched else ORANGE, lw=1.4))
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_axis_off()
    m = prf(int(pm.matched.sum()), int((~pm.matched).sum()), int((~gm.matched).sum()))
    return fig, m, title


st.title("Dark Vessel Detector: Sentinel-1 ship detection")
st.caption(
    "Test chips from the Open SAR Ship Detection Dataset (Sentinel-1 GRD, VV+VH, 10 m). "
    "Image colours: red = VV, green = VH, blue = VV−VH ratio. "
    "**green box** = correct detection · **orange box** = false alarm · **cyan dashed circle** = missed ship."
)

chips, gt = load_test()
with st.sidebar:
    st.header("Chip")
    kind = st.radio("Show", ["all", "land", "open water"], horizontal=True)
    pool = chips if kind == "all" else chips[chips.type == kind]
    pool = pool.sort_values("ships", ascending=False)
    key = st.selectbox("Test chip (most ships first)", pool.scene_id, format_func=lambda k: f"{k}  ·  {int(pool.set_index('scene_id').ships[k])} ships")
    st.header("Thresholds")
    st.caption("Defaults = values tuned on the validation set.")
    pfa_exp = st.slider("CFAR false-alarm rate: 10^x", -14, -3, -10)
    conf = st.slider("YOLO confidence", 0.05, 0.9, 0.15, 0.05)

left, right = st.columns(2)
for col, (det, title) in zip(
    [left, right],
    [(run_cfar(key, 10.0**pfa_exp), f"CFAR baseline (PFA 1e{pfa_exp})"), (run_yolo(key, conf), f"Fine-tuned YOLOv8s (conf ≥ {conf:.2f})")],
):
    fig, m, t = panel(key, det, gt, title)
    with col:
        st.subheader(t)
        st.pyplot(fig, clear_figure=True)
        a, b, c = st.columns(3)
        a.metric("Correct", m["tp"])
        b.metric("False alarms", m["fp"])
        c.metric("Missed", m["fn"])
        st.caption(f"precision {m['precision']:.2f} · recall {m['recall']:.2f} · F1 {m['f1']:.2f}")

st.divider()
st.subheader("Whole test set (200 chips, 1,171 ships)")
st.image(str(ROOT / "results" / "figures" / "metrics_comparison.png"))
