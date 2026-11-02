"""Plotting: detections on SAR imagery. Colour code used everywhere (README, slides, demo):
  green  = vessel matched to AIS      red    = dark vessel (no AIS)
  yellow = fixed structure            cyan   = ground truth (evaluation figures)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

COLOURS = {"ais": "#2ecc71", "dark": "#e74c3c", "fixed_structure": "#f1c40f", "gt": "#00e5ff"}


def sar_display(vh_db: np.ndarray, lo: float = -30.0, hi: float = 0.0) -> np.ndarray:
    """Greyscale VH for display only (the model uses src.sar.encode_3ch)."""
    return np.clip((np.nan_to_num(vh_db, nan=lo) - lo) / (hi - lo), 0, 1)


def _colour(d) -> str:
    if d.get("cls") == "fixed_structure":
        return COLOURS["fixed_structure"]
    return COLOURS["dark"] if d.get("is_dark", False) else COLOURS["ais"]


def plot_detections(
    image: np.ndarray,
    detections: pd.DataFrame | None = None,
    gt: pd.DataFrame | None = None,
    title: str = "",
    out: str | Path | None = None,
    box: int = 14,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None, interpolation="nearest")
    if gt is not None:
        for _, g in gt.iterrows():
            ax.add_patch(plt.Circle((g.col, g.row), box * 0.9, fill=False, ec=COLOURS["gt"], lw=1.0, ls="--"))
    if detections is not None:
        for _, d in detections.iterrows():
            ax.add_patch(
                Rectangle((d.col - box / 2, d.row - box / 2), box, box, fill=False, ec=_colour(d), lw=1.5)
            )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def plot_yolo_chip(png: Path, label_txt: Path, out: Path):
    """Overlay YOLO labels on a chip: used to check the chip/label conversion visually."""
    img = np.array(plt.imread(png))
    h, w = img.shape[:2]
    rows = []
    for line in Path(label_txt).read_text().splitlines():
        c, x, y, *_ = line.split()
        rows.append({"row": float(y) * h, "col": float(x) * w, "cls": "fixed_structure" if c == "1" else "vessel"})
    plot_detections(img, pd.DataFrame(rows), title=png.stem, out=out)
