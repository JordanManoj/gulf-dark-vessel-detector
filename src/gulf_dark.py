"""Dark-vessel flag on a Gulf scene + agreement with GFW's own SAR detections (DECISIONS.md D26).

Inputs: our detections (src.predict CSV with lat/lon), GFW AIS presence + GFW SAR detections
(src.gfw_ais CSVs). Outputs: flagged CSV, summary JSON, figure (teal = AIS-matched, red = dark).

Usage:
  python -m src.gulf_dark --scene data/raw/gulf/khalifa_port_2026-10-15.tif \
      --det results/gulf/khalifa_port_2026-10-15_yolov8s_floor.csv --gfw data/raw/gfw/khalifa_port_2026-10-15 --hour 2
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Rectangle

from .gfw_ais import CELL_DEG, ais_cells, flag_dark_gridded

TEAL, RED, GOLD = "#00b3b3", "#e74c3c", "#f1c40f"


def cells_of(df: pd.DataFrame) -> set:
    if df is None or df.empty:
        return set()
    return {(int(round(a / CELL_DEG)), int(round(b / CELL_DEG))) for a, b in zip(df.lat, df.lon)}


def near_any(det: pd.DataFrame, cells: set, k: int = 1) -> list[bool]:
    li = np.round(det.lat / CELL_DEG).astype(int)
    lo = np.round(det.lon / CELL_DEG).astype(int)
    return [any((a + i, b + j) in cells for i in range(-k, k + 1) for j in range(-k, k + 1)) for a, b in zip(li, lo)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--gfw", required=True, help="prefix of src.gfw_ais outputs")
    ap.add_argument("--hour", type=int, required=True)
    ap.add_argument("--out", default="results/gulf")
    a = ap.parse_args()

    date = Path(a.scene).stem.split("_")[-1]
    det = pd.read_csv(a.det)
    ais = pd.read_csv(f"{a.gfw}_ais_presence.csv")
    det = flag_dark_gridded(det, ais_cells(ais, date, a.hour))

    read = lambda n: pd.read_csv(f"{a.gfw}_{n}.csv") if Path(f"{a.gfw}_{n}.csv").stat().st_size > 2 else pd.DataFrame()
    gfw_m, gfw_u = read("sar_matched"), read("sar_unmatched")
    gfw_cells = cells_of(gfw_m) | cells_of(gfw_u)
    det["gfw_sar_nearby"] = near_any(det, gfw_cells)

    stem = Path(a.det).stem
    det.to_csv(Path(a.out) / f"{stem}_dark.csv", index=False)
    vessels = det[det.get("cls", "vessel") == "vessel"]
    summary = {
        "scene": Path(a.scene).name,
        "acquisition_hour_utc": a.hour,
        "detections": int(len(det)),
        "ais_matched": int(vessels.ais_matched.sum()),
        "dark_candidates": int(vessels.is_dark.sum()),
        "gfw_sar_detections": {"matched": int(gfw_m.detections.sum()) if len(gfw_m) else 0,
                               "unmatched": int(gfw_u.detections.sum()) if len(gfw_u) else 0},
        "our_detections_near_a_gfw_detection": int(det.gfw_sar_nearby.sum()),
        "method": "AIS presence (GFW, 0.01 deg cells, +-1 h, +-1 cell) - cell-level, not per-ship",
    }
    (Path(a.out) / f"{stem}_dark_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    img = rasterio.open(a.scene).read(1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray", vmin=-30, vmax=0)
    for _, d in det.iterrows():
        ax.add_patch(Rectangle((d.col - 16, d.row - 16), 32, 32, fill=False, ec=RED if d.is_dark else TEAL, lw=1.8))
    ax.set_title(f"{Path(a.scene).stem}: teal = AIS present nearby · red = dark candidate (no AIS)", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(Path("results/figures") / f"{stem}_dark.png", dpi=110)


if __name__ == "__main__":
    main()
