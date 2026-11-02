"""Real failure-case crops for the slide deck (slide 6), at the deck's picture aspect (2.75 x 2.36 in).

Colour code as everywhere: green = correct, orange = false alarm, cyan dashed = missed ship.
Usage: python -m src.slide_crops
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Circle, Rectangle

from .eval_chips import load_gt
from .evaluate import match_table

DATA = Path("data/processed/ossdd")
OUT = Path("presentation/figures/failures")
GREEN, ORANGE, CYAN = "#2ecc71", "#f39c12", "#00e5ff"
ASPECT = 2.75 / 2.36


def save(img, marks, r0, c0, h, out, cmap=None, vmin=None, vmax=None, box=16):
    w = int(round(h * ASPECT))
    crop = img[r0 : r0 + h, c0 : c0 + w]
    fig = plt.figure(figsize=(2.75 * 3, 2.36 * 3), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(crop, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for kind, r, c in marks:
        r, c = r - r0, c - c0
        if not (0 <= r < h and 0 <= c < w):
            continue
        if kind == "miss":
            ax.add_patch(Circle((c, r), box * 0.9, fill=False, ec=CYAN, lw=3, ls="--"))
        else:
            ax.add_patch(Rectangle((c - box / 2, r - box / 2), box, box, fill=False, ec=GREEN if kind == "tp" else ORANGE, lw=3))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_axis_off()
    fig.savefig(OUT / out)
    plt.close(fig)


def chip_marks(model, key):
    _, _, gt = load_gt(DATA, "test")
    pred = pd.read_csv(f"results/preds/{model}_test.csv")
    pm, gm = match_table(pred[pred.scene_id == key], gt[gt.scene_id == key])
    marks = [("tp" if p.matched else "fp", p.row, p.col) for _, p in pm.iterrows()]
    marks += [("miss", g.row, g.col) for _, g in gm[~gm.matched].iterrows()]
    return pm, gm, marks


def chip_img(key):
    return plt.imread(DATA / "images" / "test" / f"{key}.png")[:512, :512]


def around(r, c, h, size=512):
    w = int(round(h * ASPECT))
    return int(np.clip(r - h / 2, 0, size - h)), int(np.clip(c - w / 2, 0, size - w))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. CFAR fragmentation: a false alarm right next to a correct detection.
    pm, gm, marks = chip_marks("cfar", "1aa1_9")
    fp, tp = pm[~pm.matched], pm[pm.matched]
    d = np.hypot(fp.row.values[:, None] - tp.row.values[None], fp.col.values[:, None] - tp.col.values[None])
    # among false alarms within 2.5 px-boxes of a correct hit, take the one farthest from the chip edge
    close = np.where(d.min(1) < 25)[0]
    edge = np.minimum.reduce([fp.row.values, fp.col.values, 512 - fp.row.values, 512 - fp.col.values])
    i = int(close[np.argmax(edge[close])])
    save(chip_img("1aa1_9"), marks, *around(fp.row.iloc[i], fp.col.iloc[i], 90), 90, "1_cfar_fragmentation.png", box=14)

    # 2. CFAR coastline clutter on the real Gulf scene (no labels: every coastline hit is a false alarm by eye).
    vv = rasterio.open("data/raw/gulf/khalifa_port_2026-10-21.tif").read(1)
    c = pd.read_csv("results/gulf/khalifa_port_2026-10-21_cfar_floor.csv")
    cnt = [((np.hypot(c.row - r, c.col - cc) < 120).sum()) for r, cc in zip(c.row, c.col)]
    k = int(np.argmax(cnt))
    h = 380
    w = int(round(h * ASPECT))
    r0 = int(np.clip(c.row.iloc[k] - h / 2, 0, vv.shape[0] - h))
    c0 = int(np.clip(c.col.iloc[k] - w / 2, 0, vv.shape[1] - w))
    save(vv, [("fp", r, cc) for r, cc in zip(c.row, c.col)], r0, c0, h, "2_cfar_coastline.png", cmap="gray", vmin=-30, vmax=0, box=22)

    # 3. YOLO misses faint / smeared ships.
    pm, gm, marks = chip_marks("yolov8s", "215c_151")
    m = gm[~gm.matched].iloc[0]
    save(chip_img("215c_151"), marks, *around(m.row, m.col, 200), 200, "3_yolo_missed_faint.png", box=16)

    # 4. Label noise: YOLO "false alarms" that look like real ships.
    pm, gm, marks = chip_marks("yolov8s", "215c_132")
    f = pm[~pm.matched]
    save(chip_img("215c_132"), marks, *around(f.row.median(), f.col.median(), 260), 260, "4_label_noise.png", box=16)
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
