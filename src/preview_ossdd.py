"""Save viewable PNG previews of OSSDD chips (VH in dB, ship boxes in green).

The .tar shards hold float32 GeoTIFFs of raw radar amplitude, which normal image viewers show as
black/white. This converts a few chips to dB, stretches them for display and draws the labels.

Usage:
  python -m src.preview_ossdd --tar data/raw/ossdd/webdataset/val/val-000000.tar --n 8
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .ossdd_to_yolo import dn_to_db, iter_samples, parse_aabb, read_tif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results/figures/ossdd_preview")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for key, g in iter_samples(Path(a.tar)):
        boxes = parse_aabb(g["aabb.txt"].decode())
        if len(boxes) == 0:  # show chips that actually contain ships
            continue
        vv, vh = dn_to_db(read_tif(g["vv.tif"])), dn_to_db(read_tif(g["vh.tif"]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, img, name in [(axes[0], vv, "VV"), (axes[1], vh, "VH")]:
            ax.imshow(img, cmap="gray", vmin=-30, vmax=0)
            for x0, y0, x1, y1 in boxes:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#2ecc71", lw=1.2))
            ax.set_title(f"{name} (dB)  {key}  ships: {len(boxes)}")
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out / f"{key}.png", dpi=110)
        plt.close(fig)
        saved += 1
        if saved >= a.n:
            break
    print(f"saved {saved} previews to {out}")


if __name__ == "__main__":
    main()
