"""Cut xView3 scenes into YOLO training chips (DECISIONS.md D01, D05-D08, D14).

For each scene folder present on disk:
  * object-centred chips: walk the labelled objects in random order; for each object not yet
    inside an existing chip, cut a 640 px chip with the object at a random offset (so objects
    aren't always in the centre). Every label inside the chip is written, not just the seed.
  * random chips (~10 % extra, as in the xView3 1st-place solution) so the model also sees
    empty sea, swath edges and coastline.
  * scenes are split train/val **by scene**, never by chip, so neighbouring chips of the same
    scene can't leak between the two.

Usage:
  python -m src.make_chips --labels data/raw/xview3/labels/validation.csv \
      --scenes data/raw/xview3/scenes --out data/processed/chips --val-frac 0.2 [--delete-raw]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image

from .sar import encode_3ch, valid_mask

CHIP = 640
BOX_PX = 10           # fixed 10x10 px box per point label (D08)
CLASSES = {"vessel": 0, "fixed_structure": 1}
RANDOM_FRAC = 0.10
MIN_VALID = 0.5       # skip chips that are mostly no-data


def scene_split(scene_ids, val_frac: float, seed: int = 0) -> dict:
    ids = sorted(scene_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_frac))
    return {s: ("val" if i < n_val else "train") for i, s in enumerate(ids)}


def class_of(is_vessel) -> int:
    # Unknown is_vessel -> vessel (D14): most unlabelled bright objects at sea are vessels.
    if pd.isna(is_vessel):
        return CLASSES["vessel"]
    return CLASSES["vessel"] if bool(is_vessel) else CLASSES["fixed_structure"]


def read_chip(vv_src, vh_src, r0: int, c0: int):
    win = Window(c0, r0, CHIP, CHIP)
    vv = vv_src.read(1, window=win, boundless=True, fill_value=-32768).astype(np.float32)
    vh = vh_src.read(1, window=win, boundless=True, fill_value=-32768).astype(np.float32)
    return vv, vh


def chip_origins(objs: pd.DataFrame, h: int, w: int, rng) -> list[tuple[int, int]]:
    origins = []
    covered = np.zeros(len(objs), bool)
    rows, cols = objs.row.to_numpy(), objs.col.to_numpy()
    for i in rng.permutation(len(objs)):
        if covered[i]:
            continue
        # Object lands anywhere in the inner 80 % of the chip.
        r0 = int(rows[i] - rng.integers(int(0.1 * CHIP), int(0.9 * CHIP)))
        c0 = int(cols[i] - rng.integers(int(0.1 * CHIP), int(0.9 * CHIP)))
        r0, c0 = int(np.clip(r0, 0, max(h - CHIP, 0))), int(np.clip(c0, 0, max(w - CHIP, 0)))
        origins.append((r0, c0))
        covered |= (rows >= r0 + BOX_PX) & (rows < r0 + CHIP - BOX_PX) & (cols >= c0 + BOX_PX) & (cols < c0 + CHIP - BOX_PX)
    n_rand = max(1, int(RANDOM_FRAC * len(origins)))
    for _ in range(n_rand):
        origins.append((int(rng.integers(0, max(h - CHIP, 1))), int(rng.integers(0, max(w - CHIP, 1)))))
    return origins


def yolo_lines(objs: pd.DataFrame, r0: int, c0: int) -> list[str]:
    inside = objs[(objs.row >= r0) & (objs.row < r0 + CHIP) & (objs.col >= c0) & (objs.col < c0 + CHIP)]
    lines = []
    for _, o in inside.iterrows():
        cx, cy = o.col - c0, o.row - r0
        # Clip the box to the chip, then convert to normalised centre/size.
        x0, x1 = max(cx - BOX_PX / 2, 0), min(cx + BOX_PX / 2, CHIP)
        y0, y1 = max(cy - BOX_PX / 2, 0), min(cy + BOX_PX / 2, CHIP)
        lines.append(
            f"{class_of(o.is_vessel)} {(x0 + x1) / 2 / CHIP:.6f} {(y0 + y1) / 2 / CHIP:.6f} {(x1 - x0) / CHIP:.6f} {(y1 - y0) / CHIP:.6f}"
        )
    return lines


def process_scene(scene_dir: Path, objs: pd.DataFrame, split: str, out: Path, rng) -> dict:
    stats = {"chips": 0, "skipped_nodata": 0, "objects": 0, "ratio_samples": []}
    with rasterio.open(scene_dir / "VV_dB.tif") as vv_src, rasterio.open(scene_dir / "VH_dB.tif") as vh_src:
        h, w = vv_src.height, vv_src.width
        for k, (r0, c0) in enumerate(chip_origins(objs, h, w, rng)):
            vv, vh = read_chip(vv_src, vh_src, r0, c0)
            ok = valid_mask(vv, vh)
            if ok.mean() < MIN_VALID:
                stats["skipped_nodata"] += 1
                continue
            name = f"{scene_dir.name}_{r0}_{c0}"
            Image.fromarray(encode_3ch(vv, vh)).save(out / "images" / split / f"{name}.png")
            lines = yolo_lines(objs, r0, c0)
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            stats["chips"] += 1
            stats["objects"] += len(lines)
            if k % 5 == 0:  # sample VV-VH to check the fixed ratio range (D06) on real data
                stats["ratio_samples"].append(rng.choice((vv - vh)[ok], 2000))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", default="data/processed/chips")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delete-raw", action="store_true", help="delete each scene folder after chipping (disk, D05)")
    a = ap.parse_args()

    labels = pd.read_csv(a.labels).rename(columns={"detect_scene_row": "row", "detect_scene_column": "col"})
    out = Path(a.out)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # The split is fixed from *all* scene ids in the label file, so it doesn't change as scenes
    # are downloaded and processed one by one.
    split_of = scene_split(labels.scene_id.unique(), a.val_frac, a.seed)
    (out / "split.json").write_text(json.dumps(split_of, indent=1))
    rng = np.random.default_rng(a.seed)

    log_path = out / "chip_log.csv"
    done = set(pd.read_csv(log_path).scene_id) if log_path.exists() else set()
    ratio_all = []
    for scene_dir in sorted(p for p in Path(a.scenes).iterdir() if p.is_dir()):
        sid = scene_dir.name
        if sid in done or sid not in split_of:
            continue
        st = process_scene(scene_dir, labels[labels.scene_id == sid], split_of[sid], out, rng)
        ratio_all += st.pop("ratio_samples")
        row = pd.DataFrame([{"scene_id": sid, "split": split_of[sid], **st}])
        row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        print(sid, split_of[sid], st)
        if a.delete_raw:
            shutil.rmtree(scene_dir)

    if ratio_all:
        r = np.concatenate(ratio_all)
        pct = {str(q): float(np.percentile(r, q)) for q in (0.5, 5, 50, 95, 99.5)}
        (out / "ratio_percentiles.json").write_text(json.dumps(pct, indent=1))
        print("VV-VH percentiles (dB):", pct)


if __name__ == "__main__":
    main()
