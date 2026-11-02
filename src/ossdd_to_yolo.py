"""Convert the Open SAR Ship Detection Dataset (OSSDD) WebDataset shards to our YOLO format
(DECISIONS.md D19, D20).

OSSDD chips are raw Sentinel-1 GRD amplitude DN (VV, VH), not calibrated. We convert them with
a single approximate calibration constant (sigma0 = DN^2 / A^2, A ~ 550) so they land on the same
dB scale as xView3 and the Gulf data, then apply the same fixed 3-channel encoding.

  train shards (700 px chips) -> centre-crop to 640 px (no resizing, D07)
  val / test shards (512 px)  -> zero-pad to 640 px
Val and test chips are also saved as .npz (VV/VH dB) plus a point ground-truth CSV, so the CFAR
baseline and YOLO are scored on exactly the same pixels with src.evaluate (thresholds tuned on
val, reported on test).

Usage:
  python -m src.ossdd_to_yolo --shards data/raw/ossdd/webdataset --out data/processed/ossdd
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

from .sar import encode_3ch
from .tiling import pad_to

A_SIGMA = 550.0      # typical Sentinel-1 IW GRDH sigmaNought LUT magnitude (approximation, D19)
CHIP = 640
MIN_KEEP = 0.5       # keep a cropped box only if >= 50 % of its area survives the crop


def dn_to_db(dn: np.ndarray) -> np.ndarray:
    dn = np.asarray(dn, np.float32)
    db = 20.0 * np.log10(np.maximum(dn, 1e-3)) - 20.0 * np.log10(A_SIGMA)
    db[dn <= 0] = np.nan  # GRD can contain non-positive values near strong reflectors / no-data
    return db


def read_tif(b: bytes) -> np.ndarray:
    with rasterio.open(io.BytesIO(b)) as s:
        return s.read(1)


def parse_aabb(text: str) -> np.ndarray:
    """OSSDD axis-aligned boxes: first line = count, then 'idx xmin ymin xmax ymax [cx cy]'."""
    lines = [l.replace("||", " ").split() for l in text.strip().splitlines()[1:] if l.strip()]
    return np.array([[float(v) for v in l[1:5]] for l in lines], np.float32).reshape(-1, 4)


def crop_boxes(boxes: np.ndarray, off: int, size: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    b = boxes - off
    clipped = np.clip(b, 0, size)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_c = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    return clipped[(area > 0) & (area_c >= MIN_KEEP * area)]


def iter_samples(tar_path: Path):
    with tarfile.open(tar_path) as t:
        groups: dict[str, dict] = {}
        for m in t.getmembers():
            key, ext = m.name.split(".", 1)
            groups.setdefault(key, {})[ext] = t.extractfile(m).read()
        for key, g in groups.items():
            if {"vv.tif", "vh.tif", "aabb.txt"} <= g.keys():
                yield key, g


def convert(shards: Path, out: Path):
    gt_rows, stats = {"val": [], "test": []}, {}
    chip_rows = {"val": [], "test": []}
    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        if split in gt_rows:
            (out / f"{split}_arrays").mkdir(parents=True, exist_ok=True)
        n_chips = n_ships = 0
        sea_vv, sea_vh = [], []
        for tar_path in sorted((shards / split).glob("*.tar")):
            for key, g in iter_samples(tar_path):
                vv, vh = dn_to_db(read_tif(g["vv.tif"])), dn_to_db(read_tif(g["vh.tif"]))
                boxes = parse_aabb(g["aabb.txt"].decode())
                h, w = vv.shape
                if h > CHIP or w > CHIP:
                    off = (h - CHIP) // 2
                    vv, vh = vv[off : off + CHIP, off : off + CHIP], vh[off : off + CHIP, off : off + CHIP]
                    boxes = crop_boxes(boxes, off, CHIP)
                img = pad_to(encode_3ch(vv, vh), CHIP)
                Image.fromarray(img).save(out / "images" / split / f"{key}.png")
                lines = [
                    f"0 {(x0 + x1) / 2 / CHIP:.6f} {(y0 + y1) / 2 / CHIP:.6f} {(x1 - x0) / CHIP:.6f} {(y1 - y0) / CHIP:.6f}"
                    for x0, y0, x1, y1 in boxes
                ]
                (out / "labels" / split / f"{key}.txt").write_text("\n".join(lines))
                if split in gt_rows:
                    meta = json.loads(g["meta.json"]) if "meta.json" in g else {}
                    chip_rows[split].append({"scene_id": key, "land_fraction": meta.get("land_fraction", np.nan)})
                    np.savez_compressed(out / f"{split}_arrays" / f"{key}.npz", vv=vv, vh=vh)
                    for x0, y0, x1, y1 in boxes:
                        gt_rows[split].append(
                            {"scene_id": key, "row": (y0 + y1) / 2, "col": (x0 + x1) / 2, "length_px": max(x1 - x0, y1 - y0)}
                        )
                n_chips += 1
                n_ships += len(lines)
                sea_vv.append(np.nanmedian(vv))
                sea_vh.append(np.nanmedian(vh))
        stats[split] = {
            "chips": n_chips,
            "ships": n_ships,
            "median_chip_vv_db": float(np.nanmedian(sea_vv)) if sea_vv else None,
            "median_chip_vh_db": float(np.nanmedian(sea_vh)) if sea_vh else None,
        }
        print(split, stats[split])
    for split, rows in chip_rows.items():
        pd.DataFrame(rows, columns=["scene_id", "land_fraction"]).to_csv(out / f"{split}_chips.csv", index=False)
    for split, rows in gt_rows.items():
        pd.DataFrame(rows, columns=["scene_id", "row", "col", "length_px"]).to_csv(out / f"{split}_gt.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: vessel\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/raw/ossdd/webdataset")
    ap.add_argument("--out", default="data/processed/ossdd")
    a = ap.parse_args()
    convert(Path(a.shards), Path(a.out))


if __name__ == "__main__":
    main()
