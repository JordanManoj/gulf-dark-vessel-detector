"""Baseline vs model on OSSDD chips with point matching (DECISIONS.md D12, D21).

1. CFAR: sweep PFA on **val**, pick the best F1, apply that single setting to **test**.
2. YOLO: predict once at a low confidence, sweep the confidence threshold on **val**, apply to **test**.
Near-shore proxy: OSSDD has no land mask, so chips with land_fraction > 0 are "near-shore chips"
and the rest are "open water". CFAR runs without a land mask here, so the open-water numbers are
the fair head-to-head; land chips show what happens without one.

Usage:
  python -m src.eval_chips --data data/processed/ossdd --weights runs/yolov8s/weights/best.pt --name yolov8s
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cfar import CfarParams, cfar_detect
from .evaluate import detection_metrics, match_table
from .sar import PIXEL_SIZE_M

PFAS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12]
CONFS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]


def load_gt(data: Path, split: str):
    chips = pd.read_csv(data / f"{split}_chips.csv")
    land = dict(zip(chips.scene_id, chips.land_fraction.fillna(0) > 0))
    gt = pd.read_csv(data / f"{split}_gt.csv")
    gt["confidence"] = "HIGH"
    gt["is_vessel"] = True
    gt["source"] = "unknown"
    gt["vessel_length_m"] = gt.length_px * PIXEL_SIZE_M
    # Proxy shore distance: 0 km for chips containing land, far otherwise (see module docstring).
    gt["distance_from_shore_km"] = gt.scene_id.map(lambda k: 0.0 if land[k] else 99.0)
    return chips, land, gt


def score(pred: pd.DataFrame, gt: pd.DataFrame, land: dict) -> dict:
    pred = pred.copy()
    pred["shore_km"] = pred.scene_id.map(lambda k: 0.0 if land[k] else 99.0)
    pm, gm = match_table(pred, gt)
    return detection_metrics(pm, gm)


def cfar_predictions(data: Path, split: str, chips: pd.DataFrame, pfa: float) -> pd.DataFrame:
    out = []
    for key in chips.scene_id:
        a = np.load(data / f"{split}_arrays" / f"{key}.npz")
        valid = np.isfinite(a["vv"]) & np.isfinite(a["vh"])
        det = cfar_detect(np.nan_to_num(a["vh"], nan=-50.0), valid, CfarParams(pfa=pfa))
        det["scene_id"] = key
        out.append(det)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["scene_id", "row", "col", "score"])


def yolo_predictions(data: Path, split: str, chips: pd.DataFrame, weights: str, device=None) -> pd.DataFrame:
    from ultralytics import YOLO

    model = YOLO(weights)
    paths = [str(data / "images" / split / f"{k}.png") for k in chips.scene_id]
    rows = []
    for i in range(0, len(paths), 32):
        # PNG paths -> loaded exactly as in training (OpenCV), so no channel-order mismatch.
        for p, r in zip(paths[i : i + 32], model.predict(paths[i : i + 32], imgsz=640, conf=0.01, verbose=False, device=device, max_det=600)):
            xywh = r.boxes.xywh.cpu().numpy()
            for (x, y, _, _), c in zip(xywh, r.boxes.conf.cpu().numpy()):
                rows.append({"scene_id": Path(p).stem, "row": float(y), "col": float(x), "score": float(c)})
    return pd.DataFrame(rows, columns=["scene_id", "row", "col", "score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/ossdd")
    ap.add_argument("--weights", help="YOLO best.pt; omit to evaluate only the CFAR baseline")
    ap.add_argument("--name", default="cfar")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)

    val_chips, val_land, val_gt = load_gt(data, "val")
    test_chips, test_land, test_gt = load_gt(data, "test")
    sweep = []

    if a.weights is None:
        for pfa in PFAS:
            m = score(cfar_predictions(data, "val", val_chips, pfa), val_gt, val_land)["overall"]
            sweep.append({"pfa": pfa, **m})
            print("val pfa", pfa, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
        best = max(sweep, key=lambda r: r["f1"])
        pred_test = cfar_predictions(data, "test", test_chips, best["pfa"])
        chosen = {"pfa": best["pfa"]}
    else:
        pred_val = yolo_predictions(data, "val", val_chips, a.weights)
        for c in CONFS:
            m = score(pred_val[pred_val.score >= c], val_gt, val_land)["overall"]
            sweep.append({"conf": c, **m})
            print("val conf", c, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
        best = max(sweep, key=lambda r: r["f1"])
        pred_all = yolo_predictions(data, "test", test_chips, a.weights)
        pred_all.to_csv(out / "preds" / f"{a.name}_test_all.csv", index=False)
        pred_test = pred_all[pred_all.score >= best["conf"]]
        chosen = {"conf": best["conf"]}

    pred_test.to_csv(out / "preds" / f"{a.name}_test.csv", index=False)
    test_metrics = score(pred_test, test_gt, test_land)
    report = {"model": a.name, "chosen_on_val": chosen, "val_sweep": sweep, "test": test_metrics,
              "n_test_chips": len(test_chips), "n_test_ships": len(test_gt)}
    (out / f"metrics_{a.name}.json").write_text(json.dumps(report, indent=1, default=float))
    print("TEST", a.name, chosen, json.dumps(test_metrics["overall"], default=float))


if __name__ == "__main__":
    main()
