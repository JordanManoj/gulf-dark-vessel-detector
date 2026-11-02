"""Result figures for README / slides, built only from saved predictions and metrics (no re-running).

  results/figures/metrics_comparison.png   precision / recall / F1, overall + land vs open water
  results/figures/cfar_vs_<model>_*.png     same chip, both detectors side by side
  results/failures/<model>_false_alarms_*.png, <model>_missed_*.png, <model>_success_*.png

Colour code: green = correct detection, orange = false alarm, cyan dashed circle = missed ship.

Usage:
  python -m src.make_figures --models cfar yolov8s [yolo26s]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

from .eval_chips import load_gt
from .evaluate import match_table

DATA = Path("data/processed/ossdd")
NAMES = {"cfar": "CFAR baseline", "yolov8s": "YOLOv8s", "yolo26s": "YOLO26s"}
GREEN, ORANGE, CYAN = "#2ecc71", "#f39c12", "#00e5ff"


def metrics_chart(models, out: Path):
    rows = []
    for m in models:
        t = json.loads(Path(f"results/metrics_{m}.json").read_text())["test"]
        for subset, label in [("overall", "All test chips"), ("near_shore", "Chips with land"), ("open_sea", "Open water")]:
            rows.append({"model": NAMES[m], "subset": label, **{k: t[subset][k] for k in ["precision", "recall", "f1"]}})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    colours = ["#8a96a3", "#1f6fb2", "#0b8a6f"][: len(models)]
    w = 0.8 / len(models)
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        subsets = df.subset.unique()
        x = np.arange(len(subsets))
        for i, (m, c) in enumerate(zip(df.model.unique(), colours)):
            vals = df[df.model == m].set_index("subset").loc[subsets, metric].to_numpy()
            bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w, label=m, color=c)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(x, subsets, fontsize=9)
        ax.set_title(metric.upper() if metric == "f1" else metric.capitalize(), fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Test set: 200 Sentinel-1 chips, 1,171 ships (OSSDD). Thresholds tuned on validation only.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    df.to_csv(out.with_suffix(".csv"), index=False)


def matched(model: str):
    chips, land, gt = load_gt(DATA, "test")
    pred = pd.read_csv(f"results/preds/{model}_test.csv")
    pm, gm = match_table(pred, gt)
    return chips, land, pm, gm


def draw(ax, key, pm, gm, title):
    img = plt.imread(DATA / "images" / "test" / f"{key}.png")
    ax.imshow(img, interpolation="nearest")
    for _, g in gm[(gm.scene_id == key) & ~gm.matched].iterrows():
        ax.add_patch(Circle((g.col, g.row), 14, fill=False, ec=CYAN, lw=1.4, ls="--"))
    for _, p in pm[pm.scene_id == key].iterrows():
        ax.add_patch(Rectangle((p.col - 8, p.row - 8), 16, 16, fill=False, ec=GREEN if p.matched else ORANGE, lw=1.3))
    tp = int((pm[pm.scene_id == key].matched).sum())
    fp = int((~pm[pm.scene_id == key].matched).sum())
    fn = int((gm[gm.scene_id == key].matched == False).sum())
    ax.set_title(f"{title}\ncorrect {tp} · false alarms {fp} · missed {fn}", fontsize=9)
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_axis_off()


def per_chip(pm, gm):
    s = pd.DataFrame({
        "tp": pm[pm.matched].groupby("scene_id").size(),
        "fp": pm[~pm.matched].groupby("scene_id").size(),
        "fn": gm[~gm.matched].groupby("scene_id").size(),
        "n_gt": gm.groupby("scene_id").size(),
    }).fillna(0).astype(int)
    return s


def gallery(model: str, base: str = "cfar"):
    chips, land, pm, gm = matched(model)
    _, _, bpm, bgm = matched(base)
    s = per_chip(pm, gm)
    out = Path("results/failures")
    out.mkdir(parents=True, exist_ok=True)
    picks = {
        "false_alarms": s.sort_values("fp", ascending=False).index[:3],
        "missed": s.sort_values("fn", ascending=False).index[:3],
        "success": s[(s.fp == 0) & (s.fn == 0)].sort_values("tp", ascending=False).index[:3],
    }
    for kind, keys in picks.items():
        for i, key in enumerate(keys):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5.3))
            draw(axes[0], key, bpm, bgm, f"{NAMES[base]}  ({'land' if land[key] else 'open water'})")
            draw(axes[1], key, pm, gm, NAMES[model])
            fig.text(0.5, 0.01, "green = correct · orange = false alarm · cyan dashed = missed ship", ha="center", fontsize=8.5)
            fig.tight_layout(rect=(0, 0.03, 1, 0.97))
            fig.savefig(out / f"{model}_{kind}_{i + 1}_{key}.png", dpi=140)
            plt.close(fig)
    print(model, {k: list(v) for k, v in picks.items()})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cfar", "yolov8s"])
    a = ap.parse_args()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    metrics_chart(a.models, Path("results/figures/metrics_comparison.png"))
    for m in a.models:
        if m != "cfar":
            gallery(m)


if __name__ == "__main__":
    main()
