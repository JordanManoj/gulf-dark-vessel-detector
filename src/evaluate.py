"""xView3-style point matching and metrics (DECISIONS.md D12).

Predictions and ground truth are points (scene row/col at 10 m). Per scene we build the pairwise
distance matrix, solve the optimal one-to-one assignment (Hungarian algorithm) and accept pairs
within `tol_m` metres. Same logic as the official xView3 `metric.py`, plus the breakdowns this
project needs (near-shore vs open sea, vessel size, dark-flag quality).

Ground-truth columns used: scene_id, row, col, confidence, is_vessel, source,
vessel_length_m, distance_from_shore_km.
Prediction columns used:   scene_id, row, col, [score], [is_dark].
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .sar import PIXEL_SIZE_M

SIZE_BINS = [(0, 40, "small (<40 m)"), (40, 120, "medium (40-120 m)"), (120, np.inf, "large (>120 m)")]
NEAR_SHORE_KM = 2.0


def match_points(pred_rc: np.ndarray, gt_rc: np.ndarray, tol_m: float = 200.0) -> list[tuple[int, int]]:
    """Optimal one-to-one matching; returns (pred_idx, gt_idx) pairs with distance <= tol_m."""
    if len(pred_rc) == 0 or len(gt_rc) == 0:
        return []
    d = np.linalg.norm(pred_rc[:, None, :] - gt_rc[None, :, :], axis=-1) * PIXEL_SIZE_M
    # Pairs beyond tolerance get a prohibitive cost so the solver never prefers them
    # over a valid pair ("costly_dist" in the official scorer).
    cost = np.where(d <= tol_m, d, 1e9)
    pi, gi = linear_sum_assignment(cost)
    return [(p, g) for p, g in zip(pi, gi) if d[p, g] <= tol_m]


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def size_bin(length_m: float) -> str:
    if length_m is None or not np.isfinite(length_m):
        return "unknown length"
    for lo, hi, name in SIZE_BINS:
        if lo <= length_m < hi:
            return name
    return "unknown length"


def match_table(pred: pd.DataFrame, gt: pd.DataFrame, tol_m: float = 200.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate every prediction and every GT object with its match.

    LOW-confidence GT objects take part in matching (so a detection on them isn't counted as a
    false positive) but are flagged `scored=False` and excluded from metrics, as in the official
    scorer with --drop_low_detect.
    """
    pred = pred.copy()
    gt = gt.copy()
    pred["gt_index"] = -1
    gt["pred_index"] = -1
    for scene_id, g in gt.groupby("scene_id"):
        p = pred[pred.scene_id == scene_id]
        pairs = match_points(p[["row", "col"]].to_numpy(float), g[["row", "col"]].to_numpy(float), tol_m)
        for pi, gi in pairs:
            pred.loc[p.index[pi], "gt_index"] = g.index[gi]
            gt.loc[g.index[gi], "pred_index"] = p.index[pi]
    gt["matched"] = gt.pred_index >= 0
    gt["scored"] = gt.confidence.str.upper().isin(["HIGH", "MEDIUM"])
    pred["matched"] = pred.gt_index >= 0
    low_idx = set(gt.index[~gt.scored])
    pred["scored"] = ~pred.gt_index.isin(low_idx)
    return pred, gt


def detection_metrics(pred_m: pd.DataFrame, gt_m: pd.DataFrame) -> dict:
    """Overall + breakdowns. Inputs are the outputs of match_table()."""
    gt_s = gt_m[gt_m.scored]
    pred_s = pred_m[pred_m.scored]
    out = {"overall": prf(int(gt_s.matched.sum()), int((~pred_s.matched).sum()), int((~gt_s.matched).sum()))}

    # Near-shore vs open sea. A matched prediction inherits its GT's shore distance; unmatched
    # predictions use their own `shore_km` column if the pipeline computed it.
    gt_near = gt_s.distance_from_shore_km <= NEAR_SHORE_KM
    if "shore_km" in pred_s:
        fp_near = (~pred_s.matched) & (pred_s.shore_km <= NEAR_SHORE_KM)
        fp_far = (~pred_s.matched) & (pred_s.shore_km > NEAR_SHORE_KM)
    else:
        fp_near = fp_far = pd.Series(False, index=pred_s.index)
    for name, gmask, fpmask in [("near_shore", gt_near, fp_near), ("open_sea", ~gt_near, fp_far)]:
        g = gt_s[gmask]
        out[name] = prf(int(g.matched.sum()), int(fpmask.sum()), int((~g.matched).sum()))
        if "shore_km" not in pred_s:
            out[name]["note"] = "precision not available (no shore distance for predictions)"

    # Size: recall only (unmatched predictions have no known length).
    vessels = gt_s[gt_s.is_vessel.fillna(True).astype(bool)]
    for name, g in vessels.groupby(vessels.vessel_length_m.map(size_bin)):
        out[f"recall_{name}"] = {"n": len(g), "recall": float(g.matched.mean())}
    return out


def dark_flag_metrics(pred_m: pd.DataFrame, gt_m: pd.DataFrame) -> dict:
    """Quality of the dark flag on matched vessel detections.

    Truth: GT vessel with source == 'manual' (no AIS correlation, DECISIONS.md D02).
    Prediction: `is_dark` column produced by src.dark.flag_dark().
    """
    m = pred_m[pred_m.matched & pred_m.scored].join(gt_m[["source", "is_vessel"]], on="gt_index")
    m = m[m.is_vessel.fillna(True).astype(bool)]
    truth = m.source.str.lower().eq("manual")
    flag = m.is_dark.astype(bool)
    res = prf(int((truth & flag).sum()), int((~truth & flag).sum()), int((truth & ~flag).sum()))
    res["n_matched_vessels"] = len(m)
    res["true_dark_share"] = float(truth.mean()) if len(m) else 0.0
    return res
