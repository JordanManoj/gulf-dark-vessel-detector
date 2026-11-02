"""Dark-vessel flag: a vessel detection with no AIS position within `tol_m` is "dark" (DECISIONS.md D03).

This is deliberately *not* learned from pixels: "dark" is a property of the AIS record, not of
how the ship looks. Fixed structures are never flagged (they don't carry AIS, D14).
"""
import pandas as pd

from .evaluate import match_points


def flag_dark(detections: pd.DataFrame, ais: pd.DataFrame, tol_m: float = 200.0) -> pd.DataFrame:
    """detections: scene_id, row, col, cls ('vessel' | 'fixed_structure').
    ais: scene_id, row, col (AIS positions interpolated to image time, in scene pixels).
    Returns detections with `ais_matched` and `is_dark` columns."""
    det = detections.copy()
    det["ais_matched"] = False
    for scene_id, d in det.groupby("scene_id"):
        vessels = d[d.cls == "vessel"]
        a = ais[ais.scene_id == scene_id]
        pairs = match_points(vessels[["row", "col"]].to_numpy(float), a[["row", "col"]].to_numpy(float), tol_m)
        det.loc[vessels.index[[p for p, _ in pairs]], "ais_matched"] = True
    det["is_dark"] = (det.cls == "vessel") & ~det.ais_matched
    return det


def xview3_ais_points(gt: pd.DataFrame) -> pd.DataFrame:
    """In xView3, the AIS positions available at image time are the labels whose source includes AIS."""
    src = gt.source.str.lower()
    return gt[src.isin(["ais", "ais/manual"])][["scene_id", "row", "col"]]
