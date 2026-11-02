"""AIS context for Gulf scenes from the Global Fishing Watch 4Wings API (DECISIONS.md D13, D26).

GFW gives gridded data only (0.01° ≈ 1.1 km cells at HIGH resolution), not ship tracks, so the
dark-vessel check on Gulf scenes is cell-based: a detection is "AIS-matched" if any AIS-broadcasting
vessel was present in its cell, or a neighbouring cell, during the acquisition hour.

  public-global-presence:latest      AIS vessel presence (1 position / vessel / hour), ~4-day lag
  public-global-sar-presence:latest  GFW's own Sentinel-1 detections, ~5-day lag

Usage:
  python -m src.gfw_ais --aoi aoi/khalifa_port.geojson --date 2026-10-15 --hour 2 --out data/raw/gfw
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

REPORT_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
AIS = "public-global-presence:latest"
SAR = "public-global-sar-presence:latest"
CELL_DEG = 0.01  # HIGH spatial resolution


def _rows(obj) -> list[dict]:
    """GFW nests rows as entries -> [{dataset_name: [rows...]}]; flatten whatever shape comes back."""
    if isinstance(obj, list):
        out = []
        for x in obj:
            out += _rows(x)
        return out
    if isinstance(obj, dict):
        if "lat" in obj and "lon" in obj:
            return [obj]
        out = []
        for v in obj.values():
            if isinstance(v, (list, dict)):
                out += _rows(v)
        return out
    return []


def report(dataset: str, aoi_geojson: dict, start: str, end: str, temporal: str = "HOURLY", group_by: str = "VESSEL_ID", filters: str | None = None) -> pd.DataFrame:
    load_dotenv()
    token = os.getenv("GFW_API_TOKEN")
    if not token:
        raise SystemExit("Set GFW_API_TOKEN in .env (see data/README.md).")
    params = {
        "spatial-resolution": "HIGH",
        "temporal-resolution": temporal,
        "datasets[0]": dataset,
        "date-range": f"{start},{end}",
        "format": "JSON",
        "group-by": group_by,
    }
    if filters:
        params["filters[0]"] = filters
    r = requests.post(
        REPORT_URL,
        params=params,
        json={"geojson": aoi_geojson},
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
    )
    if r.status_code != 200:
        raise SystemExit(f"GFW {r.status_code}: {r.text[:400]}")
    return pd.DataFrame(_rows(r.json().get("entries", [])))


def ais_cells(ais: pd.DataFrame, date: str, hour: int, window_h: int = 1) -> set[tuple[int, int]]:
    """Grid cells (lat_idx, lon_idx) with AIS presence within ±window_h hours of acquisition."""
    if ais.empty:
        return set()
    t = pd.to_datetime(ais["date"])
    t0 = pd.Timestamp(f"{date} {hour:02d}:00")
    near = ais[(t >= t0 - pd.Timedelta(hours=window_h)) & (t <= t0 + pd.Timedelta(hours=window_h))]
    return {(int(round(la / CELL_DEG)), int(round(lo / CELL_DEG))) for la, lo in zip(near.lat, near.lon)}


def flag_dark_gridded(det: pd.DataFrame, cells: set, neighbour: int = 1) -> pd.DataFrame:
    """A vessel detection is AIS-matched if its cell or any cell within `neighbour` cells has AIS."""
    det = det.copy()
    li = np.round(det.lat / CELL_DEG).astype(int)
    lo = np.round(det.lon / CELL_DEG).astype(int)
    det["ais_matched"] = [
        any((a + da, b + db) in cells for da in range(-neighbour, neighbour + 1) for db in range(-neighbour, neighbour + 1))
        for a, b in zip(li, lo)
    ]
    det["is_dark"] = (det.get("cls", "vessel") == "vessel") & ~det.ais_matched
    return det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--hour", type=int, required=True, help="acquisition hour (UTC)")
    ap.add_argument("--out", default="data/raw/gfw")
    a = ap.parse_args()
    gj = json.loads(Path(a.aoi).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(a.aoi).stem}_{a.date}"
    nxt = (pd.Timestamp(a.date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    ais = report(AIS, gj, a.date, nxt, temporal="HOURLY", group_by="VESSEL_ID")
    ais.to_csv(out / f"{stem}_ais_presence.csv", index=False)
    print(f"AIS presence rows: {len(ais)}; cells near {a.hour:02d}:00 UTC: {len(ais_cells(ais, a.date, a.hour))}")

    for flt, name in [("matched='true'", "sar_matched"), ("matched='false'", "sar_unmatched")]:
        try:
            sar = report(SAR, gj, a.date, nxt, temporal="DAILY", group_by="VESSEL_ID", filters=flt)
            sar.to_csv(out / f"{stem}_{name}.csv", index=False)
            n = int(sar["detections"].sum()) if "detections" in sar else len(sar)
            print(f"GFW SAR {name}: {n} detections in {len(sar)} rows")
        except SystemExit as e:
            print(f"GFW SAR {name}: {e}")


if __name__ == "__main__":
    main()
