"""Run a detector (CFAR baseline or fine-tuned YOLO) over a whole scene with core+halo windows.

Scene inputs:
  * xView3 scene folder: VV_dB.tif, VH_dB.tif (+ owiMask.tif used as land/ice mask)
  * Gulf GeoTIFF from src.gulf_fetch: band 1 = VV dB, band 2 = VH dB
Output: CSV with one row per detection: scene_id, row, col, score, cls, shore_km, [lat, lon].

Usage:
  python -m src.predict --scene data/raw/xview3/scenes/<id> --detector cfar --out results/preds/cfar
  python -m src.predict --scene data/raw/gulf/khalifa_port_2026-10-21.tif --detector yolo \
      --weights models/best.pt --land-mask data/interim/khalifa_land.tif --out results/preds/yolo
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage

from .cfar import CfarParams, cfar_detect
from .sar import PIXEL_SIZE_M, encode_3ch, valid_mask
from .tiling import iter_windows, pad_to

CORE, HALO, TILE = 512, 64, 640
CLASS_NAMES = {0: "vessel", 1: "fixed_structure"}
LAND_BUFFER_PX = 10  # 100 m: harbour walls / coastline edges are the commonest CFAR false alarm


class Scene:
    """Uniform access to VV/VH (dB) and a land mask for both data sources."""

    def __init__(self, path: Path, land_mask_path: Path | None = None, noise_floor_db=None):
        self.path = Path(path)
        # (VV, VH) lower clip in dB. Sentinel Hub subtracts thermal noise and clips at 0, so calm
        # sea becomes exact zeros; flooring at the training data's 1st percentile makes
        # "below the noise level" look the same as in training (DECISIONS.md D25).
        self.noise_floor_db = noise_floor_db
        if self.path.is_dir():
            self.id = self.path.name
            self._vv = rasterio.open(self.path / "VV_dB.tif")
            self._vh = rasterio.open(self.path / "VH_dB.tif")
            self._vv_band = self._vh_band = 1
            mask_src = self.path / "owiMask.tif" if land_mask_path is None else land_mask_path
        else:
            self.id = self.path.stem
            self._vv = self._vh = rasterio.open(self.path)
            self._vv_band, self._vh_band = 1, 2
            mask_src = land_mask_path
        self.height, self.width = self._vv.height, self._vv.width
        self.transform, self.crs = self._vv.transform, self._vv.crs
        self.land = self._load_land(mask_src)

    def _load_land(self, path) -> np.ndarray | None:
        """Land/ice mask resampled to the SAR grid (nearest), buffered by LAND_BUFFER_PX.

        Assumption to verify on the first real xView3 scene: owiMask == 0 is open water and any
        non-zero code is land/ice.
        """
        if path is None or not Path(path).exists():
            return None
        if str(path).endswith(".geojson"):
            land = self._rasterize_geojson(path)
        else:
            with rasterio.open(path) as src:
                m = src.read(1, out_shape=(self.height, self.width), resampling=Resampling.nearest)
            land = m != 0
        return ndimage.binary_dilation(land, iterations=LAND_BUFFER_PX) if land.any() else land

    def _rasterize_geojson(self, path) -> np.ndarray:
        """Burn OSM land polygons (EPSG:4326, from src.land_mask) onto this scene's pixel grid."""
        import json

        from rasterio.features import rasterize
        from rasterio.warp import transform_geom

        fc = json.loads(Path(path).read_text())
        geoms = [transform_geom("EPSG:4326", self.crs, f["geometry"]) for f in fc["features"]]
        if not geoms:
            return np.zeros((self.height, self.width), bool)
        return rasterize(geoms, out_shape=(self.height, self.width), transform=self.transform, fill=0, default_value=1).astype(bool)

    def read(self, w) -> tuple[np.ndarray, np.ndarray]:
        win = Window(w.col0, w.row0, w.col1 - w.col0, w.row1 - w.row0)
        vv = self._vv.read(self._vv_band, window=win).astype(np.float32)
        vh = self._vh.read(self._vh_band, window=win).astype(np.float32)
        if self.noise_floor_db is not None:
            ok = valid_mask(vv, vh)
            vv = np.where(ok, np.maximum(vv, self.noise_floor_db[0]), vv)
            vh = np.where(ok, np.maximum(vh, self.noise_floor_db[1]), vh)
        return vv, vh

    def shore_km(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        if self.land is None or not self.land.any():
            return np.full(len(rows), np.nan)
        if not hasattr(self, "_dist"):
            # Coarse is fine for a 2 km near-shore cut-off; computed on a 10x-downsampled grid.
            small = self.land[::10, ::10]
            self._dist = ndimage.distance_transform_edt(~small) * PIXEL_SIZE_M * 10 / 1000.0
        r = np.clip((rows // 10).astype(int), 0, self._dist.shape[0] - 1)
        c = np.clip((cols // 10).astype(int), 0, self._dist.shape[1] - 1)
        return self._dist[r, c]

    def lonlat(self, rows, cols):
        from rasterio.warp import transform

        xs, ys = rasterio.transform.xy(self.transform, rows, cols)
        lon, lat = transform(self.crs, "EPSG:4326", np.atleast_1d(xs), np.atleast_1d(ys))
        return np.array(lon), np.array(lat)


def _keep_core(df: pd.DataFrame, w) -> pd.DataFrame:
    df = df.copy()
    df["row"] += w.row0
    df["col"] += w.col0
    return df[w.in_core(df.row.to_numpy(), df.col.to_numpy())]


def run_cfar(scene: Scene, params: CfarParams) -> pd.DataFrame:
    out = []
    for w in iter_windows(scene.height, scene.width, CORE, HALO):
        vv, vh = scene.read(w)
        ok = valid_mask(vv, vh)
        if scene.land is not None:
            ok &= ~scene.land[w.row0 : w.row1, w.col0 : w.col1]
        if ok.mean() < 0.05:
            continue
        det = cfar_detect(vh, ok, params)
        if len(det):
            out.append(_keep_core(det, w))
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["row", "col", "score"])
    df["cls"] = "vessel"  # CFAR can't tell ships from platforms
    return df


def merge_close(df: pd.DataFrame, radius_px: float = 5.0) -> pd.DataFrame:
    """Merge detections closer than radius_px (50 m): NMS on 13 px boxes leaves near-duplicates a few
    pixels apart. Greedy, highest score first (DECISIONS.md D26)."""
    if len(df) < 2:
        return df
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    rc = df[["row", "col"]].to_numpy(float)
    keep = np.ones(len(df), bool)
    for i in range(len(df)):
        if keep[i]:
            d = np.hypot(*(rc[i + 1 :] - rc[i]).T)
            keep[i + 1 :] &= d > radius_px
    return df[keep]


def run_yolo(scene: Scene, weights: str, conf: float = 0.05, batch: int = 16, device=None) -> pd.DataFrame:
    from ultralytics import YOLO

    model = YOLO(weights)
    windows = [w for w in iter_windows(scene.height, scene.width, CORE, HALO)]
    out = []
    for i in range(0, len(windows), batch):
        chunk, imgs = [], []
        for w in windows[i : i + batch]:
            vv, vh = scene.read(w)
            if valid_mask(vv, vh).mean() < 0.05:
                continue
            # Training PNGs are loaded by OpenCV as BGR and Ultralytics treats numpy input as BGR
            # too, so reversing our RGB-ordered encoding reproduces exactly what training saw.
            imgs.append(np.ascontiguousarray(pad_to(encode_3ch(vv, vh), TILE)[..., ::-1]))
            chunk.append(w)
        if not imgs:
            continue
        results = model.predict(imgs, imgsz=TILE, conf=conf, verbose=False, device=device)
        for w, r in zip(chunk, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xywh = r.boxes.xywh.cpu().numpy()
            df = pd.DataFrame(
                {
                    "row": xywh[:, 1],
                    "col": xywh[:, 0],
                    "score": r.boxes.conf.cpu().numpy(),
                    "cls": [CLASS_NAMES[int(c)] for c in r.boxes.cls.cpu().numpy()],
                }
            )
            out.append(_keep_core(df, w))
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["row", "col", "score", "cls"])
    if scene.land is not None and len(df):
        on_land = scene.land[df.row.astype(int).clip(0, scene.height - 1), df.col.astype(int).clip(0, scene.width - 1)]
        df = df[~on_land]
    return merge_close(df)


def predict_scene(path, detector, weights=None, land_mask=None, conf=0.05, cfar_params=None, noise_floor_db=None) -> pd.DataFrame:
    scene = Scene(Path(path), Path(land_mask) if land_mask else None, noise_floor_db)
    df = run_cfar(scene, cfar_params or CfarParams()) if detector == "cfar" else run_yolo(scene, weights, conf)
    df = df.reset_index(drop=True)
    df.insert(0, "scene_id", scene.id)
    df["shore_km"] = scene.shore_km(df.row.to_numpy(), df.col.to_numpy())
    if scene.crs is not None and len(df):
        df["lon"], df["lat"] = scene.lonlat(df.row.to_numpy(), df.col.to_numpy())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--detector", choices=["cfar", "yolo"], required=True)
    ap.add_argument("--weights")
    ap.add_argument("--land-mask")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--pfa", type=float, default=1e-6)
    ap.add_argument("--noise-floor", type=float, nargs=2, metavar=("VV_DB", "VH_DB"),
                    help="clip sub-noise sea, e.g. -29.2 -31.9 (OSSDD 1st percentile) for Sentinel Hub scenes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    df = predict_scene(a.scene, a.detector, a.weights, a.land_mask, a.conf, CfarParams(pfa=a.pfa), a.noise_floor)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(a.scene).stem}{a.suffix}.csv"
    df.to_csv(path, index=False)
    print(f"{len(df)} detections -> {path}")


if __name__ == "__main__":
    main()
