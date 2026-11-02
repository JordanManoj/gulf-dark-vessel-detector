"""SAR radiometry helpers: dB conversion, no-data handling and the 3-channel encoding fed to YOLO.

See DECISIONS.md D06 for why the scaling is fixed (global) rather than per-chip.
"""
import numpy as np

PIXEL_SIZE_M = 10.0  # xView3 VV/VH rasters and our Gulf rasters are both 10 m

# Fixed clip ranges. VV/VH match the xView3 4th-place solution; the ratio range covers
# calm sea (large VV-VH) down to ships/structures (small VV-VH).
DB_MIN, DB_MAX = -50.0, 20.0
RATIO_MIN, RATIO_MAX = -5.0, 25.0

# Anything below this is a fill value (xView3 uses large negative fills), not real backscatter.
NODATA_BELOW_DB = -100.0


def linear_to_db(x: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """σ⁰ linear power -> dB. The floor avoids log(0) on no-data pixels."""
    return 10.0 * np.log10(np.maximum(x, floor))


def valid_mask(*bands_db: np.ndarray) -> np.ndarray:
    """True where every band holds real backscatter (finite and above the fill threshold)."""
    mask = np.ones(bands_db[0].shape, dtype=bool)
    for b in bands_db:
        mask &= np.isfinite(b) & (b > NODATA_BELOW_DB)
    return mask


def _scale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip(x, lo, hi) - lo) / (hi - lo)


def encode_3ch(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """Stack (VV, VH, VV-VH) into an HxWx3 uint8 image with fixed scaling.

    No-data pixels become 0 in all channels (same as the xView3 1st-place solution), so the
    network sees a consistent "nothing here" value at swath edges.
    """
    vv = vv_db.astype(np.float32)
    vh = vh_db.astype(np.float32)
    ok = valid_mask(vv, vh)
    vv = np.where(ok, vv, DB_MIN)
    vh = np.where(ok, vh, DB_MIN)
    img = np.stack(
        [_scale(vv, DB_MIN, DB_MAX), _scale(vh, DB_MIN, DB_MAX), _scale(vv - vh, RATIO_MIN, RATIO_MAX)],
        axis=-1,
    )
    img[~ok] = 0.0
    return (img * 255.0 + 0.5).astype(np.uint8)
