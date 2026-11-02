"""Two-parameter CFAR ship detector: the no-ML baseline (DECISIONS.md D09).

For every pixel x (VH, dB):   detect if  x > mu_b + k * sigma_b
where mu_b, sigma_b are the mean / std of the *background ring* (background window minus the
guard window) and k is set from the probability of false alarm assuming Gaussian clutter in dB.

Box sums are computed with uniform filters (integral-image equivalent), so the cost per pixel is
constant regardless of window size. Land / no-data pixels are excluded from the statistics by
weighting with a validity mask.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import norm


@dataclass
class CfarParams:
    guard: int = 41        # px, ~ largest ship (≈400 m at 10 m spacing)
    background: int = 81   # px, ~800 m: SNAP's default background window
    pfa: float = 1e-6      # probability of false alarm per pixel
    min_pixels: int = 2    # connected components smaller than this are speckle
    max_pixels: int = 2500 # bigger blobs are land/clutter leaks, not ships
    # Speckle in multi-look GRD (ENL ~4.4) has a std of ~2.2 dB in log space
    # (4.34 * sqrt(trigamma(4.4))). A background that looks smoother than that is an artefact
    # (e.g. noise-subtracted / floored calm sea), so sigma_b is never allowed below it (D25).
    min_sigma_db: float = 2.2


def _box_sum(x: np.ndarray, size: int) -> np.ndarray:
    # uniform_filter returns the mean over size x size; multiply back to a sum.
    return ndimage.uniform_filter(x, size=size, mode="constant", cval=0.0) * float(size * size)


def cfar_statistic(vh_db: np.ndarray, valid: np.ndarray, p: CfarParams) -> np.ndarray:
    """Return z = (x - mu_b) / sigma_b per pixel (NaN where undefined)."""
    x = np.where(valid, vh_db, 0.0).astype(np.float64)
    w = valid.astype(np.float64)

    n = _box_sum(w, p.background) - _box_sum(w, p.guard)
    s1 = _box_sum(x, p.background) - _box_sum(x, p.guard)
    s2 = _box_sum(x * x, p.background) - _box_sum(x * x, p.guard)

    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        var = np.maximum(s2 / n - mu * mu, p.min_sigma_db**2)
        z = (x - mu) / np.sqrt(var)
    # Need enough background samples for stable statistics (e.g. right next to land).
    enough = n >= 0.25 * (p.background**2 - p.guard**2)
    z[~(valid & enough)] = np.nan
    return z.astype(np.float32)


def cfar_detect(vh_db: np.ndarray, valid: np.ndarray, p: CfarParams | None = None) -> pd.DataFrame:
    """Detect bright targets. Returns one row per object: row, col (intensity-weighted centroid),
    score (peak z), n_pixels. Coordinates are in the input array's pixel frame."""
    p = p or CfarParams()
    k = norm.isf(p.pfa)
    z = cfar_statistic(vh_db, valid, p)
    hits = np.nan_to_num(z, nan=-np.inf) > k

    labels, n = ndimage.label(hits, structure=np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame(columns=["row", "col", "score", "n_pixels"])
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum_labels(hits, labels, idx)
    weights = np.where(hits, z, 0.0)
    centroids = ndimage.center_of_mass(weights, labels, idx)
    peaks = ndimage.maximum(z, labels, idx)

    df = pd.DataFrame(
        {
            "row": [c[0] for c in centroids],
            "col": [c[1] for c in centroids],
            "score": peaks,
            "n_pixels": sizes.astype(int),
        }
    )
    keep = (df.n_pixels >= p.min_pixels) & (df.n_pixels <= p.max_pixels)
    return df[keep].reset_index(drop=True)
