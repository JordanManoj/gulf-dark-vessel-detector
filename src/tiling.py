"""Core + halo tiling (DECISIONS.md D07, D07b).

The scene is cut into non-overlapping *cores*. Each core is read together with a *halo* of
context on every side. A detection is kept only if its centre lies in the core, so every point
in the scene belongs to exactly one window: no duplicates, no cross-tile merging.
"""
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Window:
    # Read region in scene pixels (clipped to the scene).
    row0: int
    col0: int
    row1: int
    col1: int
    # Core region in scene pixels.
    core_row0: int
    core_col0: int
    core_row1: int
    core_col1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.row1 - self.row0, self.col1 - self.col0

    def in_core(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Mask of scene-coordinate points whose centre lies inside this window's core."""
        return (
            (rows >= self.core_row0) & (rows < self.core_row1) & (cols >= self.core_col0) & (cols < self.core_col1)
        )


def iter_windows(height: int, width: int, core: int = 512, halo: int = 64) -> Iterator[Window]:
    for r in range(0, height, core):
        for c in range(0, width, core):
            cr1, cc1 = min(r + core, height), min(c + core, width)
            yield Window(
                row0=max(r - halo, 0),
                col0=max(c - halo, 0),
                row1=min(cr1 + halo, height),
                col1=min(cc1 + halo, width),
                core_row0=r,
                core_col0=c,
                core_row1=cr1,
                core_col1=cc1,
            )


def pad_to(img: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad an HxW(xC) array at the bottom/right to size x size (edge windows are smaller).

    We pad ourselves instead of letting the detector letterbox-resize, because resizing would
    change the apparent size of few-pixel ships.
    """
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    pad = [(0, size - h), (0, size - w)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad)
