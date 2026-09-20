"""Floor and ceiling heights from a height histogram of the point cloud.

The world is gravity aligned, so a floor or ceiling shows up as a thin spike in the
histogram of point heights. Bins are 1 cm and the spike is refined by the median of
the points around it, because the ceiling gate (1.5 cm) is finer than a bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN = 0.01


@dataclass
class HeightPlane:
    y: float  # world height of the plane, metres
    spread: float  # robust std of the points that support it, metres
    support: int  # number of points in the spike
    sharpness: float  # fraction of all points that lie in the spike


def _spike(y: np.ndarray, lo: float, hi: float, half_width: float = 0.03) -> HeightPlane | None:
    sel = y[(y >= lo) & (y <= hi)]
    if len(sel) < 50:
        return None
    edges = np.arange(sel.min(), sel.max() + BIN, BIN)
    if len(edges) < 3:
        return None
    hist, edges = np.histogram(sel, bins=edges)
    # the voxel size is 2 cm, so counts alternate between 1 cm bins; smooth over 3 bins
    k = int(round(2 * half_width / BIN)) + 1
    smooth = np.convolve(hist, np.ones(k), mode="same")
    j = int(smooth.argmax())
    centre = 0.5 * (edges[j] + edges[j + 1])
    near = sel[np.abs(sel - centre) <= half_width]
    y0 = float(np.median(near))
    near = sel[np.abs(sel - y0) <= half_width]
    spread = 1.4826 * float(np.median(np.abs(near - np.median(near)))) if len(near) else 0.0
    return HeightPlane(y0, spread, len(near), len(near) / len(y))


def find_floor(y: np.ndarray) -> HeightPlane | None:
    """Floor: the strongest spike among the lowest points.

    Stray depth pixels can sit far below the floor, so the search window starts at the
    0.5th percentile, not at the minimum.
    """
    lo = float(np.percentile(y, 0.5))
    return _spike(y, lo, lo + 1.2)


def find_ceiling(
    y: np.ndarray,
    floor_y: float,
    footprint_ok: "callable | None" = None,
) -> HeightPlane | None:
    """Ceiling: the highest strong spike between 1.9 m and 5 m above the floor.

    Furniture tops also make spikes, so a candidate must be *wide*: ``footprint_ok`` is
    a function taking a candidate height and returning True when the horizontal
    footprint of the points at that height is large enough to be a ceiling.
    """
    h = y - floor_y
    band = (h >= 1.9) & (h <= 5.0)
    if band.sum() < 200:
        return None
    hb = h[band]
    hist, edges = np.histogram(hb, bins=np.arange(1.9, 5.0 + BIN, BIN))
    smooth = np.convolve(hist, np.ones(3), mode="same")
    # candidate peaks: local maxima above a fraction of the strongest
    thresh = max(0.25 * smooth.max(), 100)
    cands = [
        j
        for j in range(1, len(smooth) - 1)
        if smooth[j] >= thresh and smooth[j] >= smooth[j - 1] and smooth[j] > smooth[j + 1]
    ]
    for j in sorted(cands, reverse=True):  # highest first
        centre = floor_y + 0.5 * (edges[j] + edges[j + 1])
        plane = _spike(y, centre - 0.05, centre + 0.05)
        if plane is None:
            continue
        if footprint_ok is None or footprint_ok(plane.y):
            return plane
    return None
