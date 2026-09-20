"""Walls of a levelled photo cloud as top-down line segments, from RANSAC vertical planes.

The 5 cm grid layout of the LiDAR tier needs walls that are thin and dense; monocular depth gives
walls 5-10 cm thick that are sparse where nothing was seen. Here walls are fitted as planes instead:

  1. RANSAC finds vertical planes (normal within 8 degrees of horizontal) in a slab of points at eye level,
     where the floor, the ceiling and low furniture do not reach,
  2. a plane also catches stray points that happen to lie near it, so its inliers are cut along the line into
     dense runs (5 cm bins, gaps up to 25 cm closed) and only runs of at least 0.6 m are kept,
  3. the line is refitted to the points of the runs, and parallel planes on the same wall are merged.

Everything is in the cloud's own metres; scale is not touched here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .layout import fit_line
from .photo_scene import EYE_SLAB
from .ransac import fit_planes

MAX_PLANES = 12
PLANE_THRESH, PLANE_RANGE_SLOPE = 0.05, 0.02  # inlier distance (m) and its growth per metre of range
PLANE_TILT_DEG = 8.0  # a wall plane's normal is within this of horizontal
MIN_PLANE_FRAC = 0.02
RUN_BIN = 0.05
RUN_GAP = 0.25  # a gap this short along a wall does not break a run
RUN_MIN_LEN = 0.6
RUN_MIN_DENSITY = 0.3  # a bin is occupied when it holds this fraction of the median occupied bin (and at least 3 points)
MIN_SUPPORT = 150  # points in the runs of a wall
MERGE_ANGLE_DEG, MERGE_OFFSET = 6.0, 0.12  # planes this parallel and this close (m) are one wall


@dataclass
class WallSegment:
    normal: np.ndarray  # (2,) unit, in x-z, pointing away from the station; the line is normal @ p == offset
    offset: float
    t0: float  # extent along tangent = (-normal[1], normal[0]): first run start to last run end
    t1: float
    runs: list[tuple[float, float]]  # dense stretches along the tangent, in metres
    support: int  # points in the runs

    @property
    def tangent(self) -> np.ndarray:
        return np.array([-self.normal[1], self.normal[0]])

    @property
    def length(self) -> float:
        return self.t1 - self.t0

    @property
    def p0(self) -> np.ndarray:
        return self.offset * self.normal + self.t0 * self.tangent

    @property
    def p1(self) -> np.ndarray:
        return self.offset * self.normal + self.t1 * self.tangent


def _runs(a: np.ndarray) -> list[tuple[float, float]]:
    """Dense stretches of the 1-D positions ``a`` (metres): occupied 5 cm bins, gaps up to 25 cm closed, at least 0.6 m long."""
    lo = float(a.min())
    counts = np.bincount(np.floor((a - lo) / RUN_BIN).astype(int)).astype(float)
    occ = counts >= max(3.0, RUN_MIN_DENSITY * float(np.median(counts[counts > 0])))
    k = int(round(RUN_GAP / RUN_BIN)) + 1
    closed = ndimage.binary_closing(np.pad(occ, k), structure=np.ones(k, bool))[k:-k]
    labels, n = ndimage.label(closed)
    out = []
    for i in range(1, n + 1):
        bins = np.nonzero((labels == i) & occ)[0]
        if len(bins) and (bins[-1] + 1 - bins[0]) * RUN_BIN >= RUN_MIN_LEN:
            out.append((lo + bins[0] * RUN_BIN, lo + (bins[-1] + 1) * RUN_BIN))
    return out


def _same_wall(a: WallSegment, b: WallSegment) -> bool:
    if abs(float(a.normal @ b.normal)) < np.cos(np.radians(MERGE_ANGLE_DEG)):
        return False
    n = a.normal if a.normal @ b.normal > 0 else -a.normal
    if abs(a.offset - (n @ (b.offset * b.normal))) > MERGE_OFFSET:
        return False
    ta = (b.p0 - a.offset * a.normal) @ a.tangent, (b.p1 - a.offset * a.normal) @ a.tangent
    return min(max(ta), a.t1) - max(min(ta), a.t0) > 0  # they overlap along the wall


def wall_segments(points: np.ndarray, seed: int = 0) -> list[WallSegment]:
    """Vertical wall segments (largest support first) of a cloud with +y up and the camera at the origin."""
    slab = np.asarray(points, float)
    slab = slab[(slab[:, 1] > EYE_SLAB[0]) & (slab[:, 1] < EYE_SLAB[1])]
    if len(slab) < 500:
        return []
    planes = fit_planes(
        slab, max_planes=MAX_PLANES, thresh=PLANE_THRESH, range_slope=PLANE_RANGE_SLOPE, min_inlier_frac=MIN_PLANE_FRAC,
        normal_hint=(0.0, 1.0, 0.0), max_angle_deg=PLANE_TILT_DEG, hint_mode="perpendicular", seed=seed,
    )
    segs: list[WallSegment] = []
    for pl in planes:
        q = slab[pl.inlier_mask][:, [0, 2]]
        n = pl.normal[[0, 2]] / np.linalg.norm(pl.normal[[0, 2]])
        a = q @ np.array([-n[1], n[0]])
        runs = _runs(a)
        if not runs:
            continue
        inrun = np.zeros(len(q), bool)
        for r0, r1 in runs:
            inrun |= (a >= r0) & (a <= r1)
        line = fit_line(q[inrun])  # refit on the runs only; the RANSAC plane also holds stray points
        if line is None or inrun.sum() < MIN_SUPPORT:
            continue
        n, c = line
        if c < 0:
            n, c = -n, -c
        t = np.array([-n[1], n[0]])
        a = q[inrun] @ t
        seg_runs = [(r0, r1) for r0, r1 in _runs(a) if r1 - r0 >= RUN_MIN_LEN]
        if seg_runs:
            segs.append(WallSegment(n, float(c), seg_runs[0][0], seg_runs[-1][1], seg_runs, int(inrun.sum())))
    segs.sort(key=lambda s: -s.support)
    kept: list[WallSegment] = []
    for s in segs:  # largest first; a smaller parallel plane on the same wall is dropped
        if not any(_same_wall(k, s) for k in kept):
            kept.append(s)
    return kept
