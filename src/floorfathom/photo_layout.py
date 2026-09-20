"""Walls of a levelled photo cloud as top-down line segments, from RANSAC vertical planes.

The 5 cm grid layout of the LiDAR tier needs walls that are thin and dense; monocular depth gives
walls 5-10 cm thick that are sparse where nothing was seen. Here walls are fitted as planes instead:

  1. RANSAC finds vertical planes (normal within 8 degrees of horizontal) in a slab of points from just below
     to well above the camera, where the floor, the ceiling and most furniture do not reach,
  2. a plane also catches stray points that happen to lie near it, so its inliers are cut along the line into
     dense runs (5 cm bins, gaps up to 25 cm closed) and only runs of at least 0.6 m are kept,
  3. the line is refitted to the points of the runs, and parallel planes on the same wall are merged.

Everything is in the cloud's own metres; scale is not touched here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from . import layout as L
from .estimate import OpeningEst
from .layout import fit_line
from .photo_scene import _is_area
from .planes import HeightPlane
from .ransac import fit_planes

WALL_SLAB = (-0.2, 0.8)  # heights relative to the camera (m) whose points are fitted as walls: above most furniture,
# below the ceiling; provisional, compared on two real rooms only
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
RAY_DEG = 1.0  # the outline is the visible boundary from the station, sampled at this angular step
RAY_SLACK = 0.5  # a ray still hits a wall this far (m) beyond its observed ends: corners are seldom observed
MIN_ARC_DEG = 4.0  # a wall owns at least this much of the view to become an edge
MIN_EDGE, CORNER_CUT = 0.5, 0.9  # edges shorter than this go; up to CORNER_CUT when both neighbours meet nearby (layout.drop_short)
CORNER_REACH = 1.0  # two walls meet at their line intersection if it is within this (m) of both observed ends
HEIGHT_PLANES = 6
HEIGHT_THRESH, HEIGHT_RANGE_SLOPE = 0.08, 0.03  # floors and ceilings from depth are bowed: a wide inlier band
HEIGHT_TILT_DEG = 6.0
MIN_HEIGHT_FRAC = 0.04  # share of all points a floor or ceiling plane must carry
CEILING_RANGE = (2.0, 4.5)  # a floor-to-ceiling distance outside this is not reported
OPENING_WIDTH = (0.55, 1.8)  # doorway widths (m) looked for, as in the LiDAR tier's Params
SEEN_THROUGH_MIN, SEEN_THROUGH_BEHIND = 30, 0.4  # points at least this far (m) behind the wall inside the gap's view


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
    slab = slab[(slab[:, 1] > WALL_SLAB[0]) & (slab[:, 1] < WALL_SLAB[1])]
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


def _cast_rays(segs: list[WallSegment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit directions (R, 2), distance to the nearest wall along each (inf if none) and its index (-1 if none).

    Direction k is (sin a, cos a) with a the azimuth from +z towards +x, as in the free-space fan of photo_scene.
    """
    a = np.radians(np.arange(-180.0 + RAY_DEG / 2, 180.0, RAY_DEG))
    r = np.column_stack([np.sin(a), np.cos(a)])
    dist = np.full((len(segs), len(a)), np.inf)
    for i, s in enumerate(segs):
        den = r @ s.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            d = s.offset / den
        along = (d[:, None] * r) @ s.tangent
        ok = (den > 1e-3) & (along >= s.t0 - RAY_SLACK) & (along <= s.t1 + RAY_SLACK)
        dist[i, ok] = d[ok]
    nearest = dist.min(axis=0)
    return r, nearest, np.where(np.isfinite(nearest), dist.argmin(axis=0), -1)


def _cyclic_runs(owner: np.ndarray) -> tuple[list[list[int]], int]:
    """[[owner, first, last], ...] over the circle, indices relative to ``start`` (the first ray of the first run)."""
    n = len(owner)
    change = np.nonzero(owner != np.roll(owner, 1))[0]
    if len(change) == 0:
        return [[int(owner[0]), 0, n - 1]], 0
    start = int(change[0])
    rolled = np.roll(owner, -start)
    runs, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and rolled[j + 1] == rolled[i]:
            j += 1
        runs.append([int(rolled[i]), i, j])
        i = j + 1
    return runs, start


def _clean_runs(runs: list[list[int]], min_bins: int) -> list[list[int]]:
    """A wall that owns less than ``min_bins`` rays is noise: absorbed if the same wall is on both sides, else a gap."""
    while len(runs) > 1:
        short = next((k for k, (o, a, b) in enumerate(runs) if o >= 0 and b - a + 1 < min_bins), None)
        if short is None:
            break
        before, after = runs[short - 1][0], runs[(short + 1) % len(runs)][0]
        runs[short][0] = before if before == after else -1
        merged: list[list[int]] = []
        for run in runs:
            if merged and merged[-1][0] == run[0]:
                merged[-1][2] = run[2]
            else:
                merged.append(list(run))
        runs = merged
    return runs


def outline_from_segments(segs: list[WallSegment], grid: L.Grid) -> L.RoomOutline | None:
    """The visible boundary of the room from the station (the origin), as edges the LiDAR tier's report can read.

    Rays from the station hit the nearest wall; a wall's stretch of rays is one edge. Neighbouring walls meet
    at their line intersection (corners are seldom observed) when it lies near where their rays end; otherwise
    the boundary jumps, and an unsupported edge joins the two ends. None when fewer than two walls are visible.
    """
    if len(segs) < 2:
        return None
    r, dist, owner = _cast_rays(segs)
    n_rays = len(owner)
    runs, start = _cyclic_runs(owner)
    runs = [x for x in _clean_runs(runs, int(round(MIN_ARC_DEG / RAY_DEG))) if x[0] >= 0]
    while len(runs) > 1 and runs[0][0] == runs[-1][0]:  # one wall seen on both sides of the wrap
        runs[0][1] = runs[-1][1] - n_rays
        runs.pop()
    merged: list[list[int]] = []
    for run in runs:  # one wall on both sides of a gap (a doorway, or something unobserved) is one edge
        if merged and merged[-1][0] == run[0]:
            merged[-1][2] = run[2]
        else:
            merged.append(list(run))
    if len({o for o, _, _ in merged}) < 2:
        return None
    m = len(merged)
    first = [dist[(a + start) % n_rays] * r[(a + start) % n_rays] for _, a, _ in merged]
    last = [dist[(b + start) % n_rays] * r[(b + start) % n_rays] for _, _, b in merged]
    begin, end, jump = [None] * m, [None] * m, [False] * m
    for i in range(m):
        j = (i + 1) % m
        a, b = segs[merged[i][0]], segs[merged[j][0]]
        x = L.intersect((a.normal, a.offset), (b.normal, b.offset))
        if x is not None and np.linalg.norm(x - last[i]) <= CORNER_REACH and np.linalg.norm(x - first[j]) <= CORNER_REACH:
            end[i], begin[j] = x, x
        else:
            end[i], begin[j], jump[i] = last[i], first[j], True
    edges: list[L.Edge] = []
    for i in range(m):
        s = segs[merged[i][0]]
        lo, hi = sorted((float(begin[i] @ s.tangent), float(end[i] @ s.tangent)))
        covered = sum(max(0.0, min(hi, r1) - max(lo, r0)) for r0, r1 in s.runs)
        support = min(1.0, covered / (hi - lo)) if hi > lo else 0.0
        edges.append(L.Edge(begin[i], end[i], True, support, (s.normal.copy(), s.offset)))
        j = (i + 1) % m
        if jump[i] and np.linalg.norm(begin[j] - end[i]) > 0.05:
            edges.append(L.Edge(end[i], begin[j], False, 0.0, None))
    edges = L.drop_short(L.merge_collinear(edges), min_len=MIN_EDGE, corner_cut=CORNER_CUT)
    polygon = np.array([e.p0 for e in edges])
    if len(polygon) < 3:
        return None
    if L.signed_area(polygon) < 0:  # counter-clockwise, like the LiDAR outlines
        edges = [L.Edge(e.p1, e.p0, e.supported, e.support, e.line) for e in reversed(edges)]
        polygon = np.array([e.p0 for e in edges])
    pix = np.floor((polygon - [grid.x0, grid.z0]) / grid.cell).astype(np.int32)
    mask = np.zeros((grid.nz, grid.nx), np.uint8)
    cv2.fillPoly(mask, [pix.reshape(-1, 1, 2)], 1)
    return L.RoomOutline(mask.astype(bool), polygon, edges, 0)


def floor_and_ceiling(points: np.ndarray, seed: int = 0) -> tuple[HeightPlane | None, HeightPlane | None]:
    """Floor and ceiling heights (cloud frame, camera at y = 0) as the lowest area-like horizontal plane below the
    camera and the highest above it, found by RANSAC. Depth-model floors are bowed and furniture tops are horizontal
    too, so each plane must carry ``MIN_HEIGHT_FRAC`` of all points and spread over an area. Either may be None."""
    p = np.asarray(points, float)
    planes = fit_planes(
        p, max_planes=HEIGHT_PLANES, thresh=HEIGHT_THRESH, range_slope=HEIGHT_RANGE_SLOPE, min_inlier_frac=MIN_HEIGHT_FRAC,
        normal_hint=(0.0, 1.0, 0.0), max_angle_deg=HEIGHT_TILT_DEG, seed=seed,
    )
    found = []
    for pl in planes:
        inl = p[pl.inlier_mask]
        if not _is_area(inl):
            continue
        y = float(np.median(inl[:, 1]))
        spread = 1.4826 * float(np.median(np.abs(inl[:, 1] - y)))
        found.append(HeightPlane(y, spread, int(pl.inlier_mask.sum()), float(pl.inlier_mask.mean())))
    below = [h for h in found if h.y < -0.5]
    above = [h for h in found if h.y > 0.3]
    return (min(below, key=lambda h: h.y) if below else None), (max(above, key=lambda h: h.y) if above else None)


def _seen_through(slab_xz: np.ndarray, seg: WallSegment, a: float, b: float) -> bool:
    """True when points lie well behind the wall line inside the wedge the gap [a, b] subtends at the station."""
    pa = seg.offset * seg.normal + a * seg.tangent
    pb = seg.offset * seg.normal + b * seg.tangent
    side = np.sign(pa[0] * pb[1] - pa[1] * pb[0])
    if side == 0:
        return False
    inside = ((pa[0] * slab_xz[:, 1] - pa[1] * slab_xz[:, 0]) * side >= 0) & ((slab_xz[:, 0] * pb[1] - slab_xz[:, 1] * pb[0]) * side >= 0)
    behind = slab_xz @ seg.normal > seg.offset + SEEN_THROUGH_BEHIND
    return int((inside & behind).sum()) >= SEEN_THROUGH_MIN


def find_openings(segs: list[WallSegment], outline: L.RoomOutline, points: np.ndarray) -> list[OpeningEst]:
    """Doorways: a gap between two dense runs of one wall, of doorway width, that something was seen through.
    A gap nothing was seen through may be a door, a window or an unobserved patch, and is not reported."""
    p = np.asarray(points, float)
    slab = p[(p[:, 1] > WALL_SLAB[0]) & (p[:, 1] < WALL_SLAB[1])][:, [0, 2]]
    out: list[OpeningEst] = []
    for seg in segs:
        edge = next((i for i, e in enumerate(outline.edges) if e.line is not None
                     and np.allclose(e.line[0], seg.normal) and e.line[1] == seg.offset), None)
        if edge is None:
            continue
        for (_, a), (b, _) in zip(seg.runs, seg.runs[1:]):
            if OPENING_WIDTH[0] <= b - a <= OPENING_WIDTH[1] and _seen_through(slab, seg, a, b):
                out.append(OpeningEst(seg.offset * seg.normal + a * seg.tangent, seg.offset * seg.normal + b * seg.tangent, edge))
    return out
