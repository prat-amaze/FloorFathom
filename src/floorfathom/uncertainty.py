"""Intervals by delete-d jackknife over frame chunks, plus an assumed systematic term.

The scan is split into contiguous chunks of frames. Each replicate leaves ``drop`` random
chunks out, re-runs the whole estimator and matches the resulting rooms, walls and
openings back to the full-data estimate. The spread of a quantity across replicates,
scaled by ``jackknife_scale``, is its sampling uncertainty. Chunks, not single frames,
are left out because neighbouring frames see nearly the same surface and would give
overconfident intervals.

What this does NOT see is systematic error (pose drift, scale bias, a wall whose plaster
is thicker than the lidar can resolve). Those are covered by an assumed systematic term
added in quadrature. The constants below are assumptions to be calibrated against tape
measurements, not measured values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estimate import Estimate, Params, estimate
from .points import Cloud

Z95 = 1.96
# (absolute metres, relative fraction), added in quadrature to the sampling term
SYS_LENGTH = (0.015, 0.005)  # depth noise after line fitting, pose scale drift
SYS_HEIGHT = (0.015, 0.003)  # floor/ceiling plane offset
SYS_AREA_REL = 0.01
SYS_OPENING = (0.03, 0.01)  # 5 cm grid plus door-frame endpoint ambiguity


@dataclass
class RoomSamples:
    area: list[float] = field(default_factory=list)
    ceiling_height: list[float] = field(default_factory=list)
    wall_length: dict[int, list[float]] = field(default_factory=dict)
    opening_width: dict[int, list[float]] = field(default_factory=dict)
    matched: int = 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _match_edge(ref, edges, max_offset: float = 0.15, min_cos: float = 0.98) -> float | None:
    """Length of the replicate edge that lies on the same wall as ``ref``."""
    d = ref.p1 - ref.p0
    L = np.linalg.norm(d)
    if L < 1e-6:
        return None
    t = d / L
    n = np.array([-t[1], t[0]])
    best, best_overlap = None, 0.0
    for e in edges:
        de = e.p1 - e.p0
        Le = np.linalg.norm(de)
        if Le < 1e-6 or abs(float(t @ (de / Le))) < min_cos:
            continue
        if abs(float(n @ (0.5 * (e.p0 + e.p1) - ref.p0))) > max_offset:
            continue
        a0, a1 = sorted([float((e.p0 - ref.p0) @ t), float((e.p1 - ref.p0) @ t)])
        overlap = max(0.0, min(a1, L) - max(a0, 0.0))
        if overlap > best_overlap and overlap >= 0.5 * min(L, Le):
            best, best_overlap = Le, overlap
    return best


def _resample_points(cloud: Cloud, order: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.concatenate([cloud.points[bounds[c] : bounds[c + 1]] for c in order])


def jackknife_scale(n_chunks: int, drop: int) -> float:
    """Factor turning the spread of delete-``drop`` replicates into a standard error.

    Delete-d jackknife: Var(estimate) = (n - d) / d * spread^2 of the replicates.
    """
    return float(np.sqrt((n_chunks - drop) / drop))


def bootstrap(
    cloud: Cloud,
    traj_xz: np.ndarray,
    ref: Estimate,
    params: Params,
    replicates: int = 20,
    seed: int = 0,
    drop: int = 2,
) -> list[RoomSamples]:
    """Samples of every reference quantity, in the order of ``ref.rooms``.

    Each replicate leaves ``drop`` random chunks out (delete-d jackknife). Resampling
    chunks *with* replacement was tried first and abandoned: it removes about a third
    of the chunks per replicate, so the spread mostly measured lost coverage
    (room areas of [4, 27] m2) and not measurement noise.
    """
    rng = np.random.default_rng(seed)
    bounds = np.searchsorted(cloud.chunk, np.arange(cloud.n_chunks + 1))
    out = [RoomSamples() for _ in ref.rooms]
    for _ in range(replicates):
        order = np.sort(rng.choice(cloud.n_chunks, size=cloud.n_chunks - drop, replace=False))
        est = estimate(_resample_points(cloud, order, bounds), traj_xz, params, grid=ref.grid)
        for k, rr in enumerate(ref.rooms):
            best, best_iou = None, 0.5
            for cand in est.rooms:
                v = _iou(rr.outline.mask, cand.outline.mask)
                if v > best_iou:
                    best, best_iou = cand, v
            if best is None:
                continue
            s = out[k]
            s.matched += 1
            s.area.append(best.area)
            if best.ceiling_height is not None:
                s.ceiling_height.append(best.ceiling_height)
            for i, e in enumerate(rr.outline.edges):
                length = _match_edge(e, best.outline.edges)
                if length is not None:
                    s.wall_length.setdefault(i, []).append(length)
            for j, o in enumerate(rr.openings):
                cands = [c for c in best.openings if np.linalg.norm(c.centre - o.centre) <= 0.4]
                if cands:
                    c = min(cands, key=lambda c: np.linalg.norm(c.centre - o.centre))
                    s.opening_width.setdefault(j, []).append(c.width)
    return out


def interval(
    value: float, samples: list[float], sys_abs: float, sys_rel: float, scale: float = 1.0
) -> tuple[float, float, bool]:
    """(lo, hi, stable) for one quantity.

    Sampling term: robust spread (MAD) of the replicates times ``scale`` (see
    ``jackknife_scale``), so one or two replicates where the room came out differently
    do not dominate. Systematic term: see the constants at the top of this module.

    ``stable`` is False when fewer than 5 replicates matched, or when 15% or more of
    them disagree with the bulk by a wide margin (the shape flipped between
    replicates). In that case the interval is stretched to the 5th-95th percentile of
    the replicates so the disagreement is visible in the numbers, not only in a flag.
    """
    sys_term = float(np.hypot(sys_abs, sys_rel * abs(value)))
    if len(samples) >= 5:
        s = np.asarray(samples, dtype=float)
        med = float(np.median(s))
        sd = 1.4826 * float(np.median(np.abs(s - med)))
        half = float(np.hypot(Z95 * scale * sd, sys_term))
        lo, hi = value - half, value + half
        far = np.abs(s - med) > max(4 * sd, half, 0.03 * abs(value))
        stable = float(far.mean()) < 0.15
        if not stable:
            lo, hi = min(lo, float(np.percentile(s, 5))), max(hi, float(np.percentile(s, 95)))
    else:
        # too few matches to trust a spread: fall back to a wide, honest interval
        half = float(np.hypot(0.05 * abs(value), sys_term))
        lo, hi = value - half, value + half
        stable = False
    if value >= 0:
        lo = max(lo, 0.0)
    return lo, hi, stable
