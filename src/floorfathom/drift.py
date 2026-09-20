"""Drift correction for one continuous multi-room walk.

Visual-inertial poses accumulate error, so the same wall seen minutes apart lands in two
slightly different places. Every time chunk of the cloud is registered against the others
in the plan view (wall points only, point to line), and one small (x, z, yaw) correction
per chunk is solved by least squares so that all pairs agree as well as possible.

Each pairwise measurement is weighted by how well the geometry constrains it: two chunks
that overlap only along parallel walls fix the distance between the walls but not the
sliding along them, and the information matrix says so, so a degenerate pair cannot invent
a shift. Chunk 0 is the reference; height is untouched.

The tolerances below (MAX_SHIFT, MAX_YAW, STEP_SIGMA, SIGMA) are assumptions, not
calibrated against a device. Verified on synthetic drift only (tests/test_drift.py); on the
real Cozmo scans, block-wise correction did not yet improve how well two walks agree
(scripts/cross_scan_repeatability.py --drift), so the pipeline does not apply it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from .layout import BAND_H, BAND_HI, BAND_LO, CELL
from .points import Cloud

MIN_BANDS = 5  # a wall cell is occupied in at least this many 15 cm height bands
GATE = 0.15  # metres, nearest-neighbour gate of the registration
MIN_MATCHES = 40
MIN_INLIER = 0.25  # fraction of a chunk's wall cells that find a partner
SIGMA = 0.03  # metres, assumed noise of a wall point
MAX_SHIFT = 0.5  # metres: a larger pairwise shift is a wrong match, not drift
MAX_YAW = np.radians(3.0)
STEP_SIGMA = np.array([0.05, 0.05, np.radians(0.5)])  # assumed drift between neighbouring chunks


@dataclass
class Drift:
    centre: np.ndarray  # (2,) metres; the yaw corrections rotate about this point
    shifts: np.ndarray  # (K, 3) per chunk: dx, dz in metres and yaw in radians, to add
    n_pairs: int
    chi2_before: float  # weighted disagreement of all pairs, uncorrected
    chi2_after: float

    @property
    def max_shift(self) -> float:
        return float(np.abs(self.shifts[:, :2]).max())

    @property
    def max_yaw_deg(self) -> float:
        return float(np.degrees(np.abs(self.shifts[:, 2]).max()))


def _wall_cells(points: np.ndarray, floor_y: float) -> np.ndarray:
    """Mean (x, z) of the points in each cell occupied over most of the wall height range.

    The mean, not the cell centre: a 5 cm cell cannot show a 2 cm shift, its mean can.
    """
    nb = int(round((BAND_HI - BAND_LO) / BAND_H))
    band = np.floor((points[:, 1] - floor_y - BAND_LO) / BAND_H).astype(np.int64)
    ok = (band >= 0) & (band < nb)
    xz = points[ok][:, [0, 2]]
    if len(xz) == 0:
        return np.zeros((0, 2))
    keys, inv = np.unique(np.floor(xz / CELL).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    bits = np.zeros(len(keys), np.int64)
    np.bitwise_or.at(bits, inv, 1 << band[ok])
    n_bands = sum((bits >> b) & 1 for b in range(nb))
    counts = np.bincount(inv)
    mean = np.stack([np.bincount(inv, weights=xz[:, a]) / counts for a in range(2)], axis=1)
    return mean[n_bands >= MIN_BANDS]


def _normals(cells: np.ndarray) -> np.ndarray:
    """Unit normal of the local line through each cell's 8 nearest neighbours."""
    _, nb = cKDTree(cells).query(cells, k=min(8, len(cells)))
    q = cells[nb] - cells[nb].mean(axis=1, keepdims=True)
    _, v = np.linalg.eigh(np.einsum("mki,mkj->mij", q, q))
    return v[:, :, 0]


def _register(src: np.ndarray, dst: np.ndarray, normals: np.ndarray, iters: int = 30):
    """Point-to-line registration of src onto dst, from identity.

    Returns (m, H, inlier fraction): m = (dx, dz, yaw) that moves src onto dst, about the
    origin, and H the 3x3 information matrix of that estimate. None if too little overlaps.
    """
    tree = cKDTree(dst)
    cur, t, theta = src.copy(), np.zeros(2), 0.0
    for _ in range(iters):
        d, idx = tree.query(cur)
        ok = d < GATE
        if ok.sum() < MIN_MATCHES:
            return None
        p, q, n = cur[ok], dst[idx[ok]], normals[idx[ok]]
        jac = np.column_stack([n[:, 0], n[:, 1], n[:, 1] * p[:, 0] - n[:, 0] * p[:, 1]])
        r = ((p - q) * n).sum(axis=1)
        a = jac.T @ jac
        step = np.linalg.solve(a + 1e-3 * np.trace(a) / 3 * np.eye(3), -jac.T @ r)
        c, s = np.cos(step[2]), np.sin(step[2])
        rot = np.array([[c, -s], [s, c]])
        cur = cur @ rot.T + step[:2]
        t = rot @ t + step[:2]
        theta += step[2]
        if np.abs(step[:2]).max() < 1e-4 and abs(step[2]) < 1e-5:
            break
    d, idx = tree.query(cur)
    ok = d < GATE
    if ok.sum() < MIN_MATCHES:
        return None
    p, n = cur[ok], normals[idx[ok]]
    jac = np.column_stack([n[:, 0], n[:, 1], n[:, 1] * p[:, 0] - n[:, 0] * p[:, 1]])
    return np.array([t[0], t[1], theta]), jac.T @ jac / SIGMA**2, float(ok.mean())


def estimate_drift(cloud: Cloud, floor_y: float) -> Drift:
    """One (x, z, yaw) correction per chunk, from all chunk pairs that overlap."""
    k = cloud.n_chunks
    centre = cloud.points[:, [0, 2]].mean(axis=0).astype(np.float64)
    cells, normals = [], []
    for c in range(k):
        w = _wall_cells(cloud.points[cloud.chunk == c].astype(np.float64), floor_y) - centre
        cells.append(w)
        normals.append(_normals(w) if len(w) >= 8 else w)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            if len(cells[i]) < MIN_MATCHES or len(cells[j]) < MIN_MATCHES:
                continue
            got = _register(cells[j], cells[i], normals[i])
            if got is None or got[2] < MIN_INLIER:
                continue
            m, h, _ = got
            if np.linalg.norm(m[:2]) > MAX_SHIFT or abs(m[2]) > MAX_YAW:
                continue
            pairs.append((i, j, m, h))

    a = np.zeros((3 * k, 3 * k))
    b = np.zeros(3 * k)

    def add(i: int, j: int, m: np.ndarray, h: np.ndarray) -> None:
        si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
        a[si, si] += h
        a[sj, sj] += h
        a[si, sj] -= h
        a[sj, si] -= h
        b[sj] += h @ m
        b[si] -= h @ m

    for i, j, m, h in pairs:
        add(i, j, m, h)
    for c in range(k - 1):  # weak prior: neighbouring chunks drift by little
        add(c, c + 1, np.zeros(3), np.diag(1.0 / STEP_SIGMA**2))
    a[:3, :3] += 1e6 * np.eye(3)  # chunk 0 is the reference
    x = np.linalg.solve(a, b).reshape(k, 3)

    def chi2(shifts: np.ndarray) -> float:
        return float(
            sum((shifts[j] - shifts[i] - m) @ h @ (shifts[j] - shifts[i] - m) for i, j, m, h in pairs)
        )

    return Drift(centre, x, len(pairs), chi2(np.zeros((k, 3))), chi2(x))


def _apply(xz: np.ndarray, shifts: np.ndarray, centre: np.ndarray) -> np.ndarray:
    rel = xz.astype(np.float64) - centre
    c, s = np.cos(shifts[:, 2]), np.sin(shifts[:, 2])
    x = c * rel[:, 0] - s * rel[:, 1] + shifts[:, 0]
    z = s * rel[:, 0] + c * rel[:, 1] + shifts[:, 1]
    return np.stack([x, z], axis=1) + centre


def correct_points(cloud: Cloud, drift: Drift) -> Cloud:
    pts = cloud.points.copy()
    xz = _apply(cloud.points[:, [0, 2]], drift.shifts[cloud.chunk], drift.centre)
    pts[:, 0], pts[:, 2] = xz[:, 0], xz[:, 1]
    return replace(cloud, points=pts)


def correct_trajectory(traj_xz: np.ndarray, drift: Drift) -> np.ndarray:
    """Frames map to chunks by their share of the walk, as in ``build_cloud``."""
    k = len(drift.shifts)
    chunk = np.minimum(np.arange(len(traj_xz)) * k // len(traj_xz), k - 1)
    return _apply(traj_xz, drift.shifts[chunk], drift.centre)
