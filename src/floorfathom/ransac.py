"""Robust plane fitting (RANSAC) and levelling. Arrays in, arrays out; no tier knowledge.

A least-squares plane is dragged by everything that is not on it (walls, furniture, depth
errors). RANSAC instead proposes planes from random point triples and keeps the one that most
points agree on, then refits on that plane's points only. Every draw comes from a seeded
generator, so the same points and seed always give the same planes.

A plane is ``normal @ x == offset`` with a unit normal.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

MAX_SCORED = 50_000  # hypotheses are scored on at most this many points; refits use all of them


class Plane(NamedTuple):
    normal: np.ndarray  # (3,) unit
    offset: float
    inlier_mask: np.ndarray  # (N,) bool, over the points passed in


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _tls(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Total least squares plane through p: the direction of least spread."""
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    return n, float(n @ c)


def _fit_one(points, thresh, range_slope, iters, seed, hint, max_angle_deg, hint_mode) -> Plane | None:
    n_pts = len(points)
    if n_pts < 3:
        return None
    rng = np.random.default_rng(seed)
    tol = thresh + range_slope * np.linalg.norm(points, axis=1)
    keep = rng.choice(n_pts, MAX_SCORED, replace=False) if n_pts > MAX_SCORED else np.arange(n_pts)
    sub, sub_tol = points[keep], tol[keep]

    normals, anchors = [], []
    for _ in range(10):  # a hint can reject many draws; redraw until enough are valid
        tri = points[rng.integers(0, n_pts, (iters, 3))]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1)
        ok = norm > 1e-9
        n = n[ok] / norm[ok, None]
        a = tri[ok, 0]
        if hint is not None:
            cos = np.abs(n @ hint)
            angle = np.radians(max_angle_deg)
            good = cos >= np.cos(angle) if hint_mode == "parallel" else cos <= np.sin(angle)
            n, a = n[good], a[good]
        normals.append(n)
        anchors.append(a)
        if sum(len(x) for x in normals) >= iters:
            break
    normals, anchors = np.concatenate(normals)[:iters], np.concatenate(anchors)[:iters]
    if len(normals) == 0:
        return None

    counts = [(np.abs(sub @ n - n @ a) < sub_tol).sum() for n, a in zip(normals, anchors)]
    best = int(np.argmax(counts))
    n, d = normals[best], float(normals[best] @ anchors[best])
    for _ in range(3):  # refit on the inliers, then recount them on all points
        mask = np.abs(points @ n - d) < tol
        if mask.sum() < 3:
            break
        n, d = _tls(points[mask])
    mask = np.abs(points @ n - d) < tol
    if hint is not None and hint_mode == "parallel":
        if n @ hint < 0:
            n, d = -n, -d
    elif d < 0:
        n, d = -n, -d
    return Plane(n, d, mask)


def fit_plane(points, *, thresh=0.02, iters=300, seed=0, normal_hint=None, max_angle_deg=None,
              hint_mode="parallel", range_slope=0.0) -> Plane | None:
    """The plane most points agree on, or None. See ``fit_planes`` for the arguments."""
    hint = None if normal_hint is None else _unit(normal_hint)
    if hint is not None and max_angle_deg is None:
        raise ValueError("normal_hint needs max_angle_deg")
    return _fit_one(np.asarray(points, float), thresh, range_slope, iters, seed, hint, max_angle_deg, hint_mode)


def fit_planes(points, *, max_planes=4, thresh=0.02, iters=300, min_inlier_frac=0.03, seed=0,
               normal_hint=None, max_angle_deg=None, hint_mode="parallel", range_slope=0.0) -> list[Plane]:
    """Up to ``max_planes`` planes, largest first; each is fitted to what the earlier ones left.

    thresh: inlier distance in the units of the points. range_slope adds this much per unit of
    distance from the origin (depth noise grows with range). normal_hint + max_angle_deg only
    accept planes whose normal is near +-hint (hint_mode "parallel": floor, ceiling) or near
    perpendicular to it ("perpendicular": vertical walls when the hint is a rough up). A plane
    needs at least min_inlier_frac of all points. Normals point along the hint when hint_mode is
    "parallel", otherwise away from the origin (offset >= 0). Masks index the input points, and
    a point belongs to at most one plane. The final refit can tilt a plane a little past
    max_angle_deg; the hint only steers the search.
    """
    pts = np.asarray(points, float)
    hint = None if normal_hint is None else _unit(normal_hint)
    if hint is not None and max_angle_deg is None:
        raise ValueError("normal_hint needs max_angle_deg")
    remaining = np.arange(len(pts))
    out: list[Plane] = []
    for k in range(max_planes):
        found = _fit_one(pts[remaining], thresh, range_slope, iters, seed + k, hint, max_angle_deg, hint_mode)
        if found is None or found.inlier_mask.sum() < max(3, min_inlier_frac * len(pts)):
            break
        mask = np.zeros(len(pts), bool)
        mask[remaining[found.inlier_mask]] = True
        out.append(Plane(found.normal, found.offset, mask))
        remaining = remaining[~found.inlier_mask]
    return out


def level(points, plane: Plane, up_hint, up=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Rotate so ``plane``'s normal becomes ``up``; the sign comes from ``up_hint``.

    Returns (rotated points, 3x3 rotation R), with rotated = points @ R.T.
    """
    n = plane.normal if plane.normal @ np.asarray(up_hint, float) >= 0 else -plane.normal
    a, b = _unit(n), _unit(up)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:  # already aligned, or exactly opposite
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)
    return np.asarray(points, float) @ R.T, R
