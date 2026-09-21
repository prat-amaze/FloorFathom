"""Merge the pieces of one physical wall in a video room outline.

The polygon of a room is cut from a raster mask, so a straight wall whose depth-model points bow by a few
centimetres comes out as several short edges that leave the straight line by more than the simplification
tolerance. A person measuring the room with a tape sees one wall. Two consecutive supported edges are one wall
when one straight line explains the wall points of both about as well as each explains its own:

  * the union of the points near the two edges is fitted by total least squares,
  * the RMS distance of those points to that line must stay within ``max_rms`` or 1.5 times the worse of the two
    single-edge RMS values (the wall's own thickness under depth noise sets what "straight" means),
  * the two edges must not turn by more than ``max_turn_deg`` (a real corner turns 30 degrees or more; a bowed
    wall turns a few degrees),
  * the fitted line must pass close to the outer ends of both edges.

A merged wall takes the fitted line and its end points are re-derived where it meets its neighbours
(``layout.rebuild_polygon``). Unsupported edges (doorways, unseen boundary) are never merged.
"""

from __future__ import annotations

import numpy as np

from . import layout as L

NEAR = 0.15  # points within this of an edge's own line, along it, belong to it (m)
MAX_RMS = 0.04  # m
GROWTH = 1.5  # the union may be this much thicker than the worse single edge
MAX_TURN_DEG = 30.0  # half of a bevelled corner (45 degrees): a bowed wall turns less
MAX_END_OFFSET = 0.15  # the merged line must pass this close to the outer ends of the two edges (m)
MIN_POINTS = 20


def _points_near(edge: L.Edge, pts: np.ndarray, near: float = NEAR) -> np.ndarray:
    d = edge.p1 - edge.p0
    length = float(np.linalg.norm(d))
    if length < 1e-6 or len(pts) == 0:
        return pts[:0]
    t = d / length
    n = np.array([-t[1], t[0]])
    rel = pts - edge.p0
    along, perp = rel @ t, rel @ n
    if edge.line is not None:  # the fitted line, not the chord, is where the wall is
        perp = pts @ edge.line[0] - edge.line[1]
    return pts[(along > -0.1) & (along < length + 0.1) & (np.abs(perp) <= near)]


def _tls(p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(unit normal, offset, RMS distance) of the total least squares line through 2-D points."""
    mu = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - mu, full_matrices=False)
    n = np.array([-vt[0][1], vt[0][0]])
    res = (p - mu) @ n
    return n, float(n @ mu), float(np.sqrt(np.mean(res**2)))


def _try_merge(a: L.Edge, b: L.Edge, pts: np.ndarray, max_rms: float, max_turn_deg: float) -> tuple[float, L.Edge] | None:
    if a.line is None or b.line is None:
        return None
    if abs(float(a.line[0] @ b.line[0])) < np.cos(np.radians(max_turn_deg)):
        return None
    pa, pb = _points_near(a, pts), _points_near(b, pts)
    if len(pa) < MIN_POINTS or len(pb) < MIN_POINTS:
        return None
    ra, rb = _tls(pa)[2], _tls(pb)[2]
    union = np.unique(np.vstack([pa, pb]), axis=0)
    n, c, rms = _tls(union)
    if rms > max(max_rms, GROWTH * max(ra, rb)):
        return None
    if max(abs(float(n @ a.p0) - c), abs(float(n @ b.p1) - c)) > MAX_END_OFFSET:
        return None
    return rms, L.Edge(a.p0, b.p1, True, max(a.support, b.support), (n, c))


def consolidate_walls(
    edges: list[L.Edge],
    wall_pts: np.ndarray,
    max_rms: float = MAX_RMS,
    max_turn_deg: float = MAX_TURN_DEG,
) -> list[L.Edge]:
    """``edges`` (a closed outline, in order) with the pieces of each straight wall merged into one edge.

    ``wall_pts`` are the (x, z) wall points near the room, as the estimator hands them to ``build_outline``. The
    cheapest merge (lowest union RMS) is applied first and the search repeats until none is left. The result is a new
    list; the corners are re-derived from the merged lines.
    """
    edges = [L.Edge(e.p0.copy(), e.p1.copy(), e.supported, e.support, e.line) for e in edges]
    pts = np.asarray(wall_pts, float)
    merged_any = False
    while len(edges) > 3:
        best = None
        for i in range(len(edges)):
            j = (i + 1) % len(edges)
            got = _try_merge(edges[i], edges[j], pts, max_rms, max_turn_deg)
            if got is not None and (best is None or got[0] < best[0]):
                best = (got[0], i, j, got[1])
        if best is None:
            break
        _, i, j, edge = best
        edges[i] = edge
        del edges[j]
        merged_any = True
    if not merged_any:
        return edges
    verts = L.rebuild_polygon(edges)
    for i, e in enumerate(edges):
        e.p0, e.p1 = verts[i], verts[(i + 1) % len(edges)]
    return edges
