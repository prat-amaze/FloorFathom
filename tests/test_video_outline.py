"""Merging the pieces of one wall in a video outline (video_outline.consolidate_walls)."""

from __future__ import annotations

import numpy as np

from floorfathom import layout as L
from floorfathom.video_outline import consolidate_walls

RNG = np.random.default_rng(0)


def _wall(p0, p1, bow: float = 0.0, n: int = 600, noise: float = 0.015) -> np.ndarray:
    """Points along the wall p0 -> p1, bowed sideways by up to ``bow`` metres in the middle, with depth noise."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = RNG.uniform(0, 1, n)
    d = p1 - p0
    side = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return p0 + t[:, None] * d + (bow * np.sin(np.pi * t) + RNG.normal(0, noise, n))[:, None] * side


def _edge(p0, p1, pts: np.ndarray) -> L.Edge:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    t = d / np.linalg.norm(d)
    n = np.array([-t[1], t[0]])
    near = pts[(np.abs((pts - p0) @ n) < 0.2) & ((pts - p0) @ t > -0.1) & ((pts - p0) @ t < np.linalg.norm(d) + 0.1)]
    return L.Edge(p0, p1, True, 1.0, L.fit_line(near))


def test_a_bowed_wall_cut_into_pieces_is_one_wall_again():
    # 5 x 4 m room; the bottom wall (z = 0) is bowed by 4 cm and came out as three edges
    pts = np.vstack([_wall((0, 0), (5, 0), bow=0.04, n=1500), _wall((5, 0), (5, 4)), _wall((5, 4), (0, 4)), _wall((0, 4), (0, 0))])
    bottom = [(0.0, 1.7), (1.7, 3.3), (3.3, 5.0)]
    edges = [_edge((a, 0.0), (b, 0.0), pts) for a, b in bottom]
    edges += [_edge((5, 0), (5, 4), pts), _edge((5, 4), (0, 4), pts), _edge((0, 4), (0, 0), pts)]
    out = consolidate_walls(edges, pts)
    assert len(out) == 4
    lengths = sorted(e.length for e in out)
    assert abs(lengths[-1] - 5.0) < 0.1 and abs(lengths[0] - 4.0) < 0.1
    assert abs(L.signed_area(np.array([e.p0 for e in out])) - 20.0) < 0.5


def test_corners_are_kept_a_bevel_and_an_l_shape_do_not_merge():
    # L-shaped room, every corner 90 degrees, plus the bevelled (45 degree) corner of a 5 x 4 m room
    l_room = [(0, 0), (5, 0), (5, 2), (2.5, 2), (2.5, 4), (0, 4)]
    pts = np.vstack([_wall(l_room[i], l_room[(i + 1) % 6]) for i in range(6)])
    edges = [_edge(l_room[i], l_room[(i + 1) % 6], pts) for i in range(6)]
    assert len(consolidate_walls(edges, pts)) == 6
    bevel = [(0, 0), (5, 0), (5, 3), (4, 4), (0, 4)]
    pts = np.vstack([_wall(bevel[i], bevel[(i + 1) % 5]) for i in range(5)])
    edges = [_edge(bevel[i], bevel[(i + 1) % 5], pts) for i in range(5)]
    assert len(consolidate_walls(edges, pts)) == 5


def test_a_doorway_and_thin_evidence_are_never_merged_across():
    pts = np.vstack([_wall((0, 0), (5, 0)), _wall((5, 0), (5, 4)), _wall((5, 4), (0, 4)), _wall((0, 4), (0, 0))])
    edges = [_edge((0, 0), (2, 0), pts), L.Edge(np.array([2.0, 0.0]), np.array([3.0, 0.0]), False, 0.0, None), _edge((3, 0), (5, 0), pts)]
    edges += [_edge((5, 0), (5, 4), pts), _edge((5, 4), (0, 4), pts), _edge((0, 4), (0, 0), pts)]
    assert len(consolidate_walls(edges, pts)) == 6  # the doorway edge has no line: nothing merges over it
    assert len(consolidate_walls(edges, pts[:10])) == 6  # too few points to say two edges are one wall
