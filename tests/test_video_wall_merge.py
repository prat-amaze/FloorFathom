"""Synthetic tests for _augment_unsupported_edges + merge_collinear in video_pipeline.

Test 1: Three consecutive edges on the same physical wall, the middle one unsupported
        (line=None). After augmentation + merge the 5-edge outline reduces to 3.

Test 2: Middle unsupported edge is far from all wall lines (>0.25 m). It keeps
        line=None and nothing merges.
"""

from __future__ import annotations

import numpy as np

from floorfathom.layout import Edge, WallLine, merge_collinear
from floorfathom.video_pipeline import _augment_unsupported_edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge(p0, p1, *, line=None) -> Edge:
    """Make an Edge; supported=True when line is provided."""
    return Edge(
        p0=np.array(p0, float),
        p1=np.array(p1, float),
        supported=line is not None,
        support=1.0 if line is not None else 0.0,
        line=line,
    )


def _wall_line(normal, offset) -> WallLine:
    """WallLine with dummy extent parameters."""
    n = np.array(normal, float)
    return WallLine(normal=n, offset=float(offset), t0=0.0, t1=5.0)


# ---------------------------------------------------------------------------
# Test 1 — fragments merge when the middle edge is assigned a wall line
# ---------------------------------------------------------------------------

def test_augment_merges_fragments_on_same_wall():
    """5-edge outline with 3 collinear edges on the north wall (z = 3.0).

    Layout (looking down):
        south wall:  (0,0) -> (4,0)   — one edge
        east wall:   (4,0) -> (4,3)   — one edge
        north wall:  (4,3) -> (2,3)   supported  (line set)
                     (2,3) -> (1,3)   UNSUPPORTED (line=None) — simulates a sparse patch
                     (1,3) -> (0,3)   supported  (line set)
        west wall:   (0,3) -> (0,0)   — one edge

    Total: 6 edges before augmentation.  After augment + merge: 4.
    """
    north_normal = np.array([0.0, 1.0])  # points in +z direction
    north_offset = 3.0                   # z = 3.0

    north_line = (north_normal, north_offset)

    edges = [
        _edge([0, 0], [4, 0], line=(np.array([0.0, -1.0]), 0.0)),  # south
        _edge([4, 0], [4, 3], line=(np.array([1.0, 0.0]), 4.0)),   # east
        _edge([4, 3], [2, 3], line=north_line),                      # north-right (supported)
        _edge([2, 3], [1, 3]),                                        # north-mid  (unsupported)
        _edge([1, 3], [0, 3], line=north_line),                      # north-left (supported)
        _edge([0, 3], [0, 0], line=(np.array([-1.0, 0.0]), 0.0)),  # west
    ]

    wall_lines = [_wall_line([0.0, 1.0], 3.0)]  # the physical north wall

    assert len(edges) == 6
    assert edges[3].line is None  # middle north edge is unsupported

    _augment_unsupported_edges(edges, wall_lines)

    # After augmentation the middle north edge must have a line now
    assert edges[3].line is not None, "middle edge should have been assigned a wall line"
    assert edges[3].supported is True

    # After merge, the three north edges collapse to one
    merged = merge_collinear(edges)
    assert len(merged) == 4, f"expected 4 edges after merge, got {len(merged)}"

    # The remaining north edge should still have a line (the wall line)
    north_edges = [e for e in merged if abs(e.p0[1] - 3.0) < 0.01 and abs(e.p1[1] - 3.0) < 0.01]
    assert len(north_edges) == 1, "north wall should be a single edge after merge"
    assert north_edges[0].line is not None


# ---------------------------------------------------------------------------
# Test 2 — far edge keeps line=None; nothing merges
# ---------------------------------------------------------------------------

def test_augment_does_not_match_edge_far_from_all_wall_lines():
    """Middle edge is > 0.25 m from every wall line — must not be assigned one.

    Same 5-edge room but the middle 'north' edge is actually at z = 2.5,
    not z = 3.0.  The only wall line is at z = 3.0 (distance 0.5 m > _AUG_DIST_M).
    """
    north_line = (np.array([0.0, 1.0]), 3.0)

    edges = [
        _edge([0, 0], [4, 0], line=(np.array([0.0, -1.0]), 0.0)),   # south
        _edge([4, 0], [4, 3], line=(np.array([1.0, 0.0]), 4.0)),    # east
        _edge([4, 3], [2, 3], line=north_line),                       # north-right (supported)
        _edge([2, 2.5], [1, 2.5]),                                     # stray edge far from wall
        _edge([1, 3], [0, 3], line=north_line),                       # north-left (supported)
        _edge([0, 3], [0, 0], line=(np.array([-1.0, 0.0]), 0.0)),   # west
    ]

    wall_lines = [_wall_line([0.0, 1.0], 3.0)]  # wall at z = 3.0

    _augment_unsupported_edges(edges, wall_lines)

    # The stray edge is 0.5 m from the wall line — must stay unsupported
    assert edges[3].line is None, "edge far from wall line must not be assigned one"
    assert edges[3].supported is False

    # Nothing should merge incorrectly: count must stay at 6
    merged = merge_collinear(edges)
    assert len(merged) == 6, f"no merge expected, got {len(merged)} edges"
