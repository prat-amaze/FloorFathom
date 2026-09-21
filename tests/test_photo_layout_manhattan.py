import numpy as np

from floorfathom import layout as L
from floorfathom.photo_layout import snap_manhattan


def _edge(p0, p1, angle_offset_deg=0.0, support=0.9):
    d = np.array(p1) - np.array(p0)
    length = float(np.linalg.norm(d))
    t = d / length
    n = np.array([-t[1], t[0]])
    if angle_offset_deg:
        c, s = np.cos(np.radians(angle_offset_deg)), np.sin(np.radians(angle_offset_deg))
        n = np.array([c * n[0] - s * n[1], s * n[0] + c * n[1]])
    return L.Edge(np.array(p0, float), np.array(p1, float), True, support, (n, float(n @ np.array(p0, float))))


def test_near_right_angle_corner_snaps_to_90_when_well_supported():
    edges = [_edge((0, 0), (4, 0), angle_offset_deg=0.0, support=0.9),
             _edge((4, 0), (4, 0.5), angle_offset_deg=2.0, support=0.9)]
    out = snap_manhattan(edges, tolerance_deg=12.0)
    n0, n1 = out[0].line[0], out[1].line[0]
    angle = np.degrees(np.arccos(np.clip(abs(n0 @ n1), 0, 1)))
    assert angle > 89.5


def test_genuinely_non_manhattan_corner_is_preserved():
    edges = [_edge((0, 0), (4, 0), angle_offset_deg=0.0, support=0.9),
             _edge((4, 0), (2.6, 2.75), angle_offset_deg=20.0, support=0.9)]
    out = snap_manhattan(edges, tolerance_deg=12.0)
    assert np.allclose(out[1].line[0], edges[1].line[0])


def test_closure_edges_are_never_snapped():
    edges = [_edge((0, 0), (4, 0), support=0.9),
             L.Edge(np.array([4.0, 0.0]), np.array([4.0, 3.0]), False, 0.0, None)]
    out = snap_manhattan(edges, tolerance_deg=12.0)
    assert out[1].line is None and out[1].supported is False


def test_weak_support_does_not_drag_a_well_supported_wall():
    edges = [_edge((0, 0), (4, 0), angle_offset_deg=0.0, support=0.95),
             _edge((4, 0), (4, 2.9), angle_offset_deg=5.0, support=0.1)]
    out = snap_manhattan(edges, tolerance_deg=12.0)
    n0 = out[0].line[0]
    assert np.allclose(n0, edges[0].line[0], atol=1e-6)
