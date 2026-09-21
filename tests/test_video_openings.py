"""Doorways from camera rays: a 5 x 4 m room whose right wall (x = W) has an opening onto a second room behind it."""

from __future__ import annotations

import numpy as np

from floorfathom.layout import Edge
from floorfathom.video_openings import find_openings
from floorfathom.video_rays import RayBundle

W, D, H = 5.0, 4.0, 2.6
BACK = 3.0  # the room behind the right wall spans x in [W, W + BACK]
CAM_Y = 1.4


def _cast(o: np.ndarray, ang: float, elev: float, gap: tuple[float, float] | None, sill: float) -> np.ndarray:
    """End point of the ray from o = (x, z) at azimuth ``ang`` and elevation ``elev``: the first of the room's walls,
    the opening in the wall x = W (z in ``gap``, from height ``sill`` up), the back room's walls, floor and ceiling."""
    d = np.array([np.cos(ang), np.sin(ang)])
    t_tan = np.tan(elev)
    limit = np.inf
    if t_tan < 0:
        limit = min(limit, CAM_Y / -t_tan)
    elif t_tan > 0:
        limit = min(limit, (H - CAM_Y) / t_tan)

    def first(x_max: float, through_gap: bool) -> float:
        cands = []
        for axis, bound in ((0, 0.0), (0, x_max), (1, 0.0), (1, D)):
            if abs(d[axis]) > 1e-9:
                t = (bound - o[axis]) / d[axis]
                if t > 1e-6:
                    cands.append((t, axis, bound))
        t, axis, bound = min(cands)
        return t if not (through_gap and axis == 0 and bound == x_max) else -1.0

    r = first(W, False)
    if gap is not None and r > 0:
        p = o + r * d
        if abs(p[0] - W) < 1e-6 and gap[0] <= p[1] <= gap[1] and CAM_Y + r * t_tan >= sill and r < limit:
            # through the opening: on to the walls of the back room
            cands = []
            for axis, bound in ((0, W + BACK), (1, 0.0), (1, D)):
                if abs(d[axis]) > 1e-9:
                    t = (bound - o[axis]) / d[axis]
                    if t > r:
                        cands.append(t)
            r = min(cands)
    r = min(r, limit)
    return np.array([o[0] + r * d[0], CAM_Y + r * t_tan, o[1] + r * d[1]])


def _bundle(gap: tuple[float, float] | None, sill: float = 0.0, frames: int = 40, per_frame: int = 300) -> RayBundle:
    rng = np.random.default_rng(0)
    origins, ends = [], []
    for k in range(frames):
        o = np.array([2.0 + 2.0 * rng.random(), 1.0 + 2.0 * rng.random()])
        e = [_cast(o, rng.uniform(0, 2 * np.pi), rng.uniform(-0.7, 0.7), gap, sill) for _ in range(per_frame)]
        origins.append([o[0], CAM_Y, o[1]])
        ends.append(np.array(e, np.float32))
    return RayBundle(np.array(origins), ends, (np.arange(frames) * 4 // frames).astype(np.int16), 4)


def _edges() -> list[Edge]:
    corners = np.array([[0.0, 0.0], [W, 0.0], [W, D], [0.0, D]])
    return [Edge(corners[i], corners[(i + 1) % 4], True, 1.0) for i in range(4)]


def _find(bundle: RayBundle):
    return find_openings(_edges(), bundle, np.eye(3), 1.0, floor_y=0.0)


def test_a_door_gap_in_the_wall_is_found_with_its_width():
    found = _find(_bundle((1.5, 2.31)))
    assert [o.edge_index for o in found] == [1]
    assert abs(found[0].width - 0.81) < 0.06
    assert abs(found[0].centre[1] - 1.905) < 0.05


def test_a_solid_wall_has_no_opening():
    assert _find(_bundle(None)) == []


def test_a_window_is_not_a_door():
    # the opening starts 0.95 m above the floor: every ray through it crosses the wall above the door band
    assert _find(_bundle((1.0, 2.2), sill=0.95)) == []


def test_a_wide_glass_door_is_one_opening():
    found = _find(_bundle((1.1, 2.9)))
    assert len(found) == 1
    assert abs(found[0].width - 1.8) < 0.08
