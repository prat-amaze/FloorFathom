import numpy as np

from floorfathom.schema import Measurement, Opening, RoomPlan, Wall
from floorfathom.stitch import refine_global, stitch


def _room(id_, poly, openings):
    return RoomPlan(id=id_, name=id_, frame="room", polygon=poly, walls=[
        Wall(id=f"{id_}_w{i}", start=poly[i], end=poly[(i + 1) % len(poly)],
             length=Measurement(value=float(np.hypot(*(np.array(poly[(i + 1) % len(poly)]) - np.array(poly[i])))),
                                lo=None, hi=None, unit="m", method="m", note=None), evidence="wall_points")
        for i in range(len(poly))
    ], openings=openings, ceiling_height=Measurement(value=2.6, lo=None, hi=None, unit="m", method="m", note=None),
        floor_area=Measurement(value=None, lo=None, hi=None, unit="m2", method="m", note=None))


def _opening(id_, centre, wall_id, width=0.9):
    return Opening(id=id_, kind="doorway", wall_id=wall_id, centre=centre, start=(centre[0], centre[1] - width / 2),
                   end=(centre[0], centre[1] + width / 2), width=Measurement(value=width, lo=None, hi=None, unit="m", method="m", note=None))


def test_refine_keeps_sequential_solve_when_underconstrained():
    a = _room("a", [(0, 0), (3, 0), (3, 3), (0, 3)], [_opening("a_o0", (3.0, 1.5), "a_w1")])
    b = _room("b", [(0, 0), (3, 0), (3, 3), (0, 3)], [_opening("b_o0", (0.0, 1.5), "b_w3")])
    st = stitch([a, b])
    refined = refine_global(st.rooms, st.placements)
    assert refined == st.placements


def test_refine_rejects_outlier_correspondence():
    a = _room("a", [(0, 0), (3, 0), (3, 3), (0, 3)], [_opening("a_o0", (3.0, 1.5), "a_w1")])
    b = _room("b", [(0, 0), (2, 0), (2, 3), (0, 3)], [_opening("b_o0", (0.0, 1.5), "b_w3"), _opening("b_o1", (2.0, 1.5), "b_w1")])
    c = _room("c", [(0, 0), (3, 0), (3, 3), (0, 3)], [_opening("c_o0", (0.0, 1.5), "c_w3")])
    st = stitch([a, b, c])
    refined = refine_global(st.rooms, st.placements)
    assert len(refined) == len(st.placements)
    for p in refined:
        assert p.overlap <= 0.05
