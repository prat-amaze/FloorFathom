"""Adjacency, overlap and footprint of rooms laid out by hand (truth is known)."""

import numpy as np
import pytest

from floorfathom.schema import Measurement, Opening, RoomPlan
from floorfathom.stitch import adjacency, footprint_area, overlap_area, overlaps

WALL = 0.15  # metres between the inner faces of two neighbouring rooms


def _m(v: float, unit: str = "m") -> Measurement:
    return Measurement(value=v, lo=v, hi=v, unit=unit, method="test")


def _door(id: str, at: tuple[float, float]) -> Opening:
    return Opening(id=id, kind="doorway", wall_id="w0", start=at, end=at, centre=at, width=_m(0.9))


def _room(id: str, x0: float, y0: float, w: float, d: float, doors=()) -> RoomPlan:
    poly = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + d), (x0, y0 + d)]  # counter-clockwise
    return RoomPlan(
        id=id, polygon=poly, walls=[], ceiling_height=_m(2.5), floor_area=_m(w * d, "m2"),
        openings=[_door(f"{id}_{i}", at) for i, at in enumerate(doors)],
    )


def _pair() -> list[RoomPlan]:
    """Two 4 x 3 m rooms with a wall between them and a doorway through it."""
    a = _room("a", 0.0, 0.0, 4.0, 3.0, doors=[(4.0, 1.5)])
    b = _room("b", 4.0 + WALL, 0.0, 4.0, 3.0, doors=[(4.0 + WALL, 1.5)])
    return [a, b]


def test_doorway_links_to_the_room_behind_it():
    links = {ln.room: ln for ln in adjacency(_pair())}
    assert links["a"].other == "b" and links["b"].other == "a"
    assert links["a"].gap == pytest.approx(WALL, abs=1e-6)
    assert links["a"].mutual and links["b"].mutual


def test_doorway_to_nowhere_has_no_link():
    rooms = _pair() + [_room("c", 20.0, 0.0, 3.0, 3.0, doors=[(20.0, 1.5)])]
    link = next(ln for ln in adjacency(rooms) if ln.room == "c")
    assert link.other is None and link.gap is None and not link.mutual


def test_doorway_picks_the_nearest_room_and_flags_a_missing_partner_door():
    hub = _room("hub", 0.0, 0.0, 4.0, 3.0, doors=[(4.0, 1.5)])
    near = _room("near", 4.0 + WALL, 0.0, 3.0, 3.0)  # has no doorway of its own
    far = _room("far", 4.0 + WALL + 3.0 + 1.0, 0.0, 3.0, 3.0)
    (link,) = adjacency([hub, near, far])
    assert link.other == "near" and not link.mutual


def test_neighbouring_rooms_do_not_overlap_but_shifted_ones_do():
    a, b = _pair()
    assert overlap_area(a, b) == 0.0 and overlaps([a, b]) == []
    b_in = _room("b", 3.5, 0.0, 4.0, 3.0)  # 0.5 m too far left: 0.5 x 3 m shared
    assert overlap_area(a, b_in) == pytest.approx(1.5, rel=0.03)
    (found,) = overlaps([a, b_in])
    assert found[:2] == ("a", "b") and found[2] == pytest.approx(1.5, rel=0.03)


def test_footprint_counts_overlap_once():
    a, b = _pair()
    assert footprint_area([a, b]) == pytest.approx(24.0, rel=0.02)
    b_in = _room("b", 3.5, 0.0, 4.0, 3.0)
    assert footprint_area([a, b_in]) == pytest.approx(7.5 * 3.0, rel=0.02)


def test_footprint_of_an_l_shaped_room():
    poly = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]  # 12 m2
    room = RoomPlan(id="l", polygon=poly, walls=[], ceiling_height=_m(2.5), floor_area=_m(12.0, "m2"), openings=[])
    assert np.isclose(footprint_area([room]), 12.0, rtol=0.02)
