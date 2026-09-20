"""Adjacency, overlap and footprint of rooms laid out by hand (truth is known)."""

import numpy as np
import pytest

from floorfathom.schema import Measurement, Opening, RoomPlan
from floorfathom.stitch import _dist_to_outline, adjacency, footprint_area, overlap_area, overlaps, stitch, transform_room

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


# ---- placing rooms that come in their own frames ---------------------------------------

def _door_w(id: str, at: tuple[float, float], width: float) -> Opening:
    return Opening(id=id, kind="doorway", wall_id="w", start=at, end=at, centre=at, width=_m(width))


def _box(id: str, x0: float, y0: float, w: float, d: float, doors: list[tuple[str, tuple[float, float], float]]) -> RoomPlan:
    room = _room(id, x0, y0, w, d)
    return room.model_copy(update={"openings": [_door_w(f"{id}_{n}", at, wd) for n, at, wd in doors]})


def _in_own_frame(room: RoomPlan, angle: float, shift: tuple[float, float]) -> RoomPlan:
    """The room as a video or photo session would report it: rotated and shifted, frame='room'."""
    c, s = np.cos(-angle), np.sin(-angle)
    inv = np.array([[c, -s], [s, c]])

    def pt(p):
        q = inv @ (np.asarray(p, float) - shift)
        return float(q[0]), float(q[1])

    return room.model_copy(
        update={
            "frame": "room",
            "polygon": [pt(p) for p in room.polygon],
            "openings": [o.model_copy(update={"start": pt(o.start), "end": pt(o.end), "centre": pt(o.centre)}) for o in room.openings],
        }
    )


def _flat() -> list[RoomPlan]:
    """A hub with three doorways of different widths, each leading to its own room."""
    return [
        _box("hub", 0, 0, 6, 4, [("e", (6.0, 2.0), 0.9), ("n", (3.0, 4.0), 0.8), ("w", (0.0, 1.0), 1.0)]),
        _box("east", 6 + WALL, 0.5, 4, 3, [("d", (6 + WALL, 2.0), 0.9)]),
        _box("north", 1.5, 4 + WALL, 3, 3, [("d", (3.0, 4 + WALL), 0.8)]),
        _box("west", -3 - WALL, -1, 3, 4, [("d", (-WALL, 1.0), 1.0)]),
    ]


def _max_vertex_error(a: list[RoomPlan], b: list[RoomPlan]) -> float:
    by_id = {r.id: r for r in b}
    return max(
        float(np.abs(np.asarray(r.polygon) - np.asarray(by_id[r.id].polygon)).max()) for r in a
    )


def test_rooms_in_their_own_frames_are_put_back_where_they_were():
    truth = _flat()
    moves = {"east": (1.0, (12.0, -3.0)), "north": (-2.2, (-7.0, 5.0)), "west": (0.4, (3.0, 9.0))}
    rooms = [truth[0].model_copy(update={"frame": "room"})] + [
        _in_own_frame(r, *moves[r.id]) for r in truth[1:]
    ]
    out = stitch(rooms[::-1])  # order must not matter
    assert out.unplaced == []
    assert all(r.frame == "capture" for r in out.rooms)
    assert _max_vertex_error(out.rooms, truth) < 0.01
    assert overlaps(out.rooms) == []
    assert {p.room: p.host for p in out.placements if p.host} == {"east": "hub", "north": "hub", "west": "hub"}


def test_identical_doorways_are_flagged_ambiguous_and_still_do_not_overlap():
    hub = _box("hub", 0, 0, 6, 4, [("a", (6.0, 1.0), 0.9), ("b", (6.0, 3.0), 0.9), ("c", (0.0, 2.0), 0.9)])
    b1 = _box("b1", 6 + WALL, 0.1, 3, 1.8, [("d", (6 + WALL, 1.0), 0.9)])
    b2 = _box("b2", 6 + WALL, 2.1, 3, 1.8, [("d", (6 + WALL, 3.0), 0.9)])
    out = stitch([hub.model_copy(update={"frame": "room"}), _in_own_frame(b1, 0.7, (2, 2)), _in_own_frame(b2, -1.1, (5, 1))])
    assert out.unplaced == []
    assert overlaps(out.rooms) == []
    assert any("placement_ambiguous" in r.flags for r in out.rooms)


def test_a_room_whose_doorway_fits_nowhere_is_reported_not_guessed():
    hub = _box("hub", 0, 0, 6, 4, [("e", (6.0, 2.0), 0.9), ("w", (0.0, 2.0), 0.9)])
    wide = _in_own_frame(_box("wide", 6 + WALL, 0, 4, 4, [("d", (6 + WALL, 2.0), 2.4)]), 0.3, (1, 1))
    shut = _in_own_frame(_box("shut", 20, 0, 3, 3, []), 0.0, (0, 0))
    out = stitch([hub.model_copy(update={"frame": "room"}), wide, shut])
    assert sorted(out.unplaced) == ["shut", "wide"]
    assert [r.id for r in out.rooms] == ["hub"]


def test_rooms_already_in_the_capture_frame_stay_put():
    truth = _flat()
    east = _in_own_frame(truth[1], 0.5, (4.0, 4.0))
    out = stitch([truth[0], east])  # the hub is frame='capture'
    assert out.placements[0].host == "hub" and len(out.placements) == 1
    hub = next(r for r in out.rooms if r.id == "hub")
    assert hub.polygon == truth[0].polygon
    assert _max_vertex_error(out.rooms, truth[:2]) < 0.01


def test_transform_keeps_doorways_on_the_walls_and_the_area():
    east = _flat()[1]
    moved = transform_room(east.model_copy(update={"frame": "room"}), 1.3, np.array([5.0, -2.0]))
    assert moved.frame == "capture"
    assert footprint_area([moved]) == pytest.approx(footprint_area([east]), rel=0.02)
    assert _dist_to_outline(np.asarray(moved.openings[0].centre), np.asarray(moved.polygon)) < 1e-9
