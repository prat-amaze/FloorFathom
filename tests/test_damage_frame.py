"""damage_frame.reframe_damage keeps ceiling/floor damage aligned after a rigid room move."""

import numpy as np
import pytest

from floorfathom.damage_frame import reframe_damage
from floorfathom.schema import DamageRegion, Measurement, RoomPlan, SurfaceRef, Wall
from floorfathom.stitch import transform_room

ANGLE, SHIFT = 0.7, np.array([3.0, -1.0])


def _m(v, unit="m"):
    return Measurement(value=v, lo=v, hi=v, unit=unit, method="test")


def _wall(id, start, end):
    length = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    return Wall(id=id, start=start, end=end, length=_m(length), evidence="wall_points")


def _dmg(did, kind, sid, polygon, centre):
    return DamageRegion(
        id=did, surface=SurfaceRef(kind=kind, id=sid), damage_class="water_stain", class_confidence=0.8,
        polygon=polygon, centre=centre, width=_m(0.3), height=_m(0.3), length=_m(0.3), area=_m(0.09, "m2"),
        evidence="test",
    )


def _room(damage=()):
    """A 4 x 3 m rectangle at the origin, walls following the polygon edges."""
    poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    walls = [_wall(f"w{i}", poly[i], poly[(i + 1) % 4]) for i in range(4)]
    return RoomPlan(
        id="room_0", polygon=poly, walls=walls, ceiling_height=_m(2.5), floor_area=_m(12.0, "m2"), openings=[],
        damage=list(damage),
    )


def _where(room: RoomPlan, points, angle=ANGLE, shift=SHIFT):
    """Where transform_room sends ``points``, so the expected answer never repeats its rotation math."""
    probe = room.model_copy(update={"polygon": list(points)})
    return transform_room(probe, angle, shift).polygon


def test_a_ceiling_region_follows_the_room_to_its_new_frame():
    ceiling = _dmg("c", "ceiling", "room_0_ceiling", [(1.0, 1.0), (1.5, 1.0), (1.5, 1.5), (1.0, 1.5)], (1.25, 1.25))
    old = _room(damage=[ceiling])
    new = transform_room(old, ANGLE, SHIFT)

    reframe_damage(old, new)

    expected_centre, *expected_polygon = _where(old, [ceiling.centre, *ceiling.polygon])
    got = next(r for r in new.damage if r.id == "c")
    assert got.centre == pytest.approx(expected_centre, abs=1e-4)
    for got_pt, exp_pt in zip(got.polygon, expected_polygon):
        assert got_pt == pytest.approx(exp_pt, abs=1e-4)


def test_a_floor_region_behaves_like_a_ceiling_region():
    floor = _dmg("f", "floor", "room_0_floor", [(2.0, 2.0), (2.4, 2.0), (2.4, 2.4)], (2.2, 2.1))
    old = _room(damage=[floor])
    new = transform_room(old, ANGLE, SHIFT)

    reframe_damage(old, new)

    expected_centre, *expected_polygon = _where(old, [floor.centre, *floor.polygon])
    got = next(r for r in new.damage if r.id == "f")
    assert got.centre == pytest.approx(expected_centre, abs=1e-4)
    for got_pt, exp_pt in zip(got.polygon, expected_polygon):
        assert got_pt == pytest.approx(exp_pt, abs=1e-4)


def test_a_wall_region_is_left_alone():
    wall_dmg = _dmg("w", "wall", "w0", [(0.5, 0.2), (0.8, 0.2), (0.8, 0.6)], (0.65, 0.4))
    old = _room(damage=[wall_dmg])
    new = transform_room(old, ANGLE, SHIFT)

    reframe_damage(old, new)

    got = next(r for r in new.damage if r.id == "w")
    assert got.centre == wall_dmg.centre and got.polygon == wall_dmg.polygon


def test_identity_transform_changes_nothing():
    ceiling = _dmg("c", "ceiling", "room_0_ceiling", [(1.0, 1.0), (1.5, 1.5)], (1.25, 1.25))
    old = _room(damage=[ceiling])
    new = transform_room(old, 0.0, np.array([0.0, 0.0]))

    reframe_damage(old, new)

    got = next(r for r in new.damage if r.id == "c")
    assert got.centre == ceiling.centre and got.polygon == ceiling.polygon


def test_a_room_with_no_walls_does_not_crash_and_changes_nothing():
    ceiling = _dmg("c", "ceiling", "room_0_ceiling", [(1.0, 1.0)], (1.0, 1.0))
    old = _room(damage=[ceiling]).model_copy(update={"walls": []})
    new = old.model_copy()

    reframe_damage(old, new)

    got = next(r for r in new.damage if r.id == "c")
    assert got.centre == ceiling.centre and got.polygon == ceiling.polygon
