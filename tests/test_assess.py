"""assess_room end to end on the synthetic painted room of test_surfaces: regions in surface coordinates, flags, scope."""

import numpy as np
import pytest

from floorfathom.assess import assess_room
from floorfathom.schema import RoomPlan
from test_surfaces import FLOOR_Y, HEIGHT, Quad, _frames, _plan, _room_quads, _stained


@pytest.fixture(scope="module")
def wall_stain_frames():
    return _frames(_room_quads())[::3]


def test_wall_stain_becomes_a_region_a_scope_line_and_valid_json(wall_stain_frames):
    room = _plan()
    notes = assess_room(room, FLOOR_Y, FLOOR_Y + HEIGHT, wall_stain_frames, mpp=0.02)
    assert notes == []
    assert len(room.damage) == 1
    d = room.damage[0]
    assert d.surface.kind == "wall" and d.surface.id == "r0_w0" and d.damage_class == "water_stain"
    assert abs(d.centre[0] - 1.8) < 0.06 and abs(d.centre[1] - 1.2) < 0.06  # (s, h) on the wall
    assert abs(d.width.value - 0.30) < 0.08 and abs(d.height.value - 0.40) < 0.08
    assert d.width.lo <= d.width.value <= d.width.hi and d.area.unit == "m2"
    item = [s for s in room.scope if s.damage_ids == [d.id]][0]
    assert item.surface == d.surface and item.quantity.unit == "m2" and abs(item.quantity.value - d.area.value) < 1e-9
    assert RoomPlan.model_validate(room.model_dump()) == room


def test_ceiling_stain_is_in_plan_coordinates_and_flagged():
    q = _room_quads(stain=False)
    q["ceiling"] = Quad([0, HEIGHT, 0], [1, 0, 0], [0, 0, -1], 4, 3, _stained(3.0, 2.0, 0.2, 0.2))  # plan x 3.0, plan y 2.0
    room = _plan()
    assess_room(room, FLOOR_Y, FLOOR_Y + HEIGHT, _frames(q)[::3], mpp=0.02)
    assert [d.surface.id for d in room.damage] == ["room_0_ceiling"]
    d = room.damage[0]
    assert d.damage_class == "water_stain" and abs(d.centre[0] - 3.0) < 0.08 and abs(d.centre[1] - 2.0) < 0.08
    assert "ceiling_stain_source_above" in [f.rule for f in room.concealed_flags]
    assert any(s.action.startswith("Open up") for s in room.scope)


def test_a_function_of_the_plane_and_a_kinds_filter(wall_stain_frames):
    room, seen = _plan(), []

    def frames(plane):
        seen.append(plane)
        return wall_stain_frames

    assess_room(room, FLOOR_Y, FLOOR_Y + HEIGHT, frames, kinds=("wall",), mpp=0.02)
    assert len(seen) == 4 and [d.surface.kind for d in room.damage] == ["wall"]


def test_no_frames_means_no_damage_and_a_note_per_surface_and_none_when_no_ceiling():
    room = _plan()
    notes = assess_room(room, FLOOR_Y, None, [], mpp=0.05)
    assert room.damage == [] and room.concealed_flags == [] and room.scope == []
    assert sum("only 0%" in n for n in notes) == 5 and any("no ceiling" in n for n in notes)  # 4 walls and the floor
