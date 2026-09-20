"""One room from photos: walls, heights, intervals, the null policy for thin input, and the JSON contract."""

import numpy as np
import pytest
from synth import rect
from synth_photo import CAMERA_HEIGHT, room_photos

from floorfathom.photo_pipeline import MONO_SCALE_REL_SIGMA, plan_photo_room
from floorfathom.schema import CapturePlan

HEIGHT = 2.6
STATION = (2.2, CAMERA_HEIGHT, 1.7)
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]
TRUE_WALLS = [4, 4, 5, 5]


def _plan(yaws=YAWS, bias=1.0, **kw):
    n = len(yaws)
    photos, poses, depth, _ = room_photos(rect(0, 0, 5, 4), HEIGHT, STATION, yaws, [8.0] + [0.0] * (n - 1),
                                          [5.0] + [0.0] * (n - 1), noise=0.005, bias=bias)
    return plan_photo_room("room", photos, poses, depth, **kw)


@pytest.fixture(scope="module")
def anchored():  # the scale is known (a reference): the intervals are then the photo tier's own
    return _plan(scale=1.0, scale_rel_sigma=0.02)


@pytest.fixture(scope="module")
def mono_biased():  # nothing anchors the scale and the depth model reads 30% long
    return _plan(bias=1.3)


def _covers(m, truth):
    return m.value is not None and m.lo <= truth <= m.hi


def test_a_full_room_gives_its_walls_area_and_ceiling_with_intervals_that_cover_the_truth(anchored):
    assert anchored.tier == "photo" and len(anchored.rooms) == 1
    room = anchored.rooms[0]
    assert room.name == "room" and room.frame == "room" and room.stations[0].height_above_floor == pytest.approx(CAMERA_HEIGHT, abs=0.1)
    assert sorted(round(w.length.value) for w in room.walls) == TRUE_WALLS
    for w in room.walls:
        assert _covers(w.length, min(TRUE_WALLS, key=lambda t: abs(t - w.length.value))) and w.evidence == "wall_points"
    assert _covers(room.floor_area, 20.0) and _covers(room.ceiling_height, HEIGHT)
    assert room.flags == [] or room.flags == ["scale_from_depth_model_only"]


def test_photo_intervals_are_wider_than_a_few_percent_even_when_the_scale_is_known(anchored):
    room = anchored.rooms[0]
    assert all((w.length.hi - w.length.lo) / w.length.value > 0.12 for w in room.walls)  # assumed systematic 10% at 95%


def test_the_scale_source_is_recorded(anchored):
    d = anchored.diagnostics
    assert (d.scale_method, d.scale_factor, d.scale_rel_sigma) == ("reference_object", 1.0, 0.02)
    assert d.models == [] and d.frames_used is None


def test_without_a_reference_the_scale_is_the_models_and_a_biased_model_is_still_covered(mono_biased):
    d = mono_biased.diagnostics
    assert d.scale_method == "monocular_depth" and d.scale_rel_sigma == MONO_SCALE_REL_SIGMA
    room = mono_biased.rooms[0]
    assert "scale_from_depth_model_only" in room.flags
    assert all(_covers(w.length, min(TRUE_WALLS, key=lambda t: abs(t - w.length.value))) for w in room.walls)
    assert _covers(room.floor_area, 20.0) and _covers(room.ceiling_height, HEIGHT)


def test_one_photo_is_too_thin_and_gives_an_explicit_null_room():
    photos, poses, depth, _ = room_photos(rect(0, 0, 5, 4), HEIGHT, STATION, [0])
    poses.rotations[0] = None
    plan = plan_photo_room("hall", photos, poses, depth)
    room = plan.rooms[0]
    assert room.polygon == [] and room.walls == [] and room.floor_area.value is None and room.ceiling_height.value is None
    assert "insufficient_views" in room.flags and room.name == "hall"


def test_half_a_room_leaves_walls_without_lengths_and_no_area():
    plan = _plan(yaws=[0, 45, 90, 135], scale=1.0, scale_rel_sigma=0.02)
    room = plan.rooms[0]
    chords = [w for w in room.walls if w.evidence == "closure"]
    assert chords and all(w.length.value is None and w.length.note for w in chords)
    assert room.floor_area.value is None and "room_outline_incomplete" in room.flags
    assert any(w.length.value is not None for w in room.walls)  # what was seen is still measured


def test_the_plan_round_trips_through_the_schema_and_a_rerun_is_identical(anchored):
    again = CapturePlan.model_validate_json(anchored.model_dump_json())
    assert again == anchored
    rerun = _plan(scale=1.0, scale_rel_sigma=0.02)
    a, b = anchored.model_dump(), rerun.model_dump()
    a["diagnostics"].pop("seconds"), b["diagnostics"].pop("seconds")
    assert a == b


def test_an_opening_width_is_never_negative_and_intervals_are_ordered(anchored):
    room = anchored.rooms[0]
    for m in [room.floor_area, room.ceiling_height, *[w.length for w in room.walls]]:
        assert m.lo <= m.value <= m.hi and m.lo >= 0
    assert np.isfinite(room.floor_area.value)
