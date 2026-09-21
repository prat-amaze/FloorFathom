"""One room from photos: walls, heights, intervals, the null policy for thin input, and the JSON contract.

NOTE: These tests use the old rotation-only interface (build_photo_room with poses argument) which is
superseded by the new SfM-based multi-view pipeline. Replaced by test_photo_pipeline_integration.py.
"""

import numpy as np
import pytest
from synth import rect

pytestmark = pytest.mark.skip(reason="rotation-only interface superseded by SfM pipeline (see test_photo_pipeline_integration.py)")
from synth_photo import CAMERA_HEIGHT, paint_ruler, room_photos

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


@pytest.fixture(scope="module")
def with_ruler():  # the depth model reads 30% long, and the yellow ruler is on the south wall, as the protocol asks
    station = (2.2, CAMERA_HEIGHT, 1.0)
    photos, poses, depth, _ = room_photos(rect(0, 0, 5, 4), HEIGHT, station, YAWS, [8.0] + [0.0] * 7, [5.0] + [0.0] * 7,
                                          noise=0.005, bias=1.3)
    for im, c2w in zip(photos.images, depth.c2ws):
        paint_ruler(im.rgb, station, c2w, (2.2, CAMERA_HEIGHT, 0.0))
    return plan_photo_room("room", photos, poses, depth)


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


def test_the_ruler_in_the_photos_fixes_a_biased_depth_model_and_narrows_the_intervals(with_ruler, mono_biased):
    room, d = with_ruler.rooms[0], with_ruler.diagnostics
    assert d.scale_method == "reference_object" and d.scale_factor == pytest.approx(1 / 1.3, rel=0.05) and d.scale_rel_sigma < 0.08
    assert "scale_from_depth_model_only" not in room.flags and "reference_ruler_not_found" not in room.flags
    lengths = sorted(w.length.value for w in room.walls)
    assert np.allclose(lengths, TRUE_WALLS, rtol=0.08)  # inside the +-8% gate, from a model that read 30% long
    assert _covers(room.floor_area, 20.0) and _covers(room.ceiling_height, HEIGHT)
    width = lambda p: np.mean([(w.length.hi - w.length.lo) / w.length.value for w in p.rooms[0].walls])  # noqa: E731
    assert width(with_ruler) < 0.6 * width(mono_biased)


def test_without_a_ruler_the_room_says_so(mono_biased):
    flags = mono_biased.rooms[0].flags
    assert "reference_ruler_not_found" in flags and "scale_from_depth_model_only" in flags


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


def test_run_photo_gives_every_room_folder_a_plan_with_unique_ids_and_keeps_a_room_it_could_not_build(tmp_path, monkeypatch):
    from floorfathom import photo_pipeline as PP

    rooms = {"a": (rect(0, 0, 5, 4), (2.2, CAMERA_HEIGHT, 1.7), [0, 60, 120, 180, 240, 300]),
             "b": (rect(0, 0, 3.5, 3), (1.5, CAMERA_HEIGHT, 1.2), [0, 60, 120, 180, 240, 300]),
             "thin": (rect(0, 0, 5, 4), STATION, [0])}
    poses_of, fakes = {}, {}
    for r, name in enumerate(rooms):
        (tmp_path / "images" / name).mkdir(parents=True)
        (tmp_path / "images" / name / "1.png").write_bytes(b"")  # only discovered; loading is stubbed

    def fake_load(name, paths, long_side=1008):  # the frames reload the same photos at a larger size
        walls, pos, yaws = rooms[name]
        photos, poses, depth, _ = room_photos(walls, HEIGHT, pos, yaws, [8.0] + [0.0] * (len(yaws) - 1), noise=0.005)
        for im in photos.images:
            im.rgb[0, 0, 1] = list(rooms).index(name)  # which room's fake depth model to ask
        if name == "thin":
            poses.rotations[0] = None
        poses_of[id(photos.images)], fakes[list(rooms).index(name)] = poses, depth
        return photos

    monkeypatch.setattr(PP, "load_photo_set", fake_load)
    monkeypatch.setattr(PP, "register_rotations", lambda images, seed=0: poses_of[id(images)])
    from floorfathom import photo_damage

    debug_dirs = []
    monkeypatch.setattr(photo_damage, "assess", lambda room, result, **kw: debug_dirs.append(kw.get("debug_dir")))
    plan = PP.run_photo(tmp_path, tmp_path / "out", depth=lambda rgb: fakes[int(rgb[0, 0, 1])](rgb),
                        scale=1.0, scale_rel_sigma=0.02)
    assert debug_dirs and set(debug_dirs) == {tmp_path / "out" / "debug" / "damage"}  # damage pictures by default
    debug_dirs.clear()
    PP.run_photo(tmp_path, tmp_path / "out2", depth=lambda rgb: fakes[int(rgb[0, 0, 1])](rgb),
                 scale=1.0, scale_rel_sigma=0.02, debug=False)
    assert debug_dirs and set(debug_dirs) == {None}
    assert plan.tier == "photo" and sorted(r.name for r in plan.rooms) == ["a", "b", "thin"]
    ids = [r.id for r in plan.rooms] + [w.id for r in plan.rooms for w in r.walls]
    assert len(ids) == len(set(ids))
    assert next(r for r in plan.rooms if r.name == "thin").polygon == []
    assert sorted(p.name for p in (tmp_path / "out" / "rooms").iterdir()) == ["a.json", "b.json", "thin.json"]
    assert (tmp_path / "out" / "plan.json").exists() and (tmp_path / "out" / "plan.png").exists()
