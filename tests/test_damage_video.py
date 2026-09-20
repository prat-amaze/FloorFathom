"""The video damage adapter against a synthetic clip whose damage was painted at known sizes and places."""

from __future__ import annotations

import numpy as np
import pytest

from floorfathom import damage_video as D
from floorfathom import surfaces as S
from floorfathom.assess import assess_room
from synth_damage_video import FLOOR_Y, GRAVITY, SCALE, WALL_H, cameras, make_clip, room, texture

STAIN = (2.0, 1.2, 0.21, 0.25)  # centre s, centre h, width, height (m)
CRACK = (3.1, 1.0, 0.22, 0.003)  # start s, start h, length, width (m)
BOARD_IN = set(range(0, 14, 2))  # frames with a round object standing in front of the wall


def wall_plane():
    return S.surface_planes(room(), FLOOR_Y, FLOOR_Y + WALL_H)[0][1]


def views_of(clip, depth: bool = True, align: bool = True) -> D.Views:
    kf, sfm, depths = clip
    fn = (lambda rgb: depths[rgb.tobytes()]) if depth else None
    return D.Views(kf, sfm, D.world_poses(sfm, GRAVITY, SCALE), SCALE, depth=fn, align_depth=align)


def judge(clip, depth: bool = True, scale_rel: float = 0.02, align: bool = True):
    """The wall judged the way the pipeline does it, except for the surface kinds (only the painted wall exists)."""
    r = room()
    notes = assess_room(
        r, FLOOR_Y, FLOOR_Y + WALL_H, views_of(clip, depth, align).frames_for, kinds=("wall",), use_relief=False, scale_rel_sigma=scale_rel
    )
    return r, notes


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Stain and crack on the wall, a round object in front of it in every other frame."""
    return make_clip(tmp_path_factory.mktemp("clip"), texture(stain=STAIN, crack=CRACK), occluder_in=BOARD_IN)


def test_poses_come_back_from_sfm_units_to_metres(clip):
    _kf, sfm, _d = clip
    poses = D.world_poses(sfm, GRAVITY, SCALE)
    for i, truth in enumerate(cameras()):
        assert np.allclose(poses[i], truth, atol=1e-9)


def test_unregistered_frames_have_no_pose(clip):
    _kf, sfm, _d = clip
    reg = sfm.registered.copy()
    reg[3] = False
    fake = type(sfm)(**{**sfm.__dict__, "registered": reg})
    assert np.isnan(D.world_poses(fake, GRAVITY, SCALE)[3]).all()


def test_choice_is_few_sorted_frames_that_see_the_wall(clip):
    chosen = views_of(clip).choose(wall_plane())
    assert 2 <= len(chosen) <= D.MAX_VIEWS and chosen == sorted(chosen)


def test_a_surface_behind_the_cameras_gets_no_frames(clip):
    behind = S.Plane(
        origin=np.array([0.0, FLOOR_Y, -8.0]), u=np.array([1.0, 0, 0]), v=np.array([0, 1.0, 0]), normal=np.array([0, 0, 1.0]), size=(3.0, 2.0)
    )
    assert views_of(clip).choose(behind) == []


def test_stain_and_crack_are_found_at_their_painted_size_and_place(clip):
    r, _notes = judge(clip)
    on_wall = [d for d in r.damage if d.surface.id == "r0_w0"]
    assert len(on_wall) == 2
    stain = next(d for d in on_wall if d.damage_class == "water_stain")
    crack = next(d for d in on_wall if d.damage_class == "structural_crack")
    assert stain.width.value == pytest.approx(STAIN[2], abs=0.02) and stain.height.value == pytest.approx(STAIN[3], abs=0.02)
    assert stain.width.lo <= STAIN[2] <= stain.width.hi and stain.height.lo <= STAIN[3] <= stain.height.hi
    assert crack.length.value == pytest.approx(CRACK[2], abs=0.03)
    assert stain.centre[0] == pytest.approx(STAIN[0], abs=0.03) and stain.centre[1] == pytest.approx(STAIN[1], abs=0.03)
    assert {tuple(i.damage_ids) for i in r.scope} >= {(stain.id,), (crack.id,)}


def test_a_larger_scale_error_widens_the_intervals(clip):
    tight, _ = judge(clip, scale_rel=0.02)
    loose, _ = judge(clip, scale_rel=0.15)
    a = next(d for d in tight.damage if d.damage_class == "water_stain")
    b = next(d for d in loose.damage if d.damage_class == "water_stain")
    assert b.width.value == a.width.value and b.width.hi - b.width.lo > 2 * (a.width.hi - a.width.lo)


def test_depth_gating_leaves_only_the_painted_damage_with_an_object_in_front(clip):
    """With depth the round object in front of the wall is left out of the patch. (Without depth the detector's own
    shape and area filters also drop it on this synthetic wall, so the two are no longer compared here.)"""
    r, _ = judge(clip)
    assert sorted(d.damage_class for d in r.damage) == ["structural_crack", "water_stain"]


def test_a_clean_wall_with_an_object_in_front_has_no_damage(tmp_path):
    c = make_clip(tmp_path, texture(stain=None, crack=None), occluder_in=BOARD_IN)
    r, _ = judge(c)
    assert r.damage == [] and r.concealed_flags == [] and r.scope == []


def test_the_pipeline_entry_searches_walls_and_ceiling_and_never_reports_relief(clip):
    kf, sfm, depths = clip
    r = room()
    notes = D.assess_video(r, FLOOR_Y, kf, sfm, GRAVITY, SCALE, 0.02, depth=lambda rgb: depths[rgb.tobytes()])
    assert {d.surface.kind for d in r.damage} <= {"wall", "ceiling"}
    assert not any(d.damage_class == "sagging_or_bulging" for d in r.damage)
    assert isinstance(notes, list)
    assert any(d.damage_class == "water_stain" and d.surface.id == "r0_w0" for d in r.damage)


def test_a_room_without_a_ceiling_height_leaves_the_ceiling_out(clip):
    kf, sfm, depths = clip
    r = room(ceiling=None)
    notes = D.assess_video(r, FLOOR_Y, kf, sfm, GRAVITY, SCALE, 0.02, depth=lambda rgb: depths[rgb.tobytes()])
    assert all(d.surface.kind == "wall" for d in r.damage) and any("ceiling" in n for n in notes)


def test_renumbering_a_room_moves_every_id_and_surface_that_names_it(clip):
    kf, sfm, depths = clip
    r = room()
    D.assess_video(r, FLOOR_Y, kf, sfm, GRAVITY, SCALE, 0.02, depth=lambda rgb: depths[rgb.tobytes()])
    r.id = "room_0"  # the pipeline names a clip's room room_0; the synthetic one is r0
    assert r.damage
    D.renumber_damage(r, 3)
    ids = [i.id for i in [*r.damage, *r.concealed_flags, *r.scope]]
    assert ids and all(i.startswith("r3_") for i in ids)
    assert all(d.surface.id.startswith("r3_") for d in r.damage if d.surface.kind == "wall")
    assert {i for s in r.scope for i in s.damage_ids} <= {d.id for d in r.damage}


def test_the_pipeline_entry_writes_a_picture_per_judged_surface(clip, tmp_path):
    kf, sfm, depths = clip
    D.assess_video(room(), FLOOR_Y, kf, sfm, GRAVITY, SCALE, 0.02, depth=lambda rgb: depths[rgb.tobytes()], debug_dir=tmp_path / "damage")
    assert list((tmp_path / "damage").glob("*.png"))
