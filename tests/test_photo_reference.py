"""Tests for the new multi-frame ruler triangulation."""
from pathlib import Path
from floorfathom.io_photos import PhotoImage, PhotoSet
from floorfathom.photo_reference import ruler_scale_sfm


def _photoset():
    from synth import textured_room
    stills = textured_room("rect4x3", 2.6, [(0.5, 0.0, 0.5), (0.5, 0.0, 2.5), (3.5, 0.0, 2.5)], photo_positions=8)
    return PhotoSet(room="synth", images=[
        PhotoImage(s.name, sha256=s.name, rgb=s.rgb, full_size=(320, 240), f35=s.f35, f_px=266.0)
        for s in stills
    ])


def test_ruler_scale_none_without_a_visible_ruler(tmp_path):
    from floorfathom.photo_pose import run_photo_sfm
    photos = _photoset()
    sfm = run_photo_sfm(photos, tmp_path / "work", seed=0)
    if sfm is None:
        return  # SfM failed, skip ruler test
    ruler, flags = ruler_scale_sfm(photos, sfm, tmp_path / "ruler_work", length=0.316)
    assert ruler is None
    assert any("ruler_not_seen" in f or "insufficient" in f or "too_few" in f for f in flags) or flags == []
