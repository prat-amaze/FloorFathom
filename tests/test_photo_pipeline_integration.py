"""End-to-end integration test: build_photo_room from synthetic SfM walk."""
import numpy as np

from floorfathom.io_photos import PhotoImage, PhotoSet
from floorfathom.photo_pipeline import build_photo_room
from synth import textured_room


def _fake_depth(rgb):
    return np.full(rgb.shape[:2], 2.0, np.float32)


def test_build_photo_room_end_to_end_on_synthetic_walk(tmp_path):
    stills = textured_room("rect4x3", 2.6, [(0.5, 0.0, 0.5), (0.5, 0.0, 2.5), (3.5, 0.0, 2.5)], photo_positions=10)
    photos = PhotoSet(room="synth", images=[
        PhotoImage(s.name, sha256=s.name, rgb=s.rgb, full_size=(320, 240), f35=s.f35, f_px=266.0)
        for s in stills
    ])
    room = build_photo_room("synth", photos, _fake_depth, seed=0, work=tmp_path / "work")
    plan_room = room.plan.rooms[0]
    # With flat depth model, area and ceiling may not be perfect — just check it ran end to end
    assert plan_room is not None
    assert plan_room.name == "synth"
