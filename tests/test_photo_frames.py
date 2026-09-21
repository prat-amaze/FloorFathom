"""The photo room handed to the damage stage: frames in the room's own frame, checked with the framework's Accumulator.

NOTE: Tests the old rotation-only interface. Superseded by the SfM-based pipeline.
"""

import pytest
pytestmark = pytest.mark.skip(reason="rotation-only photo_frames interface superseded by SfM pipeline")

import cv2
import numpy as np
import pytest
from synth import rect
from synth_photo import CAMERA_HEIGHT, paint_ruler, room_photos

from floorfathom.io_photos import PhotoImage, PhotoSet
from floorfathom.photo_pipeline import build_photo_room

surfaces = pytest.importorskip("floorfathom.surfaces")

HEIGHT = 2.6
STATION = (2.2, CAMERA_HEIGHT, 1.0)
YAWS = list(range(0, 360, 30))  # two views of every direction, so each wall pixel has two accepted views


@pytest.fixture(scope="module")
def room():
    n = len(YAWS)
    photos, poses, depth, _ = room_photos(rect(0, 0, 5, 4), HEIGHT, STATION, YAWS, [8.0] + [0.0] * (n - 1),
                                          [5.0] + [0.0] * (n - 1), noise=0.005)
    painted = [paint_ruler(im.rgb, STATION, c2w, (2.2, CAMERA_HEIGHT, 0.0)) for im, c2w in zip(photos.images, depth.c2ws)]
    big = PhotoSet("room", [PhotoImage(im.name, im.sha256, cv2.resize(im.rgb, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                                       (im.full_size[0] * 2, im.full_size[1] * 2), im.f35, im.f_px * 2) for im in photos.images])
    return build_photo_room("room", photos, poses, depth, scale=1.0, scale_rel_sigma=0.02, hires=lambda: big), painted


def test_the_room_carries_its_frames_heights_and_scale_uncertainty(room):
    pr, _ = room
    assert len(pr.frames) == len(YAWS) and pr.scale_rel_sigma == pytest.approx(0.02)
    assert pr.floor_y == pytest.approx(-CAMERA_HEIGHT, abs=0.06) and pr.ceiling_y == pytest.approx(HEIGHT - CAMERA_HEIGHT, abs=0.06)
    fr = pr.frames[3]
    assert fr.pose.shape == (4, 4) and np.allclose(fr.pose[:3, 3], 0) and np.isclose(np.linalg.det(fr.pose[:3, :3]), 1)


def test_frames_use_the_larger_picture_with_its_own_intrinsics_and_keep_depth_at_the_working_size(room):
    fr = room[0].frames[0]
    assert fr.rgb.shape[:2] == (640, 480) and fr.depth.shape == (320, 240) and fr.depth_ok.shape == (320, 240)
    assert fr.K[0, 2] == 240 and fr.K[1, 2] == 320  # principal point at the centre of the larger picture


def test_the_framework_projects_every_wall_ceiling_and_floor_onto_valid_patches(room):
    pr, _ = room
    r = pr.plan.rooms[0]
    planes = surfaces.surface_planes(r, pr.floor_y, pr.ceiling_y)
    assert sorted(ref.kind for ref, _ in planes) == ["ceiling", "floor", "wall", "wall", "wall", "wall"]
    for ref, plane in planes:
        acc = surfaces.Accumulator(plane)
        for fr in pr.frames:
            acc.add(fr)
        # one spot sees a ceiling or floor only beyond about 1.5 m (the camera's vertical view), so those cover less than walls
        need = 0.3 if ref.kind == "wall" else 0.15
        assert acc.patch().valid.mean() > need, (ref.id, float(acc.patch().valid.mean()))


def test_the_ruler_is_never_a_valid_surface_pixel(room):
    pr, painted = room
    assert any(painted)
    cam = next(i for i, ok in enumerate(painted) if ok)
    fr = pr.frames[cam]
    yellow = (fr.rgb[..., 0] > 200) & (fr.rgb[..., 1] > 200) & (fr.rgb[..., 2] < 60)  # the painted ruler, in the larger picture
    ys, xs = np.nonzero(yellow)
    assert len(ys) > 0
    dy, dx = np.clip(ys // 2, 0, 319), np.clip(xs // 2, 0, 239)  # its place on the depth grid
    assert not fr.depth_ok[dy, dx].any()
    assert fr.depth_ok.mean() > 0.5  # and the rest of the picture is still usable
