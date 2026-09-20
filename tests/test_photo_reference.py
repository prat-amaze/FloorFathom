"""Scale from the yellow reference ruler seen in one photo, in a synthetic room whose depth model reads long or short."""

import numpy as np
import pytest
from synth import rect
from synth_photo import CAMERA_HEIGHT, paint_ruler, room_photos

from floorfathom.photo_layout import wall_segments
import cv2

from floorfathom.photo_reference import find_ruler, ruler_scale
from floorfathom.photo_scene import build_scene

HEIGHT = 2.6
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]


def _room(bias=1.0, station_x=2.2, ruler=True):
    station = (station_x, CAMERA_HEIGHT, 1.0)  # 1 m from the south wall, where the ruler is
    n = len(YAWS)
    photos, poses, depth, _ = room_photos(rect(0, 0, 5, 4), HEIGHT, station, YAWS, [8.0] + [0.0] * (n - 1),
                                          [5.0] + [0.0] * (n - 1), noise=0.005, bias=bias)
    seen = [paint_ruler(im.rgb, station, c2w, (2.2, CAMERA_HEIGHT, 0.0)) for im, c2w in zip(photos.images, depth.c2ws)] if ruler else []
    scene = build_scene(photos, poses, depth)
    return photos, poses, scene, wall_segments(scene.cloud.points), seen


@pytest.mark.parametrize("bias", [1.0, 1.3, 0.8])
def test_the_scale_is_recovered_whatever_the_depth_model_reads(bias):
    photos, poses, scene, segs, seen = _room(bias)
    assert sum(seen) >= 1
    got, flags = ruler_scale(photos, poses, scene, segs)
    assert got is not None and flags == []
    assert got.factor == pytest.approx(1 / bias, rel=0.04)
    assert 0.02 <= got.rel_sigma < 0.08 and got.n_photos == sum(seen)


def test_a_ruler_seen_off_head_on_is_corrected_by_the_wall_plane():
    # the station is 0.4 m to the side of the ruler: it is about 22 degrees off the wall's normal and reads 7% short
    photos, poses, scene, segs, seen = _room(bias=1.2, station_x=1.8)
    assert sum(seen) >= 1
    got, _ = ruler_scale(photos, poses, scene, segs)
    assert got.factor == pytest.approx(1 / 1.2, rel=0.04)


def test_no_ruler_gives_no_scale():
    photos, poses, scene, segs, _ = _room(ruler=False)
    got, flags = ruler_scale(photos, poses, scene, segs)
    assert got is None and flags == []


def test_the_ruler_length_is_the_video_tiers():
    from floorfathom.anchor import REFERENCE_LENGTH_M, STRIP_COLOUR

    assert (REFERENCE_LENGTH_M, STRIP_COLOUR) == (0.316, "yellow")  # the protocol's yellow body, end to end


def _blob_image(colour, size=(60, 8), triangle=False):
    img = np.zeros((320, 240, 3), np.uint8)
    x, y = 100, 100
    if triangle:  # half of its bounding box, as trim or a pipe with fittings is
        cv2.fillConvexPoly(img, np.array([[x, y], [x + size[1], y], [x, y + size[0]]], np.int32), colour)
    else:
        img[y : y + size[0], x : x + size[1]] = colour
    return img[..., ::-1].copy()  # find_ruler takes BGR


def test_the_ruler_is_a_solid_uniform_yellow_bar_of_the_right_shape():
    assert find_ruler(_blob_image((255, 255, 0)), 40) is not None  # hue 30: the ruler
    assert find_ruler(_blob_image((255, 200, 0)), 40) is not None  # hue 24: the ruler as the phone renders it
    assert find_ruler(_blob_image((255, 160, 0)), 40) is None  # hue 19: bright wood or trim, orange-side edge
    assert find_ruler(_blob_image((255, 255, 0), size=(120, 3)), 40) is None  # a thin cable or pipe: too long for its width
    assert find_ruler(_blob_image((255, 255, 0), size=(80, 10), triangle=True), 40) is None  # half of its box, not a solid bar


def test_a_ruler_at_the_wrong_height_is_not_used():
    station = (2.2, CAMERA_HEIGHT, 1.8)
    photos, poses, depth, _ = room_photos(rect(0, 0, 5, 4), HEIGHT, station, YAWS, [8.0] + [0.0] * 7, [5.0] + [0.0] * 7, noise=0.005)
    seen = [paint_ruler(im.rgb, station, c2w, (2.2, CAMERA_HEIGHT - 0.7, 0.0)) for im, c2w in zip(photos.images, depth.c2ws)]
    assert any(seen)  # 0.7 m below the phone: in the picture, but not where the protocol puts it
    scene = build_scene(photos, poses, depth)
    got, flags = ruler_scale(photos, poses, scene, wall_segments(scene.cloud.points))
    assert got is None and any(f.startswith("ruler_height_implausible") for f in flags)
