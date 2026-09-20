"""Fusing one room's photos: gravity and floor recovered, the shared estimator then reads a room from the cloud."""

import numpy as np
from synth import rect
from synth_photo import CAMERA_HEIGHT, room_photos

from floorfathom.estimate import Params, estimate
from floorfathom.photo_scene import build_scene

ROOM = rect(0, 0, 5, 4)
HEIGHT = 2.6
STATION = (2.2, CAMERA_HEIGHT, 1.7)
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]


def _scene(**kw):
    pitches = kw.pop("pitches", [8.0] + [0.0] * 7)
    rolls = kw.pop("rolls", [5.0] + [0.0] * 7)
    photos, poses, depth, w2r = room_photos(ROOM, HEIGHT, STATION, YAWS, pitches, rolls, **kw)
    return build_scene(photos, poses, depth), photos, poses, depth, w2r


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def test_gravity_and_floor_height_are_recovered_from_a_tilted_root_camera():
    scene, _, _, _, w2r = _scene(noise=0.005)
    assert scene is not None and scene.flags == []
    up_in_cloud = scene.rotation @ (w2r @ np.array([0.0, 1.0, 0.0]))
    assert _angle_deg(up_in_cloud, [0, 1, 0]) < 1.0
    assert abs(scene.floor_y + CAMERA_HEIGHT) < 0.03
    assert 0.8 < scene.tilt_deg < 1.7  # only camera 0 is tilted (about 9.4 degrees), so the mean up is off by about 9.4 / 8


def test_the_shared_estimator_reads_the_room_from_the_fused_cloud():
    scene, *_ = _scene(noise=0.005)
    est = estimate(scene.cloud.points, scene.traj_xz, Params())  # default Params: the fan supplies the "visits"
    assert len(est.rooms) == 1
    room = est.rooms[0]
    assert abs(room.area - 20.0) < 1.0
    assert room.ceiling_height is not None and abs(room.ceiling_height - HEIGHT) < 0.05
    lengths = sorted(e.length for e in room.outline.edges)
    assert len(lengths) == 4 and np.allclose(lengths, [4, 4, 5, 5], atol=0.1)


def test_a_scale_bias_in_depth_scales_the_room_by_the_same_factor():
    scene, *_ = _scene(noise=0.005, bias=1.06)
    est = estimate(scene.cloud.points, scene.traj_xz, Params())
    assert len(est.rooms) == 1
    assert abs(est.rooms[0].area / 20.0 - 1.06**2) < 0.05  # documents what a 6% metric error does to area


def test_images_without_a_pose_are_left_out_and_chunks_follow_the_used_images():
    photos, poses, depth, _ = room_photos(ROOM, HEIGHT, STATION, YAWS, noise=0.005)
    poses.rotations[3] = None
    scene = build_scene(photos, poses, depth)
    assert scene.used == [f"{i}.jpg" for i in (0, 1, 2, 4, 5, 6, 7)]
    assert scene.cloud.n_chunks == 7 and set(np.unique(scene.cloud.chunk)) == set(range(7))
    assert np.all(np.diff(scene.cloud.chunk) >= 0)  # the jackknife slices the cloud by chunk


def test_a_view_of_only_a_wall_has_no_floor_and_gives_no_scene():
    photos, poses, _, _ = room_photos(ROOM, HEIGHT, STATION, [0, 30])
    wall = lambda rgb: np.full(rgb.shape[:2], 3.0, np.float32)  # a fronto-parallel plane: no horizontal surface
    assert build_scene(photos, poses, wall) is None


def test_same_seed_gives_an_identical_cloud():
    a, *_ = _scene(noise=0.01)
    b, *_ = _scene(noise=0.01)
    assert np.array_equal(a.cloud.points, b.cloud.points) and np.array_equal(a.traj_xz, b.traj_xz)
