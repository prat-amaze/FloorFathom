"""Wall segments from the levelled photo cloud: the four walls of a synthetic room, and nothing from clutter."""

import numpy as np
from synth import rect
from synth_photo import CAMERA_HEIGHT, room_photos

from floorfathom.photo_layout import wall_segments
from floorfathom.photo_scene import build_scene

HEIGHT = 2.6
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]


def _cloud(noise=0.005, bias=1.0):
    photos, poses, depth, _ = room_photos(rect(0, 0, 5, 4), HEIGHT, (2.2, CAMERA_HEIGHT, 1.7), YAWS,
                                          [8.0] + [0.0] * 7, [5.0] + [0.0] * 7, noise=noise, bias=bias)
    return build_scene(photos, poses, depth).cloud.points


def test_the_four_walls_of_a_rectangular_room_are_found_at_their_lengths_and_at_right_angles():
    segs = wall_segments(_cloud())
    assert len(segs) == 4
    # observed extents stop short of the corners (the depth-edge filter drops pixels there); the outline
    # gets the corners from the line intersections, so only require most of each wall to be seen
    assert np.allclose(sorted(s.length for s in segs), [4, 4, 5, 5], atol=0.35)
    for s in segs:  # every wall is parallel or perpendicular to the first one
        c = abs(float(s.normal @ segs[0].normal))
        assert c < 0.03 or c > 0.999
    assert all(s.support > 500 and s.offset > 0 for s in segs)


def test_the_distances_to_the_walls_are_those_of_the_room():
    segs = wall_segments(_cloud())
    # the station is at (2.2, 1.7) in a 5 x 4 room: its distances to the walls are 2.2, 2.8, 1.7 and 2.3
    assert np.allclose(sorted(s.offset for s in segs), [1.7, 2.2, 2.3, 2.8], atol=0.06)


def test_a_scale_bias_scales_the_walls_by_the_same_factor():
    segs = wall_segments(_cloud(bias=1.1))
    assert np.allclose(sorted(s.length for s in segs), [4.4, 4.4, 5.5, 5.5], atol=0.4)


def test_scattered_points_without_vertical_structure_give_no_walls():
    rng = np.random.default_rng(0)
    assert wall_segments(rng.uniform(-3, 3, (20000, 3))) == []
    assert wall_segments(rng.uniform(-3, 3, (100, 3))) == []  # too few points to say anything


def test_same_cloud_same_walls():
    cloud = _cloud()
    a, b = wall_segments(cloud, seed=2), wall_segments(cloud, seed=2)
    assert [(s.offset, s.t0, s.t1) for s in a] == [(s.offset, s.t0, s.t1) for s in b]
