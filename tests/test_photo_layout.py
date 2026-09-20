"""Wall segments from the levelled photo cloud: the four walls of a synthetic room, and nothing from clutter."""

import numpy as np
from synth import Wall, rect
from synth_photo import CAMERA_HEIGHT, room_photos

from floorfathom import layout as L
from floorfathom.photo_layout import find_openings, floor_and_ceiling, outline_from_segments, wall_segments
from floorfathom.photo_scene import build_scene

HEIGHT = 2.6
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]


GRID = L.Grid.around(np.array([[-8.0, -8.0], [8.0, 8.0]]))


def _cloud(noise=0.005, bias=1.0, walls=None, yaws=YAWS):
    n = len(yaws)
    photos, poses, depth, _ = room_photos(walls or rect(0, 0, 5, 4), HEIGHT, (2.2, CAMERA_HEIGHT, 1.7), yaws,
                                          [8.0] + [0.0] * (n - 1), [5.0] + [0.0] * (n - 1), noise=noise, bias=bias)
    return build_scene(photos, poses, depth).cloud.points


def _outline(**kw):
    return outline_from_segments(wall_segments(_cloud(**kw)), GRID)


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


def test_the_outline_of_a_rectangular_room_has_four_supported_edges_at_the_true_lengths_and_area():
    o = _outline()
    assert len(o.edges) == 4 and all(e.supported for e in o.edges)
    assert np.allclose(sorted(e.length for e in o.edges), [4, 4, 5, 5], atol=0.1)
    assert abs(L.signed_area(o.polygon) - 20.0) < 0.3  # positive: counter-clockwise
    assert all(e.support > 0.85 for e in o.edges)
    assert np.allclose(o.polygon, [e.p0 for e in o.edges])


def test_a_scale_bias_scales_the_outline_lengths_and_area():
    o = _outline(bias=1.1)
    assert np.allclose(sorted(e.length for e in o.edges), [4.4, 4.4, 5.5, 5.5], atol=0.15)
    assert abs(L.signed_area(o.polygon) - 20.0 * 1.21) < 0.6


def test_a_doorway_gap_keeps_one_wall_edge_with_lower_support():
    o = _outline(walls=rect(0, 0, 5, 4, gaps={"n": (1.5, 2.5)}))
    assert len(o.edges) == 4 and np.allclose(sorted(e.length for e in o.edges), [4, 4, 5, 5], atol=0.15)
    gappy = [e for e in o.edges if e.support < 0.95]
    assert len(gappy) == 1 and abs(gappy[0].length - 5) < 0.15 and 0.7 < gappy[0].support < 0.9  # 1 m of 5 m not seen


def test_walls_not_seen_leave_unsupported_edges_and_are_not_drawn_as_walls():
    o = _outline(yaws=[0, 45, 90, 135])  # half of the room was photographed
    assert o is not None and any(not e.supported for e in o.edges)
    assert sum(e.length for e in o.edges if not e.supported) > 1.0


def test_fewer_than_two_walls_give_no_outline():
    segs = wall_segments(_cloud())
    assert outline_from_segments(segs[:1], GRID) is None and outline_from_segments([], GRID) is None


def test_floor_and_ceiling_heights_are_found_and_their_gap_is_the_room_height():
    floor, ceiling = floor_and_ceiling(_cloud())
    assert abs(floor.y + CAMERA_HEIGHT) < 0.06 and abs(ceiling.y - (HEIGHT - CAMERA_HEIGHT)) < 0.06
    assert abs((ceiling.y - floor.y) - HEIGHT) < 0.1


def test_a_doorway_that_something_is_seen_through_is_an_opening_of_the_right_width():
    walls = rect(0, 0, 5, 4, gaps={"n": (1.5, 2.5)}) + [Wall((5, 6), (0, 6))]  # a corridor wall behind the 1 m gap
    cloud = _cloud(walls=walls)
    segs = wall_segments(cloud)
    outline = outline_from_segments(segs, GRID)
    ops = find_openings(segs, outline, cloud)
    assert len(ops) == 1 and abs(ops[0].width - 1.0) < 0.2
    assert 0 <= ops[0].edge_index < len(outline.edges) and outline.edges[ops[0].edge_index].supported


def test_a_gap_nothing_is_seen_through_is_not_called_a_doorway():
    cloud = _cloud(walls=rect(0, 0, 5, 4, gaps={"n": (1.5, 2.5)}))  # nothing behind the gap: unobserved, not known open
    segs = wall_segments(cloud)
    assert find_openings(segs, outline_from_segments(segs, GRID), cloud) == []
