"""RANSAC plane fitting: recovers the floor of a tilted noisy room, deterministic, honest when there is none."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from floorfathom.ransac import fit_plane, fit_planes, level

UP = np.array([0.0, 1.0, 0.0])
TILT = Rotation.from_rotvec(np.radians([18.0, 25.0, -12.0]))


def _angle(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))


def _room(seed=0, ceiling=True):
    """A 5 x 4 x 2.6 m room (floor y=0) seen from a phone that is not level, with 8% junk points."""
    rng = np.random.default_rng(seed)
    u, v = rng.random((2, 6000)), rng.random((2, 6000))
    parts = [np.c_[5 * u[0], 0 * u[0], 4 * u[1]]]  # floor
    if ceiling:
        parts.append(np.c_[5 * v[0], 2.6 + 0 * v[0], 4 * v[1]])
    for w in range(4):
        a, b = rng.random((2, 2500))
        parts.append([np.c_[5 * a, 2.6 * b, 0 * a], np.c_[5 * a, 2.6 * b, 4 + 0 * a],
                      np.c_[0 * a, 2.6 * b, 4 * a], np.c_[5 + 0 * a, 2.6 * b, 4 * a]][w])
    pts = np.concatenate(parts) + rng.normal(0, 0.004, (sum(len(p) for p in parts), 3))
    junk = rng.uniform([-1, -1, -1], [6, 4, 5], (int(0.08 * len(pts)), 3))
    return TILT.apply(np.concatenate([pts, junk]))


def test_floor_normal_is_recovered_in_a_tilted_frame():
    planes = fit_planes(_room(), max_planes=6)
    up = TILT.apply(UP)
    assert _angle(planes[0].normal, up) < 1.0  # the floor is the largest plane
    assert len(planes) >= 5  # floor, ceiling and the walls
    assert planes[0].inlier_mask.mean() > 0.2


def test_level_makes_the_floor_horizontal_with_up_from_the_hint():
    pts = _room()
    floor = fit_planes(pts, max_planes=1)[0]
    flat, R = level(pts, floor, up_hint=TILT.apply(UP))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
    assert np.std(flat[floor.inlier_mask, 1]) < 0.01  # floor is a level surface now
    assert flat[:, 1].mean() > flat[floor.inlier_mask, 1].mean()  # the rest of the room is above it


def test_same_seed_same_planes():
    a, b = fit_planes(_room(), seed=3), fit_planes(_room(), seed=3)
    assert len(a) == len(b)
    for p, q in zip(a, b):
        assert np.array_equal(p.normal, q.normal) and np.array_equal(p.inlier_mask, q.inlier_mask)


def test_planes_do_not_share_points():
    masks = np.array([p.inlier_mask for p in fit_planes(_room(), max_planes=6)])
    assert masks.sum(0).max() == 1


def test_hint_restricts_the_search_to_floor_and_ceiling_or_to_walls():
    pts, up = _room(), TILT.apply(UP)
    flat = fit_planes(pts, max_planes=2, normal_hint=up, max_angle_deg=10)
    assert len(flat) == 2 and all(_angle(p.normal, up) < 1.0 for p in flat)
    walls = fit_planes(pts, max_planes=4, normal_hint=up, max_angle_deg=10, hint_mode="perpendicular")
    assert len(walls) == 4 and all(_angle(p.normal, up) > 89.0 for p in walls)


def test_no_plane_in_a_uniform_cloud_returns_nothing():
    pts = np.random.default_rng(1).uniform(0, 5, (20000, 3))
    assert fit_planes(pts, min_inlier_frac=0.2) == []
    assert fit_plane(pts[:2]) is None  # fewer than 3 points


def test_hint_without_an_angle_is_an_error():
    with pytest.raises(ValueError):
        fit_planes(_room(), normal_hint=UP)
