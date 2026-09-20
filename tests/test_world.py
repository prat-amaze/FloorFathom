"""Gravity alignment on a synthetic room with known "up", scrambled like an SfM result."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from floorfathom.sfm import SfmResult
from floorfathom.world import estimate_gravity

UP = np.array([0.0, 1.0, 0.0])


def _room_points(rng: np.random.Generator) -> np.ndarray:
    """5 x 4 m room, 2.6 m ceiling: floor, ceiling, four walls and a 0.75 m table top."""
    floor = np.c_[rng.uniform(0, 5, 500), np.zeros(500), rng.uniform(0, 4, 500)]
    ceiling = np.c_[rng.uniform(0, 5, 400), np.full(400, 2.6), rng.uniform(0, 4, 400)]
    table = np.c_[rng.uniform(1, 2, 150), np.full(150, 0.75), rng.uniform(1, 2, 150)]
    walls = []
    for _ in range(600):
        wall = rng.integers(4)
        t, h = rng.uniform(0, 1), rng.uniform(0, 2.6)
        walls.append([[t * 5, h, 0.0], [5.0, h, t * 4], [t * 5, h, 4.0], [0.0, h, t * 4]][wall])
    pts = np.vstack([floor, ceiling, table, np.array(walls)])
    return pts + rng.normal(0, 0.01, pts.shape)


def _camera_path(rng: np.random.Generator, n: int = 60):
    """Walk around the room at 1.4 m, phone upright in portrait with a little pitch and roll."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centres = np.c_[2.5 + 1.6 * np.cos(ang), np.full(n, 1.4), 2.0 + 1.2 * np.sin(ang)]
    centres[:, 1] += rng.normal(0, 0.03, n)
    rots = []
    for a in ang:
        look = np.array([np.cos(a + np.pi), 0.0, np.sin(a + np.pi)])  # towards the walls
        z = look
        y = -UP  # OpenCV: y points down in the image
        x = np.cross(y, z)
        base = np.c_[x, y, z]
        wobble = Rotation.from_euler("xz", rng.normal(0, np.radians(6), 2)).as_matrix()
        rots.append(base @ wobble)
    return centres, np.array(rots)


def _sfm_result(seed: int, scale: float) -> tuple[SfmResult, np.ndarray]:
    rng = np.random.default_rng(seed)
    pts = _room_points(rng)
    centres, rots = _camera_path(rng)
    q = Rotation.random(random_state=seed).as_matrix()  # the arbitrary SfM frame
    n = len(centres)
    result = SfmResult(
        registered=np.ones(n, dtype=bool),
        centers=scale * centres @ q.T,
        cam_to_world=np.einsum("ij,njk->nik", q, rots),
        intrinsics=(798.0, 798.0, 360.0, 640.0),
        image_size=(720, 1280),
        points=scale * pts @ q.T,
        track_length=np.full(len(pts), 4),
        point_error=np.full(len(pts), 0.5),
        n_models=1,
        mean_reprojection_px=0.5,
    )
    return result, q @ UP  # the true up direction in the SfM frame


@pytest.mark.parametrize("seed", range(6))
def test_gravity_recovered_from_scrambled_room(seed):
    sfm, true_up = _sfm_result(seed, scale=0.37)  # SfM units are arbitrary
    g = estimate_gravity(sfm)
    err = np.degrees(np.arccos(np.clip(g.up @ true_up, -1, 1)))
    assert err < 2.0, f"up off by {err:.2f} deg"
    assert g.flags == []
    # the rotation puts the recovered up on +y, and keeps handedness
    assert np.allclose(g.rotation @ g.up, UP, atol=1e-9)
    assert np.isclose(np.linalg.det(g.rotation), 1.0)
    # the aligned camera path is level, and the floor is below it
    aligned_centres = sfm.centers @ g.rotation.T
    aligned_pts = sfm.points @ g.rotation.T
    assert np.std(aligned_centres[:, 1]) < 0.03 * np.ptp(aligned_pts[:, 1])
    assert np.median(aligned_centres[:, 1]) > aligned_pts[:, 1].min()


def test_tilted_camera_cue_is_flagged_not_trusted():
    """A phone rolled 30 degrees for the whole capture (one heading, so the roll does not
    average out) puts the camera cue outside the search area. The layers still say where up
    is, but the search cannot reach it, so the result must be flagged."""
    sfm, _ = _sfm_result(0, scale=1.0)
    look_x = np.c_[np.cross(-UP, [1.0, 0.0, 0.0]), -UP, [1.0, 0.0, 0.0]]  # camera facing +x, upright
    roll = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    q = Rotation.random(random_state=0).as_matrix()
    sfm.cam_to_world = np.repeat((q @ look_x @ roll)[None], len(sfm.cam_to_world), axis=0)
    g = estimate_gravity(sfm)
    assert "gravity_uncertain" in g.flags
