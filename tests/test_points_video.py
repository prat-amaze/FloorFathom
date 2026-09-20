"""Dense video cloud: fitting a depth model to SfM depth, back-projecting, scaling."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from floorfathom.io_video import Keyframes
from floorfathom.points_video import _edge_mask, build_dense_cloud, fit_ratio, to_cloud
from floorfathom.sfm import SfmResult

W, H, F = 72, 128, 90.0
WALL_Z = 3.0  # the wall's distance in SfM units
MODEL_RATIO = 1.7  # the fake depth model reads this many "metres" per SfM unit


def _scene(tmp_path, n=20):
    centres = np.column_stack([np.linspace(-0.5, 0.5, n), np.zeros(n), np.zeros(n)])
    rng = np.random.default_rng(0)
    obs = {}
    for i in range(n):
        uv = np.column_stack([rng.uniform(0, W, 60), rng.uniform(0, H, 60)])
        obs[i] = (uv, np.full(60, WALL_Z - centres[i, 2]))
    sfm = SfmResult(
        registered=np.ones(n, dtype=bool),
        centers=centres,
        cam_to_world=np.repeat(np.eye(3)[None], n, axis=0),
        intrinsics=(F, F, W / 2, H / 2),
        image_size=(W, H),
        points=np.zeros((1, 3)),
        track_length=np.ones(1, dtype=int),
        point_error=np.zeros(1),
        n_models=1,
        mean_reprojection_px=0.5,
        observations=obs,
    )
    names = []
    for i in range(n):
        names.append(f"{i:05d}.jpg")
        cv2.imwrite(str(tmp_path / names[-1]), np.zeros((H, W, 3), np.uint8))
    z = np.zeros(n)
    kf = Keyframes(tmp_path, names, z, z, z, (W, H), 30.0, tmp_path / "clip.MOV")
    return sfm, kf


def _flat_wall_depth(rgb):
    return np.full(rgb.shape[:2], MODEL_RATIO * WALL_Z, np.float32)


def test_ratio_recovered_despite_bad_anchor_points():
    rng = np.random.default_rng(1)
    uv = np.column_stack([rng.uniform(0, W, 100), rng.uniform(0, H, 100)])
    z = np.full(100, WALL_Z)
    depth = np.full((H, W), MODEL_RATIO * WALL_Z, np.float32)
    z[:10] *= 3.0  # ten anchors whose SfM depth is wrong
    assert fit_ratio(depth, uv, z) == pytest.approx(MODEL_RATIO, rel=0.01)
    assert fit_ratio(depth, uv[:10], z[:10]) is None  # too few anchors to trust


def test_dense_points_land_back_on_the_wall_in_sfm_units(tmp_path):
    sfm, kf = _scene(tmp_path)
    dense = build_dense_cloud(kf, sfm, _flat_wall_depth, n_chunks=4, frame_stride=1, pixel_stride=4)
    assert dense is not None and dense.flags == []
    assert np.allclose(dense.points[:, 2], WALL_Z, atol=1e-3)
    assert all(r == pytest.approx(MODEL_RATIO, rel=1e-3) for r in dense.frame_ratio.values())
    assert dense.scale_from_depth_model == pytest.approx(MODEL_RATIO, rel=1e-3)
    assert set(np.unique(dense.chunk)) == {0, 1, 2, 3}
    assert dense.chunk_ratio.shape == (4,)


def test_pixels_at_a_sharp_depth_jump_are_dropped():
    depth = np.full((H, W), 2.0, np.float32)
    depth[:, W // 2 :] = 5.0  # a step from 2 m to 5 m
    mask = _edge_mask(depth)
    assert mask[:, W // 2 - 1 : W // 2 + 1].all()  # the jump itself
    assert not mask[:, :10].any() and not mask[:, -10:].any()  # flat areas far from it


def test_frames_that_cannot_be_fitted_are_skipped(tmp_path):
    sfm, kf = _scene(tmp_path, n=6)
    sfm.observations[0] = (sfm.observations[0][0][:5], sfm.observations[0][1][:5])  # too few anchors
    dense = build_dense_cloud(kf, sfm, _flat_wall_depth, n_chunks=2, frame_stride=1, pixel_stride=8)
    assert 0 not in dense.frame_ratio and len(dense.frame_ratio) == 5
    assert build_dense_cloud(kf, sfm, lambda rgb: np.zeros(rgb.shape[:2], np.float32), 2, 1, 8) is None


def test_gravity_alignment_and_metric_scale_are_applied(tmp_path):
    sfm, kf = _scene(tmp_path)
    dense = build_dense_cloud(kf, sfm, _flat_wall_depth, n_chunks=4, frame_stride=1, pixel_stride=4)
    turn = Rotation.from_euler("x", 90, degrees=True).as_matrix()  # z (wall distance) becomes -y or +y
    cloud = to_cloud(dense, turn, scale=0.5, voxel=0.02)
    # the wall was WALL_Z units from the cameras; after the turn its distance is on the y axis, scaled by 0.5
    assert np.allclose(np.abs(cloud.points[:, 1]), 0.5 * WALL_Z, atol=1e-3)
    assert len(cloud) < len(dense.points)  # thinned to one point per voxel per chunk
    assert cloud.chunk_scale is not None and cloud.chunk_scale.shape == (4,)
    default = to_cloud(dense, turn)  # no scale given: the depth model's own median
    assert np.allclose(np.abs(default.points[:, 1]), MODEL_RATIO * WALL_Z, atol=1e-2)
