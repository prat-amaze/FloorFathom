"""Rotation registration of room photos: known rotations recovered, failures dropped and flagged."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from floorfathom.io_photos import PhotoImage, focal_px
from floorfathom.photo_pose import kabsch, register_rotations

W, H = 240, 320  # portrait, like the iPhone stills at a working size
F = focal_px(22, W, H)


def _panorama(seed=0, h=1024, w=2048):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, np.uint8)
    for _ in range(2500):
        x, y, s = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(6, 40))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + s), colour, -1)
        else:
            cv2.circle(img, (x, y), s // 2, colour, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def _rotation(yaw, pitch=0.0, roll=0.0):
    """Camera-to-world rotation."""
    return Rotation.from_euler("yxz", [yaw, pitch, roll], degrees=True).as_matrix()


def _view(pano, c2w):
    ph, pw = pano.shape[:2]
    xs, ys = np.meshgrid(np.arange(W) - W / 2 + 0.5, np.arange(H) - H / 2 + 0.5)
    d = np.stack([xs / F, ys / F, np.ones_like(xs)], -1) @ c2w.T  # world direction of every pixel
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    u = (np.arctan2(d[..., 0], d[..., 2]) / (2 * np.pi) + 0.5) * pw
    v = (np.arcsin(d[..., 1]) / np.pi + 0.5) * ph
    return cv2.remap(pano, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _image(name, rgb):
    return PhotoImage(name, "x", rgb, (W, H), 22.0, F)


def _capture(yaws, seed=0, wobble=3.0, blank=()):
    rng = np.random.default_rng(seed + 100)
    pano = _panorama(seed)
    c2w = [_rotation(y, *rng.uniform(-wobble, wobble, 2)) for y in yaws]
    imgs = [_image(f"{i}.jpg", np.full((H, W, 3), 90, np.uint8) if i in blank else _view(pano, c2w[i]))
            for i in range(len(yaws))]
    return imgs, c2w


def _errors(poses, c2w):
    """Angle (deg) between each estimated camera->root rotation and the truth."""
    root = poses.root
    out = {}
    for i, r in enumerate(poses.rotations):
        if r is not None:
            true = c2w[root].T @ c2w[i]
            out[i] = np.degrees(Rotation.from_matrix(r @ true.T).magnitude())
    return out


def test_kabsch_recovers_a_known_rotation_single_and_batched():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3))
    r = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    assert np.allclose(kabsch(a, a @ r.T), r, atol=1e-9)
    batch = kabsch(np.stack([a, a]), np.stack([a @ r.T, a]))
    assert np.allclose(batch[0], r, atol=1e-9) and np.allclose(batch[1], np.eye(3), atol=1e-9)


def test_chain_of_overlapping_photos_recovers_the_rotations():
    imgs, c2w = _capture([0, 35, 70, 105, 140])
    poses = register_rotations(imgs)
    err = _errors(poses, c2w)
    assert len(err) == 5 and max(err.values()) < 1.0
    assert not any(f.startswith("image_unregistered") for f in poses.flags)
    assert poses.loop_error_deg is not None and poses.loop_error_deg < 1.5


def test_protocol_spacing_of_45_degrees_still_registers():
    imgs, c2w = _capture([0, 45, 90, 135, 180], seed=1)  # about 26% overlap between neighbours
    poses = register_rotations(imgs)
    err = _errors(poses, c2w)
    assert len(err) == 5 and max(err.values()) < 1.5


def test_a_blank_photo_is_dropped_and_flagged_and_the_rest_still_register():
    imgs, c2w = _capture([0, 20, 40, 60, 80], blank={2})  # 1 and 3 are 40 degrees apart, inside the 61 degree view
    poses = register_rotations(imgs)
    assert poses.rotations[2] is None and "image_unregistered:2.jpg" in poses.flags
    assert max(_errors(poses, c2w).values()) < 1.0 and len(_errors(poses, c2w)) == 4


def test_disconnected_groups_keep_only_the_largest():
    imgs, c2w = _capture([0, 30, 60, 180, 210])  # the two groups look at opposite walls
    poses = register_rotations(imgs)
    assert [r is not None for r in poses.rotations] == [True, True, True, False, False]
    assert {"image_unregistered:3.jpg", "image_unregistered:4.jpg"} <= set(poses.flags)


def test_single_or_unlinkable_photos_degrade_without_crashing():
    imgs, _ = _capture([0])
    poses = register_rotations(imgs)
    assert poses.rotations == [None] and "insufficient_registration" in poses.flags
    blank, _ = _capture([0, 35], blank={0, 1})
    assert "insufficient_registration" in register_rotations(blank).flags


def test_same_seed_gives_identical_rotations():
    imgs, _ = _capture([0, 35, 70])
    a, b = register_rotations(imgs, seed=4), register_rotations(imgs, seed=4)
    assert a.root == b.root and all(np.array_equal(x, y) for x, y in zip(a.rotations, b.rotations))
