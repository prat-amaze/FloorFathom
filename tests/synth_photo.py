"""Synthetic room photos with exactly known geometry: true z-depth from any camera pose.

World is gravity aligned (+y up, floor at y = 0). The camera uses OpenCV axes (x right, y down,
z forward) like the real pipeline. The stand-in for the depth model returns the true depth, with
optional noise and scale bias, so a test can compare what the pipeline recovers to the room.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from floorfathom.io_photos import PhotoImage, PhotoSet, focal_px
from floorfathom.photo_pose import Poses

W, H = 240, 320  # portrait, the working size the photo tests use
F35 = 22.0
F = focal_px(F35, W, H)
CAMERA_HEIGHT = 1.4


def camera_to_world(yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world rotation for a camera looking along yaw (0 = +z, towards +x as it grows) and pitch (up positive)."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    down = -(np.array([0.0, 1.0, 0.0]) - f[1] * f)
    down /= np.linalg.norm(down)
    right = np.cross(down, f)
    r = np.stack([right, down, f], axis=1)
    return r @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()


def render_depth(walls, height: float, position, c2w: np.ndarray, w: int = W, h: int = H, f: float = F) -> np.ndarray:
    """True z-depth (h, w) of a room made of vertical ``walls`` (objects with .a, .b in x-z), a floor and a ceiling."""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    cam = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u, float)], axis=-1).reshape(-1, 3)
    d = cam @ c2w.T  # world direction of every pixel; camera z is 1, so the ray parameter is the z-depth
    o = np.asarray(position, float)
    t = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = (0.0 - o[1]) / d[:, 1]
        t = np.where((d[:, 1] < 0) & (tf > 0), np.minimum(t, tf), t)
        tc = (height - o[1]) / d[:, 1]
        t = np.where((d[:, 1] > 0) & (tc > 0), np.minimum(t, tc), t)
        for wall in walls:
            a, b = np.array(wall.a), np.array(wall.b)
            e = b - a
            det = d[:, 0] * (-e[1]) - d[:, 2] * (-e[0])
            rx, rz = a[0] - o[0], a[1] - o[2]
            tt = (rx * (-e[1]) - rz * (-e[0])) / det
            ss = (d[:, 0] * rz - d[:, 2] * rx) / det
            ok = (tt > 0) & (ss >= 0) & (ss <= 1) & np.isfinite(tt)
            t = np.where(ok, np.minimum(t, tt), t)
    return t.reshape(h, w).astype(np.float32)


class FakeDepth:
    """Stand-in for the depth model. The image index is read back from pixel (0, 0, 0).

    ``bias`` is one scale factor for every image, or a list with one factor per image (the real model's
    scale differs from photo to photo).
    """

    def __init__(self, walls, height, position, c2ws, noise=0.0, bias=1.0, seed=0):
        self.args = (walls, height, position)
        self.c2ws, self.noise = c2ws, noise
        self.bias = list(bias) if np.ndim(bias) else [bias] * len(c2ws)
        self.rng = np.random.default_rng(seed)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        i = int(rgb[0, 0, 0])
        z = render_depth(*self.args, self.c2ws[i]) * self.bias[i]
        z = np.where(np.isfinite(z), z * (1 + self.noise * self.rng.standard_normal(z.shape)), 0.0)
        return z.astype(np.float32)


def room_photos(walls, height, position, yaws, pitches=None, rolls=None, **depth_kw):
    """(PhotoSet, Poses, FakeDepth, world_to_root) for cameras at one station.

    The root frame is camera 0's, as in the real registration, so a camera 0 that is pitched or
    rolled leaves gravity off its y axis and the pipeline has to find it. Poses are camera-to-root
    rotations; ``world_to_root`` maps world directions (up is +y) into that frame.
    """
    n = len(yaws)
    pitches = pitches if pitches is not None else [0.0] * n
    rolls = rolls if rolls is not None else [0.0] * n
    c2w = [camera_to_world(y, p, r) for y, p, r in zip(yaws, pitches, rolls)]
    world_to_root = c2w[0].T
    images = []
    for i in range(n):
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[0, 0, 0] = i
        images.append(PhotoImage(f"{i}.jpg", str(i), rgb, (W, H), F35, F))
    poses = Poses([world_to_root @ c for c in c2w], 0, {}, 0.0, [])
    return PhotoSet("room", images), poses, FakeDepth(walls, height, position, c2w, **depth_kw), world_to_root


def paint_ruler(rgb: np.ndarray, position, c2w: np.ndarray, centre, length: float = 0.316, width: float = 0.04) -> bool:
    """Paint an upright yellow ruler into ``rgb`` where a camera at ``position`` would see it on a wall at ``centre``
    (x, y, z), facing +z. Returns False, painting nothing, when it is not fully in view."""
    cx, cy, cz = centre
    corners = np.array([[cx + sx * width / 2, cy + sy * length / 2, cz] for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    cam = (corners - np.asarray(position, float)) @ c2w  # world -> camera
    if np.any(cam[:, 2] < 0.2):
        return False
    uv = np.column_stack([F * cam[:, 0] / cam[:, 2] + W / 2, F * cam[:, 1] / cam[:, 2] + H / 2])
    if np.any(uv < 3) or np.any(uv[:, 0] > W - 4) or np.any(uv[:, 1] > H - 4):
        return False
    cv2.fillConvexPoly(rgb, np.round(uv).astype(np.int32), (255, 255, 0))
    return True
