"""A synthetic clip of one painted wall with known damage, seen from known poses: truth for the video damage adapter.

The wall lies in the plane z = 0 (plan y = 0), starting at plan (0, 0) and running along +x; the room is on the
z < 0 side. Frames are ray-cast from the camera poses, so what the adapter measures can be compared with the metres
that were painted. SfM is faked by writing the poses into arbitrary SfM units (a gravity rotation and a scale the
adapter must undo); the depth model is faked as the true depth times a bias, which the per-frame fit must undo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from floorfathom.io_video import Keyframes
from floorfathom.schema import Measurement, RoomPlan, Wall
from floorfathom.sfm import SfmResult

WALL_LEN, WALL_H, FLOOR_Y = 4.0, 2.6, -1.3
SIZE = (720, 480)  # keyframe (width, height)
FOCAL = 560.0
TEX_M = 0.002  # metres per texture pixel
SCALE = 0.37  # metres per SfM unit
GRAVITY = Rotation.from_euler("xyz", [20, -10, 15], degrees=True).as_matrix()  # SfM frame to gravity-aligned frame
DEPTH_BIAS = 0.9  # the fake depth model reads this fraction of the true depth
OCCLUDER = dict(z=-0.8, centre=(1.2, -0.5), radii=(0.2, 0.28))  # a round object (a vase, a lamp) in the room, world metres; not a rectangle, so no shape rule removes it


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation (OpenCV axes) looking from eye to target with world +y up."""
    z = target - eye
    z = z / np.linalg.norm(z)
    down = -np.array([0.0, 1.0, 0.0])
    down = down - (down @ z) * z
    down /= np.linalg.norm(down)
    x = np.cross(down, z)
    return np.stack([x, down, z], axis=1)


def texture(seed: int = 0, stain=(2.0, 1.2, 0.21, 0.25), crack: tuple | None = (3.1, 1.0, 0.22, 0.003)) -> np.ndarray:
    """Painted wall, float RGB 0..1, row 0 at the top. ``stain`` = (s, h, width, height) in m; ``crack`` = (s, h, length, width)."""
    rng = np.random.default_rng(seed)
    w, h = int(WALL_LEN / TEX_M), int(WALL_H / TEX_M)
    base = np.full((h, w, 3), (0.86, 0.84, 0.79), np.float32)
    grad = np.linspace(-0.03, 0.03, w, dtype=np.float32)[None, :, None] + np.linspace(0.03, -0.03, h, dtype=np.float32)[:, None, None]
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6) * 0.05
    tex += rng.normal(0, 0.006, (h, w)).astype(np.float32)
    img = base + grad + tex[..., None]
    if stain is not None:
        s, hh, sw, sh = stain
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.hypot((xx * TEX_M - s) / (sw / 2), ((h - yy) * TEX_M - hh) / (sh / 2))
        a = np.clip((1.15 - d) / 0.3, 0, 1)[..., None]  # soft edge, about 3 cm wide
        img = img * (1 - a) + a * img * np.array([0.82, 0.68, 0.42], np.float32)
    if crack is not None:
        s, hh, ln, wd = crack
        pts = np.array([[s, hh], [s + 0.3 * ln, hh + 0.1 * ln], [s + 0.6 * ln, hh - 0.05 * ln], [s + ln, hh + 0.12 * ln]])
        px = np.round(np.stack([pts[:, 0] / TEX_M, h - pts[:, 1] / TEX_M], axis=1)).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.polylines(mask, [px], False, 1, max(1, round(wd / TEX_M)))
        a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
        img = img * (1 - 0.8 * a)
    return np.clip(img, 0, 1)


def render(tex: np.ndarray, pose: np.ndarray, occluder: bool = False, noise: float = 0.004, seed: int = 0):
    """(uint8 RGB, true z-depth in metres) of the wall seen from ``pose`` (camera to world, metres)."""
    w, h = SIZE
    focal = FOCAL
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    d_cam = np.stack([(u - w / 2) / focal, (v - h / 2) / focal, np.ones_like(u)], axis=-1)
    d = d_cam @ pose[:3, :3].T
    eye = pose[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.0 - eye[2]) / d[..., 2]
    hit = eye + t[..., None] * d
    onwall = (t > 0) & (hit[..., 0] >= 0) & (hit[..., 0] < WALL_LEN) & (hit[..., 1] >= FLOOR_Y) & (hit[..., 1] < FLOOR_Y + WALL_H)
    mapx = (hit[..., 0] / TEX_M).astype(np.float32)
    mapy = ((FLOOR_Y + WALL_H - hit[..., 1]) / TEX_M).astype(np.float32)
    col = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    col[~onwall] = (0.35, 0.36, 0.4)
    depth = np.where(onwall, t, 9.0).astype(np.float32)
    if occluder:
        with np.errstate(divide="ignore", invalid="ignore"):
            to = (OCCLUDER["z"] - eye[2]) / d[..., 2]
        ho = eye + to[..., None] * d
        cx, cy = OCCLUDER["centre"]
        a, b = OCCLUDER["radii"]
        block = (to > 0) & (to < np.where(onwall, t, np.inf)) & (((ho[..., 0] - cx) / a) ** 2 + ((ho[..., 1] - cy) / b) ** 2 < 1)
        col[block] = (0.25, 0.2, 0.18)
        depth = np.where(block, to, depth).astype(np.float32)
    col = col + np.random.default_rng(seed).normal(0, noise, col.shape).astype(np.float32)
    return (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8), depth


def cameras(n: int = 14, dist: float = 2.0) -> list[np.ndarray]:
    """Poses along the room facing the wall: a slow sideways walk with a slight turn, camera 1.4 m up."""
    poses = []
    for k in range(n):
        f = k / (n - 1)
        eye = np.array([0.6 + 2.8 * f, 0.1, -dist + 0.25 * np.sin(6 * f)])
        target = np.array([2.0 + 0.9 * np.sin(5 * f), 0.0, 0.0])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = look_at(eye, target), eye
        poses.append(pose)
    return poses


def make_clip(
    directory: Path,
    tex: np.ndarray,
    poses: list[np.ndarray] | None = None,
    occluder_in: set[int] | None = None,
    noise: float = 0.004,
    obs_scale: float = 1.0,
) -> tuple[Keyframes, SfmResult, dict[bytes, np.ndarray]]:
    """Write the keyframes and build the fake SfM. The last item maps a decoded keyframe to its fake-model depth.

    ``obs_scale`` multiplies the depths the fake SfM reports at its sparse points, which makes the per-frame fit of the
    depth model come out that much off (its error on a real clip)."""
    poses = poses or cameras()
    occluder_in = occluder_in or set()
    directory.mkdir(parents=True, exist_ok=True)
    names, depths = [], {}
    n = len(poses)
    centres = np.zeros((n, 3))
    rot = np.zeros((n, 3, 3))
    obs = {}
    rng = np.random.default_rng(1)
    for i, pose in enumerate(poses):
        rgb, depth = render(tex, pose, occluder=i in occluder_in, noise=noise, seed=i)
        name = f"{i:05d}.jpg"
        cv2.imwrite(str(directory / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        decoded = cv2.cvtColor(cv2.imread(str(directory / name)), cv2.COLOR_BGR2RGB)
        depths[decoded.tobytes()] = depth * DEPTH_BIAS
        names.append(name)
        centres[i] = GRAVITY.T @ pose[:3, 3] / SCALE
        rot[i] = GRAVITY.T @ pose[:3, :3]
        uv = rng.uniform([40, 40], [SIZE[0] - 40, SIZE[1] - 40], (120, 2)).astype(np.float32)
        z = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        keep = (z < 8) & (z > 0.3)
        obs[i] = (uv[keep], (z[keep] / SCALE * obs_scale).astype(np.float64))
    kf = Keyframes(directory, names, np.arange(n), np.arange(n) / 6.0, np.ones(n), SIZE, 6.0, directory / "clip.MOV")
    sfm = SfmResult(
        registered=np.ones(n, bool), centers=centres, cam_to_world=rot, intrinsics=(FOCAL, FOCAL, SIZE[0] / 2, SIZE[1] / 2),
        image_size=SIZE, points=np.zeros((1, 3)), track_length=np.ones(1, int), point_error=np.zeros(1), n_models=1,
        mean_reprojection_px=0.5, observations=obs,
    )
    return kf, sfm, depths


def _m(v: float) -> Measurement:
    return Measurement(value=v, lo=v, hi=v, unit="m", method="test")


def room(ceiling: float | None = WALL_H) -> RoomPlan:
    """A 4 x 3 m room whose wall r0_w0 is the painted wall."""
    corners = [(0.0, 0.0), (WALL_LEN, 0.0), (WALL_LEN, 3.0), (0.0, 3.0)]
    walls = [
        Wall(id=f"r0_w{i}", start=a, end=b, length=_m(float(np.hypot(b[0] - a[0], b[1] - a[1]))), evidence="wall_points")
        for i, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1]))
    ]
    c = RoomPlan(
        id="r0", polygon=corners, walls=walls, ceiling_height=_m(ceiling) if ceiling else _m(0).model_copy(update={"value": None, "lo": None, "hi": None}),
        floor_area=_m(12.0).model_copy(update={"unit": "m2"}), openings=[],
    )
    return c
