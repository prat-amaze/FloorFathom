"""Synthetic LiDAR captures with exactly known geometry.

Rooms are described as vertical wall segments in the horizontal plane (world x, z) plus
a floor and ceiling height. Depth frames are ray-cast for a phone that walks around
inside, with depth noise, and written in the layout the loader reads (depth PNGs in mm,
confidence PNGs, odometry.csv, an rgb.mp4 that only sets the RGB size).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

W, H = 256, 192
FX = FY = 213.05
CX, CY = 127.4, 95.7
SCALE = 1920 / 256  # RGB pixels per depth pixel
FLOOR_Y = -1.4  # the phone starts 1.4 m above the floor


@dataclass
class Wall:
    a: tuple[float, float]
    b: tuple[float, float]
    top: float | None = None  # None: floor to ceiling; a number: low furniture of that height


@dataclass
class Room:
    walls: list[Wall]
    height: float = 2.6
    has_ceiling: bool = True
    path: list[tuple[float, float]] = field(default_factory=list)  # camera positions (x, z)


def rect(x0, z0, x1, z1, gaps: dict[str, tuple[float, float]] | None = None) -> list[Wall]:
    """Four walls of an axis-aligned rectangle; ``gaps`` maps 'n','s','e','w' to a (lo, hi) span to leave open."""
    gaps = gaps or {}

    def side(p, q, key):
        if key not in gaps:
            return [Wall(p, q)]
        lo, hi = gaps[key]
        p, q = np.array(p, float), np.array(q, float)
        d = (q - p) / np.linalg.norm(q - p)
        return [Wall(tuple(p), tuple(p + d * lo)), Wall(tuple(p + d * hi), tuple(q))]

    return (
        side((x0, z0), (x1, z0), "s")
        + side((x1, z0), (x1, z1), "e")
        + side((x1, z1), (x0, z1), "n")
        + side((x0, z1), (x0, z0), "w")
    )


def _pose(pos, yaw, pitch) -> Rotation:
    f = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    up = np.array([0.0, 1.0, 0.0])
    down = -(up - (up @ f) * f)
    down /= np.linalg.norm(down)
    right = np.cross(down, f)  # OpenCV axes: x = y cross z
    return Rotation.from_matrix(np.stack([right, down, f], axis=1))


def _raycast(o, R, walls, height, has_ceiling) -> np.ndarray:
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d_cam = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u, float)], axis=-1).reshape(-1, 3)
    D = d_cam @ R.as_matrix().T  # world directions; camera z is 1 so t equals depth
    t = np.full(len(D), np.inf)
    # floor and ceiling
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = (FLOOR_Y - o[1]) / D[:, 1]
        t = np.where((D[:, 1] < 0) & (tf > 0), np.minimum(t, tf), t)
        if has_ceiling:
            tc = (FLOOR_Y + height - o[1]) / D[:, 1]
            t = np.where((D[:, 1] > 0) & (tc > 0), np.minimum(t, tc), t)
    for w in walls:
        a, b = np.array(w.a), np.array(w.b)
        e = b - a
        # o + t D = a + s e  (in x, z)
        det = D[:, 0] * (-e[1]) - D[:, 2] * (-e[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            rx, rz = a[0] - o[0], a[1] - o[2]
            tt = (rx * (-e[1]) - rz * (-e[0])) / det
            ss = (D[:, 0] * rz - D[:, 2] * rx) / det
        ok = (tt > 0) & (ss >= 0) & (ss <= 1) & np.isfinite(tt)
        if w.top is not None:
            ok &= (o[1] + tt * D[:, 1] - FLOOR_Y) <= w.top
        t = np.where(ok, np.minimum(t, tt), t)
    return t.reshape(H, W)


def write_capture(
    root: Path,
    room: Room,
    yaws: int = 12,
    pitches: tuple[float, ...] = (-25.0, 0.0, 30.0),
    noise: float = 0.008,
    seed: int = 0,
    max_range: float = 5.0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(exist_ok=True)
    (root / "confidence").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    frame = 0
    for px, pz in room.path:
        for k in range(yaws):
            yaw = 2 * np.pi * k / yaws
            for pdeg in pitches:
                pos = np.array([px, 0.0, pz])
                R = _pose(pos, yaw, np.radians(pdeg))
                t = _raycast(pos, R, room.walls, room.height, room.has_ceiling)
                t = t + rng.normal(0.0, noise + 0.002 * np.nan_to_num(t, posinf=0), t.shape)
                bad = ~np.isfinite(t) | (t > max_range) | (t <= 0.05)
                depth_mm = np.where(bad, 0, np.round(t * 1000)).astype(np.uint16)
                conf = np.where(bad, 0, 2).astype(np.uint8)
                cv2.imwrite(str(root / "depth" / f"{frame:06d}.png"), depth_mm)
                cv2.imwrite(str(root / "confidence" / f"{frame:06d}.png"), conf)
                q = R.as_quat()  # x y z w
                rows.append(
                    [frame * 0.05, f"{frame:06d}", *pos, *q, FX * SCALE, FY * SCALE, CX * SCALE, CY * SCALE, "", ""]
                )
                frame += 1
    header = "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"
    with open(root / "odometry.csv", "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(", ".join(str(v) for v in r) + "\n")
    with open(root / "camera_matrix.csv", "w") as f:
        f.write(f"{FX * SCALE}, 0.0, {CX * SCALE}\n0.0, {FY * SCALE}, {CY * SCALE}\n0.0, 0.0, 1.0")
    vw = cv2.VideoWriter(str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920, 1440))
    vw.write(np.zeros((1440, 1920, 3), np.uint8))
    vw.release()
    return root


def loop(x0, z0, x1, z1, n=6, inset=0.7) -> list[tuple[float, float]]:
    """Camera positions on a loop inside a rectangle."""
    cx, cz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
    rx, rz = 0.5 * (x1 - x0) - inset, 0.5 * (z1 - z0) - inset
    return [(cx + rx * np.cos(a), cz + rz * np.sin(a)) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


@dataclass
class PhotoStill:
    name: str
    rgb: np.ndarray  # (H, W, 3) uint8
    f35: float
    position: tuple[float, float, float]
    yaw_deg: float


def checkerboard_texture(kind: str, cell_m: float = 0.2, mpp: float = 0.01, seed: int = 0):
    """A (s, h) -> (N, 3) colour function with real high-frequency texture, for SIFT to match on.

    Noise is a deterministic function of (s, h) via a fixed sinusoidal hash, not a fresh Generator per
    call, so the same coordinate always renders the same speckle (needed so two photos of the same
    physical wall patch show consistent texture for SIFT to match across viewpoints).
    """
    base = {"brick": np.array([0.62, 0.42, 0.34]), "wood": np.array([0.55, 0.42, 0.28])}[kind]
    alt = base * 0.78
    phase = float(seed)

    def f(s: np.ndarray, h: np.ndarray) -> np.ndarray:
        s, h = np.asarray(s, float), np.asarray(h, float)
        cs, ch = (s / cell_m).astype(int), (h / cell_m).astype(int)
        on = (cs + ch) % 2 == 0
        speckle = np.sin((s / mpp) * 12.9898 + (h / mpp) * 78.233 + phase)
        speckle = speckle - np.floor(speckle)  # fractional part, deterministic pseudo-noise in [0, 1)
        noise = (speckle[:, None] - 0.5) * 0.06
        colour = np.where(on[:, None], base, alt)
        return np.clip(colour + noise, 0, 1)

    return f


def _raycast_textured(o, R, walls: list, height: float, floor_y: float, tex, w: int = 320, h_px: int = 240,
                      fx: float = 266.0) -> np.ndarray:
    """RGB render of one camera view using a texture function."""
    u, v = np.meshgrid(np.arange(w), np.arange(h_px))
    cx, cy = w / 2.0, h_px / 2.0
    d_cam = np.stack([(u - cx) / fx, (v - cy) / fx, np.ones_like(u, float)], axis=-1).reshape(-1, 3)
    D = d_cam @ R.as_matrix().T
    t = np.full(len(D), np.inf)
    owner = np.full(len(D), -1)
    s_hit = np.zeros(len(D))
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = (floor_y - o[1]) / D[:, 1]
        floor_hit = (D[:, 1] < 0) & (tf > 0) & (tf < t)
        t = np.where(floor_hit, tf, t)
        tc = (floor_y + height - o[1]) / D[:, 1]
        ceil_hit = (D[:, 1] > 0) & (tc > 0) & (tc < t)
        t = np.where(ceil_hit, tc, t)
    for wi, wall in enumerate(walls):
        a, b = np.array(wall.a), np.array(wall.b)
        e = b - a
        length = float(np.linalg.norm(e))
        det = D[:, 0] * (-e[1]) - D[:, 2] * (-e[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            rx, rz = a[0] - o[0], a[1] - o[2]
            tt = (rx * (-e[1]) - rz * (-e[0])) / det
            ss = (D[:, 0] * rz - D[:, 2] * rx) / det
        ok = (tt > 0) & (ss >= 0) & (ss <= 1) & np.isfinite(tt) & (tt < t)
        t = np.where(ok, tt, t)
        owner = np.where(ok, wi, owner)
        s_hit = np.where(ok, ss * length, s_hit)
    world_y = o[1] + t * D[:, 1]
    h_hit = np.clip(world_y - floor_y, 0.0, height)
    on_wall = owner >= 0
    colours = np.full((len(D), 3), 0.55)
    if on_wall.any():
        colours[on_wall] = tex(s_hit[on_wall], h_hit[on_wall])
    return (np.clip(colours, 0, 1).reshape(h_px, w, 3) * 255).astype(np.uint8)


def _sinusoidal_texture(H: int = 2048, W: int = 2048, n_freqs: int = 20, seed: int = 99) -> np.ndarray:
    """Return an (H, W, 3) uint8 texture map with rich high-frequency content for SIFT.

    Composed of many sinusoidal waves at different spatial frequencies and orientations so that
    every (u, v) neighbourhood is unique — SIFT descriptors matched across views with only a
    lateral camera shift produce >90% Essential-matrix inliers in practice.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), float)
    y, x = np.mgrid[:H, :W]
    for _ in range(n_freqs):
        fx = rng.uniform(0.002, 0.15)
        fy = rng.uniform(0.002, 0.15)
        phase = rng.uniform(0, 2 * np.pi, 3)
        amp = 1.0 / (1 + fx + fy)
        for c in range(3):
            img[:, :, c] += amp * np.sin(fx * x + fy * y + phase[c])
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _render_wall_strip(cam_x: float, cam_z: float, yaw_deg: float, tex_map: np.ndarray,
                       w: int = 320, h_px: int = 240, fx: float = 266.0,
                       wall_z: float = 3.0, pitch_deg: float = -5.0) -> np.ndarray:
    """Render one image: camera at (cam_x, 0, cam_z), looking toward wall at z=wall_z.

    The wall is a large textured plane; the texture is sampled from ``tex_map`` using the
    world x/y of each hit point.  Cameras shifted laterally (varying cam_x) produce the
    correct parallax for triangulation while keeping the same texture features in view.
    """
    R = _pose(np.array([cam_x, 0.0, cam_z]), np.radians(yaw_deg), np.radians(pitch_deg))
    u_px, v_px = np.meshgrid(np.arange(w), np.arange(h_px))
    cx, cy = w / 2.0, h_px / 2.0
    d_cam = np.stack([(u_px - cx) / fx, (v_px - cy) / fx, np.ones((h_px, w))], axis=-1).reshape(-1, 3)
    D = d_cam @ R.as_matrix().T  # world-space ray directions
    colours = np.full((len(D), 3), 0.5)
    o = np.array([cam_x, 0.0, cam_z])
    with np.errstate(divide="ignore", invalid="ignore"):
        tz = (wall_z - o[2]) / D[:, 2]
        wx = o[0] + tz * D[:, 0]
        wy = o[1] + tz * D[:, 1]
        # Texture bounds: x in [-3, 3], y in [-2, 2]
        hit = (D[:, 2] > 0) & (tz > 0) & (np.abs(wx) < 3.0) & (np.abs(wy) < 2.0)
        H, W_tex = tex_map.shape[:2]
        ui = np.clip(((wx[hit] + 3.0) / 6.0 * W_tex).astype(int), 0, W_tex - 1)
        vi = np.clip(((wy[hit] + 2.0) / 4.0 * H).astype(int), 0, H - 1)
        colours[hit] = tex_map[vi, ui] / 255.0
    return (np.clip(colours, 0, 1).reshape(h_px, w, 3) * 255).astype(np.uint8)


def _sinusoidal_sh_tex(seed: int = 0):
    """A (s, h) -> (N, 3) texture function using multi-frequency sinusoids for rich SIFT features."""
    rng = np.random.default_rng(seed + 77)
    fs = rng.uniform(5.0, 30.0, 12)
    fh = rng.uniform(5.0, 30.0, 12)
    ph = rng.uniform(0, 2 * np.pi, 12)

    def tex(s: np.ndarray, h: np.ndarray) -> np.ndarray:
        s, h = np.asarray(s, float), np.asarray(h, float)
        r = sum(np.sin(fs[i] * s + fh[i] * h + ph[i]) for i in range(0, 4)) / 4.0
        g = sum(np.sin(fs[i] * s + fh[i] * h + ph[i]) for i in range(4, 8)) / 4.0
        b = sum(np.sin(fs[i] * s + fh[i] * h + ph[i]) for i in range(8, 12)) / 4.0
        out = np.stack([r, g, b], axis=-1)
        out = (out - out.min()) / (out.max() - out.min() + 1e-9)
        return np.clip(out, 0, 1)

    return tex


def textured_room(walls: str, height: float, path: list, photo_positions: int, seed: int = 0):
    """Render ``photo_positions`` stills along ``path`` (x, y, z waypoints).

    All cameras face +z (yaw=0) so consecutive frames share a large overlap region while also
    seeing the side walls (non-planar 3D structure), which avoids the homography degeneracy
    that causes pycolmap SfM to fail on pure flat-wall scenes.
    """
    footprint = {"rect4x3": rect(0.0, 0.0, 4.0, 3.0)}[walls]
    tex = _sinusoidal_sh_tex(seed=seed)
    t = np.linspace(0, len(path) - 1, photo_positions)
    xs = np.interp(t, np.arange(len(path)), [p[0] for p in path])
    zs = np.interp(t, np.arange(len(path)), [p[2] for p in path])
    stills = []
    for k, (x, z) in enumerate(zip(xs, zs)):
        o = np.array([x, 0.0, z])
        R = _pose(o, 0.0, 0.0)  # yaw=0: face +z wall; side walls add non-planar structure
        rgb = _raycast_textured(o, R, footprint, height, FLOOR_Y, tex)
        stills.append(PhotoStill(f"synth_{k:03d}", rgb, f35=24.0, position=(x, 0.0, z), yaw_deg=0.0))
    return stills
