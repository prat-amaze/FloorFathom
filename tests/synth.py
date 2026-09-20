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
