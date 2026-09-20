"""Loader for LiDAR-tier captures in the Stray-Scanner-style layout.

A capture folder holds, per moment N: ``depth/00000N.png`` (uint16, mm),
``confidence/00000N.png`` (uint8, 0/1/2), one row of ``odometry.csv`` (pose plus
intrinsics at RGB resolution) and one frame of ``rgb.mp4``. Depth is much smaller
than the RGB, so the intrinsics are rescaled to depth-pixel units here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class LidarScan:
    root: Path
    frame_ids: np.ndarray  # (N,) int, matches depth/<id>.png
    timestamps: np.ndarray  # (N,) seconds
    positions: np.ndarray  # (N, 3) metres, world frame
    quats: np.ndarray  # (N, 4) x, y, z, w
    intrinsics: np.ndarray  # (N, 4) fx, fy, cx, cy in *depth* pixels
    depth_size: tuple[int, int]  # (width, height)
    rgb_size: tuple[int, int]

    def __len__(self) -> int:
        return len(self.frame_ids)

    def depth_m(self, i: int) -> np.ndarray:
        """Depth in metres for the i-th row (not frame id); 0 marks no reading."""
        img = cv2.imread(str(self.root / "depth" / f"{self.frame_ids[i]:06d}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"missing depth frame {self.frame_ids[i]}")
        return img.astype(np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.root / "confidence" / f"{self.frame_ids[i]:06d}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"missing confidence frame {self.frame_ids[i]}")
        return img


def _video_size(path: Path) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(str(path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (w, h) if w and h else None


def load_scan(root: str | Path) -> LidarScan:
    root = Path(root)
    with open(root / "odometry.csv", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    body = rows[1:]
    col = {name: header.index(name) for name in header}
    need = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"]
    missing = [n for n in need if n not in col]
    if missing:
        raise ValueError(f"odometry.csv is missing columns {missing}")

    def num(name: str) -> np.ndarray:
        return np.array([float(r[col[name]]) for r in body], dtype=np.float64)

    frame_ids = np.array([int(r[col["frame"]]) for r in body])
    positions = np.stack([num("x"), num("y"), num("z")], axis=1)
    quats = np.stack([num("qx"), num("qy"), num("qz"), num("qw")], axis=1)

    # depth size from the first frame on disk
    first = cv2.imread(str(root / "depth" / f"{frame_ids[0]:06d}.png"), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise FileNotFoundError(f"no depth frames under {root / 'depth'}")
    dh, dw = first.shape[:2]

    if "fx" in col:
        k_rgb = np.stack([num("fx"), num("fy"), num("cx"), num("cy")], axis=1)
    else:
        k = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
        k_rgb = np.tile([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], (len(frame_ids), 1))

    rgb_size = _video_size(root / "rgb.mp4")
    if rgb_size is None:  # principal point sits near the image centre
        rgb_size = (round(2 * float(k_rgb[0, 2])), round(2 * float(k_rgb[0, 3])))
    sx, sy = dw / rgb_size[0], dh / rgb_size[1]
    intrinsics = k_rgb * np.array([sx, sy, sx, sy])

    return LidarScan(
        root=root,
        frame_ids=frame_ids,
        timestamps=num("timestamp"),
        positions=positions,
        quats=quats,
        intrinsics=intrinsics,
        depth_size=(dw, dh),
        rgb_size=rgb_size,
    )
