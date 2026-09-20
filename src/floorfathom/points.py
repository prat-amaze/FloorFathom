"""Depth frames to a world-space point cloud.

Conventions (found by a flatness search over depth flips, camera axis conventions and
pose direction, and confirmed on all three sample scans):
  * depth, intrinsics and pose share one image frame (no rotation of depth needed),
  * camera axes are OpenCV style: x right, y down, z forward,
  * the quaternion is camera-to-world,
  * world +y is up (gravity aligned), so the floor is the lowest large plane in y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .io_lidar import LidarScan

UP = 1  # world axis index that points up


@dataclass
class Cloud:
    points: np.ndarray  # (M, 3) float32, world metres
    chunk: np.ndarray  # (M,) int16, which contiguous frame chunk each point came from
    n_chunks: int
    # video only: metres per SfM unit as measured inside each chunk (nan where a chunk has none),
    # so a replicate that leaves chunks out can re-derive the scale from the chunks it keeps
    chunk_scale: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.points)


def frame_points(
    scan: LidarScan,
    i: int,
    min_conf: int = 2,
    z_min: float = 0.3,
    z_max: float = 4.0,
) -> np.ndarray:
    """World-space points of the i-th frame, from confident depth pixels only."""
    z = scan.depth_m(i)
    conf = scan.confidence(i)
    keep = (conf >= min_conf) & (z >= z_min) & (z <= z_max)
    if not keep.any():
        return np.empty((0, 3), dtype=np.float32)
    v, u = np.nonzero(keep)
    d = z[keep].astype(np.float64)
    fx, fy, cx, cy = scan.intrinsics[i]
    cam = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], axis=1)
    rot = Rotation.from_quat(scan.quats[i])
    return (rot.apply(cam) + scan.positions[i]).astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One centroid per voxel. Deterministic: the result is ordered by voxel key."""
    if len(points) == 0:
        return points
    ijk = np.floor(points / voxel).astype(np.int64)
    ijk -= ijk.min(axis=0)
    dims = ijk.max(axis=0) + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    out = np.empty((len(uniq), 3), dtype=np.float64)
    for ax in range(3):
        out[:, ax] = np.bincount(inv, weights=points[:, ax].astype(np.float64)) / counts
    return out.astype(np.float32)


def build_cloud(
    scan: LidarScan,
    target_frames: int = 1200,
    n_chunks: int = 10,
    voxel: float = 0.02,
    min_conf: int = 2,
    z_max: float = 4.0,
) -> Cloud:
    """Merge frames into one cloud.

    Frames are subsampled evenly to about ``target_frames``. They are grouped into
    ``n_chunks`` contiguous chunks and each chunk is voxelised on its own, so every
    point remembers its chunk (needed for the block bootstrap of the intervals).
    """
    n = len(scan)
    stride = max(1, n // target_frames)
    rows = np.arange(0, n, stride)
    chunk_of_row = np.minimum(np.arange(len(rows)) * n_chunks // len(rows), n_chunks - 1)
    pts, chunks = [], []
    for c in range(n_chunks):
        sel = rows[chunk_of_row == c]
        if len(sel) == 0:
            continue
        chunk_pts = np.concatenate([frame_points(scan, int(i), min_conf, z_max=z_max) for i in sel])
        chunk_pts = voxel_downsample(chunk_pts, voxel)
        pts.append(chunk_pts)
        chunks.append(np.full(len(chunk_pts), c, dtype=np.int16))
    return Cloud(np.concatenate(pts), np.concatenate(chunks), n_chunks)
