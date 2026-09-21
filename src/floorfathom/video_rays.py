"""Camera-ray visibility for the video tier: which cells of the floor plan the camera saw through.

Light reached the lens through the whole segment from the lens to the surface each pixel shows, so that
segment is air. Seen from above, every cell under it is free of wall, whether or not a depth model found
any floor points there. The room estimator (``layout.free_space``) needs to know which cells are open floor;
on a glossy floor the depth model returns the floor's reflection, the floor points are missing, and that
knowledge is lost. Rays give it back without asking anything of the floor.

Two guards keep the carving from eating real walls. A ray that would end below the floor (a reflection) is
cut where it meets the floor, and every ray stops a margin short of its surface, because depth is a few
percent off. A cell only counts as seen when rays from several different keyframes crossed it, so one wrong
depth map cannot open a hole in a wall. The mask never removes a wall: ``free_space`` still subtracts the wall
evidence from it.

The rays are stored in the SfM frame (unrotated, unscaled) so they do not depend on the gravity or scale
estimate, and are counted per chunk of the walk like the point cloud, so the bootstrap can leave chunks out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .io_video import Keyframes
from .layout import Grid
from .points_video import MAX_DEPTH_M, MIN_DEPTH_M, DenseCloud, _edge_mask
from .sfm import SfmResult

RAY_STRIDE = (16, 32)  # pixel spacing of the rays (horizontal, vertical): azimuth matters, elevation repeats
MIN_FRAMES = 3  # keyframes whose rays must cross a cell before it counts as seen
MARGIN_ABS, MARGIN_REL = 0.05, 0.02  # a ray stops this far (m, fraction of its length) short of its surface
FLOOR_CLEARANCE = 0.05  # a ray is cut this far above the floor plane (m)


@dataclass
class RayBundle:
    origins: np.ndarray  # (F, 3) camera centres, SfM frame and units
    ends: list[np.ndarray]  # per frame (K, 3) surface points the pixels show, SfM frame and units
    chunk: np.ndarray  # (F,) chunk of the walk each frame belongs to (same rule as the dense cloud)
    n_chunks: int
    flags: list[str] = field(default_factory=list)


def build_rays(
    kf: Keyframes,
    sfm: SfmResult,
    dense: DenseCloud,
    depth: Callable[[np.ndarray], np.ndarray],
    stride: tuple[int, int] = RAY_STRIDE,
) -> RayBundle:
    """Rays of every keyframe the dense cloud used, from the same depth maps (cached, so no model run)."""
    fx, fy, cx, cy = sfm.intrinsics
    frames = list(dense.frame_ratio)
    origins, ends = [], []
    for i in frames:
        bgr = cv2.imread(str(kf.directory / kf.names[i]))
        d = np.asarray(depth(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), dtype=np.float32)
        h, w = d.shape
        vv, uu = np.mgrid[stride[1] // 2 : h : stride[1], stride[0] // 2 : w : stride[0]]
        keep = ~_edge_mask(d)[vv, uu] & (d[vv, uu] > MIN_DEPTH_M) & (d[vv, uu] < MAX_DEPTH_M)
        zc = d[vv, uu][keep] / dense.frame_ratio[i]
        cam = np.column_stack([(uu[keep] - cx) / fx * zc, (vv[keep] - cy) / fy * zc, zc])
        origins.append(sfm.centers[i])
        ends.append((cam @ sfm.cam_to_world[i].T + sfm.centers[i]).astype(np.float32))
    n = len(frames)
    chunk = np.minimum(np.arange(n) * dense.n_chunks // max(n, 1), dense.n_chunks - 1)
    return RayBundle(np.array(origins), ends, chunk.astype(np.int16), dense.n_chunks)


def carve(
    bundle: RayBundle,
    rotation: np.ndarray,
    scale: float,
    grid: Grid,
    floor_y: float,
    margin_abs: float = MARGIN_ABS,
    margin_rel: float = MARGIN_REL,
) -> np.ndarray:
    """Per chunk, the number of keyframes whose rays crossed each grid cell: (n_chunks, nz, nx) uint8.

    ``rotation`` and ``scale`` (metres per SfM unit) put the rays in the plan frame, y up; ``floor_y`` is
    the floor height in that frame.
    """
    votes = np.zeros((bundle.n_chunks, grid.nz, grid.nx), dtype=np.uint8)
    rot = np.asarray(rotation).T
    for o_sfm, e_sfm, c in zip(bundle.origins, bundle.ends, bundle.chunk):
        if len(e_sfm) == 0:
            continue
        o = (o_sfm @ rot) * scale
        e = (e_sfm @ rot) * scale
        d = e - o
        length = np.linalg.norm(d, axis=1)
        t = np.ones(len(e))
        below = e[:, 1] < floor_y + FLOOR_CLEARANCE  # would end under the floor: a reflection or an error
        room = d[below, 1]
        t[below] = np.clip((floor_y + FLOOR_CLEARANCE - o[1]) / np.minimum(room, -1e-6), 0.0, 1.0)
        keep_len = t * length - (margin_abs + margin_rel * t * length)
        t = np.where(keep_len > 0, keep_len / np.maximum(length, 1e-9), 0.0)
        ok = t > 0
        if not ok.any():
            continue
        p0 = np.array([(o[0] - grid.x0) / grid.cell, (o[2] - grid.z0) / grid.cell])
        p1 = np.column_stack([(o[0] + t[ok] * d[ok, 0] - grid.x0) / grid.cell, (o[2] + t[ok] * d[ok, 2] - grid.z0) / grid.cell])
        segs = np.empty((len(p1), 2, 2), dtype=np.int32)
        segs[:, 0] = np.floor(p0)
        segs[:, 1] = np.floor(p1)
        img = np.zeros((grid.nz, grid.nx), dtype=np.uint8)
        cv2.polylines(img, segs, False, 1, 1)
        votes[c] += img
    return votes


def seen_from_votes(votes: np.ndarray, chunks: np.ndarray | None = None, min_frames: int = MIN_FRAMES) -> np.ndarray:
    """Cells crossed by rays of at least ``min_frames`` keyframes, over all chunks or only the given ones."""
    v = votes if chunks is None else votes[chunks]
    return v.sum(axis=0, dtype=np.int32) >= min_frames
