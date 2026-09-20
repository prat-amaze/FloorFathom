"""Dense point cloud for the video tier: a depth model's output fitted to the SfM reconstruction.

SfM is accurate but sparse: its points sit on textured things, while walls, floor and ceiling
are mostly blank. A depth model gives a depth for every pixel but in its own, drifting scale.
For each keyframe the model's depth is compared with SfM's own depth at the sparse points
(the median ratio) and divided by it, which puts every frame on SfM's common scale. Pixels are
then back-projected into one dense cloud. Metric scale is applied once, afterwards (``to_cloud``):
from a reference object when there is one, else from the median of the per-frame ratios.

Pixels next to sharp depth jumps are dropped: depth models smear object edges into flying points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .io_video import Keyframes
from .points import Cloud, voxel_downsample
from .sfm import SfmResult

MIN_DEPTH_M, MAX_DEPTH_M = 0.3, 12.0  # range of model depths trusted, in the model's own metres
MIN_ANCHORS = 30  # sparse points a keyframe needs to be fitted
EDGE_JUMP = 0.05  # relative depth change per pixel above which a pixel counts as an edge


@dataclass
class DenseCloud:
    points: np.ndarray  # (M, 3) float32, SfM frame and SfM units
    chunk: np.ndarray  # (M,) int16, contiguous frame chunk each point came from
    n_chunks: int
    frame_ratio: dict[int, float]  # keyframe -> model metres per SfM unit
    chunk_ratio: np.ndarray  # (n_chunks,) median of the frame ratios inside each chunk, nan if none
    flags: list[str] = field(default_factory=list)

    @property
    def scale_from_depth_model(self) -> float:
        """Metres per SfM unit according to the depth model (median over chunks)."""
        return float(np.nanmedian(self.chunk_ratio))


def fit_ratio(depth: np.ndarray, uv: np.ndarray, z_sfm: np.ndarray) -> float | None:
    """Median of (model depth / SfM depth) at the sparse points, ignoring outliers."""
    h, w = depth.shape
    u = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    d = depth[v, u]
    ok = (d > MIN_DEPTH_M) & (d < MAX_DEPTH_M) & (z_sfm > 0)
    if ok.sum() < MIN_ANCHORS:
        return None
    r = d[ok] / z_sfm[ok]
    med = float(np.median(r))
    mad = 1.4826 * float(np.median(np.abs(r - med)))
    inlier = np.abs(r - med) <= 3.0 * mad + 1e-12
    return float(np.median(r[inlier])) if inlier.sum() >= MIN_ANCHORS else med


def _edge_mask(depth: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    rel = np.hypot(gx, gy) / np.maximum(depth, 1e-3)
    return cv2.dilate((rel > EDGE_JUMP).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def build_dense_cloud(
    kf: Keyframes,
    sfm: SfmResult,
    depth: Callable[[np.ndarray], np.ndarray],
    n_chunks: int = 10,
    frame_stride: int = 2,
    pixel_stride: int = 8,
) -> DenseCloud | None:
    """Fit ``depth`` (uint8 RGB image -> float32 z-depth in metres, same size) to every ``frame_stride``-th
    registered keyframe and back-project it. None when no keyframe could be fitted."""
    fx, fy, cx, cy = sfm.intrinsics
    frames = [i for i in np.nonzero(sfm.registered)[0][::frame_stride] if i in sfm.observations]
    used, ratios, clouds = [], {}, []
    for i in frames:
        bgr = cv2.imread(str(kf.directory / kf.names[i]))
        if bgr is None:
            continue
        d = np.asarray(depth(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), dtype=np.float32)
        uv, z = sfm.observations[i]
        r = fit_ratio(d, uv, z) if len(z) else None
        if r is None or r <= 0:
            continue
        h, w = d.shape
        vv, uu = np.mgrid[pixel_stride // 2 : h : pixel_stride, pixel_stride // 2 : w : pixel_stride]
        keep = ~_edge_mask(d)[vv, uu] & (d[vv, uu] > MIN_DEPTH_M) & (d[vv, uu] < MAX_DEPTH_M)
        zc = d[vv, uu][keep] / r  # depth in SfM units
        cam = np.column_stack([(uu[keep] - cx) / fx * zc, (vv[keep] - cy) / fy * zc, zc])
        clouds.append((cam @ sfm.cam_to_world[i].T + sfm.centers[i]).astype(np.float32))
        used.append(i)
        ratios[i] = r
    if not used:
        return None
    chunk_of = np.minimum(np.arange(len(used)) * n_chunks // len(used), n_chunks - 1)
    chunk_ratio = np.full(n_chunks, np.nan)
    for c in range(n_chunks):
        r = [ratios[used[k]] for k in np.nonzero(chunk_of == c)[0]]
        if r:
            chunk_ratio[c] = float(np.median(r))
    flags = []
    spread = 1.4826 * float(np.median(np.abs(np.array(list(ratios.values())) - np.median(list(ratios.values())))))
    if spread / float(np.median(list(ratios.values()))) > 0.25:
        flags.append("depth_scale_unstable_between_frames")
    return DenseCloud(
        points=np.concatenate(clouds),
        chunk=np.concatenate([np.full(len(c), chunk_of[k], dtype=np.int16) for k, c in enumerate(clouds)]),
        n_chunks=n_chunks,
        frame_ratio=ratios,
        chunk_ratio=chunk_ratio,
        flags=flags,
    )


def to_cloud(dense: DenseCloud, rotation: np.ndarray, scale: float | None = None, voxel: float = 0.03) -> Cloud:
    """Gravity-align (``rotation``, world +y up) and scale to metres, then thin to one point per voxel per chunk.

    ``scale`` is metres per SfM unit; the depth model's own median is used when it is not given.
    """
    s = dense.scale_from_depth_model if scale is None else scale
    pts = (dense.points @ np.asarray(rotation).T) * s
    parts, chunks = [], []
    for c in range(dense.n_chunks):
        sel = pts[dense.chunk == c]
        if len(sel) == 0:
            continue
        thin = voxel_downsample(sel, voxel)
        parts.append(thin)
        chunks.append(np.full(len(thin), c, dtype=np.int16))
    return Cloud(np.concatenate(parts), np.concatenate(chunks), dense.n_chunks, chunk_scale=dense.chunk_ratio.copy())
