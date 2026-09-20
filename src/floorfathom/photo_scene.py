"""One room's registered stills to a gravity-levelled point cloud the shared estimator can read.

The estimator wants a cloud with +y up, a floor as its lowest large plane, and a trajectory
marking space that is known to be free. Stills give none of that: depth is per image, poses are
rotations only (every photo is taken from one spot), and gravity is unknown.

  1. each registered photo's depth is back-projected and rotated into the root photo's frame,
  2. the floor is found by RANSAC among points well below the camera, steered by the mean
     camera-up direction of the photos, and the cloud is rotated so the floor normal is +y,
  3. the "trajectory" is the station plus samples along every viewing direction up to just
     before the first wall points: a single station sees no floor at its own feet, so without
     the fan the free space would be a ring with a hole and the room would be dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .io_photos import PhotoSet
from .layout import BAND_HI, BAND_LO, CELL
from .photo_pose import Poses
from .points import Cloud, voxel_downsample
from .ransac import fit_plane, level

MIN_DEPTH_M, MAX_DEPTH_M = 0.3, 8.0  # depths trusted; a metric model is unreliable outside this
EDGE_JUMP = 0.05  # relative depth change per pixel above which a pixel counts as an object edge
PIXEL_STRIDE = 2
VOXEL = 0.02
FLOOR_BELOW = 0.5  # floor candidates lie at least this far below the camera (m)
MAX_TILT_DEG = 35.0  # the floor normal may differ this much from the photos' mean up
FLOOR_THRESH, FLOOR_RANGE_SLOPE = 0.03, 0.02  # inlier distance (m) and its growth per metre of range
CAMERA_HEIGHT_RANGE = (0.8, 2.0)  # a hand-held phone is not outside this; else the "floor" is furniture
MIN_FLOOR_FRAC = 0.15  # share of the below-camera points the floor plane must carry (provisional, check on real photos)
PAIR_STRIDE = 4  # pixel step when comparing two photos' depth in their overlap
MIN_OVERLAP_FRAC = 0.04  # share of an image a pair must overlap to say anything about relative scale
PAIR_SD_FLOOR = 0.03  # log-ratio noise every pair is assumed to have, so a lucky-tight pair does not dominate
MAX_SCALE_RESIDUAL = 0.12  # a pair disagreeing with the fitted scales by more than this (log units) is flagged
MIN_POINTS = 2000
FAN_BIN_DEG = 2.0
FAN_PERCENTILE = 10  # nearest wall points per direction, robust to a few stray pixels
FAN_MARGIN = 0.3  # stop the fan this far short of the wall so it does not touch the barrier (m)
FAN_MIN_POINTS = 20


@dataclass
class RoomScene:
    cloud: Cloud  # +y up, camera station at x = z = 0; chunk = index into ``used``
    traj_xz: np.ndarray  # station and the free-space fan, metres
    used: list[str]  # names of the fused images, in chunk order
    floor_y: float  # RANSAC floor height in the cloud frame (negative: below the camera)
    tilt_deg: float  # angle between the photos' mean up and the floor normal
    floor_support: float  # share of the below-camera points that lie on the floor plane
    scales: np.ndarray  # per fused image, the factor its depth was divided by (median 1)
    scale_spread: float  # std of the log scales: how much the model's scale wandered between photos
    rotation: np.ndarray  # root photo frame -> cloud frame
    flags: list[str] = field(default_factory=list)


def _edge_mask(depth: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    rel = np.hypot(gx, gy) / np.maximum(depth, 1e-3)
    return cv2.dilate((rel > EDGE_JUMP).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def back_project(depth: np.ndarray, f_px: float, stride: int = PIXEL_STRIDE) -> np.ndarray:
    """Camera-frame points (N, 3), OpenCV axes, of the trusted pixels of a z-depth map (principal point at the centre)."""
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    vv, uu = np.mgrid[stride // 2 : h : stride, stride // 2 : w : stride]
    d = depth[vv, uu]
    keep = np.isfinite(d) & (d > MIN_DEPTH_M) & (d < MAX_DEPTH_M) & ~_edge_mask(depth)[vv, uu]
    d = d[keep].astype(np.float64)
    return np.column_stack([(uu[keep] - w / 2) / f_px * d, (vv[keep] - h / 2) / f_px * d, d])


def _ray_distance(z: np.ndarray, f_px: float) -> np.ndarray:
    """Distance along the viewing ray from z-depth: z * |(x, y, 1)| of each pixel."""
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w]
    return z * np.sqrt(1.0 + ((u - w / 2) / f_px) ** 2 + ((v - h / 2) / f_px) ** 2)


def _pair_log_ratio(zi, fi, zj, fj, rij) -> tuple[float, float] | None:
    """(median, spread) of ln(ray distance in i / ray distance in j) over the overlap, or None if too small.

    Every photo is taken from one spot, so a ray seen by both has the same true length in both; the ratio
    is the relative scale of the two depth maps. ``rij`` rotates camera i directions into camera j.
    """
    h, w = zi.shape
    v, u = np.mgrid[0:h:PAIR_STRIDE, 0:w:PAIR_STRIDE]
    di = np.stack([(u - w / 2) / fi, (v - h / 2) / fi, np.ones(u.shape)], -1).reshape(-1, 3)
    dj = di @ rij.T
    ahead = dj[:, 2] > 0.2
    uj = np.where(ahead, dj[:, 0] / np.where(ahead, dj[:, 2], 1) * fj + w / 2, -1)
    vj = np.where(ahead, dj[:, 1] / np.where(ahead, dj[:, 2], 1) * fj + h / 2, -1)
    inside = ahead & (uj >= 0) & (uj <= w - 1) & (vj >= 0) & (vj <= h - 1)
    ui, vi = u.ravel()[inside], v.ravel()[inside]
    pj, qj = np.rint(uj[inside]).astype(int), np.rint(vj[inside]).astype(int)
    a, b = _ray_distance(zi, fi)[vi, ui], _ray_distance(zj, fj)[qj, pj]
    ok = np.isfinite(a) & np.isfinite(b) & (zi[vi, ui] > MIN_DEPTH_M) & (zj[qj, pj] > MIN_DEPTH_M)
    if ok.sum() < MIN_OVERLAP_FRAC * u.size:
        return None
    lr = np.log(a[ok] / b[ok])
    q1, med, q3 = np.percentile(lr, [25, 50, 75])
    return float(med), float((q3 - q1) / 1.349)


def harmonise_scales(depths, focals, rotations, names) -> tuple[np.ndarray, float, list[str]]:
    """Per-image scale factors (median 1) that make overlapping depth maps agree, their log spread, and flags.

    One equation per overlapping pair, ln s_i - ln s_j = the pair's median log ratio, weighted by the
    inverse variance of that ratio over the overlap (a tight pair, such as two views of a plain wall,
    counts more than one that sees a mirror). ``rotations`` are camera -> root; images that overlap no
    other image keep scale 1 and are flagged.
    """
    n = len(depths)
    rows, rhs, wts = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            got = _pair_log_ratio(depths[i], focals[i], depths[j], focals[j], rotations[j].T @ rotations[i])
            if got is None:
                continue
            row = np.zeros(n)
            row[i], row[j] = 1.0, -1.0
            rows.append(row)
            rhs.append(got[0])
            wts.append(1.0 / (got[1] ** 2 + PAIR_SD_FLOOR**2))
    flags: list[str] = []
    ln_s = np.zeros(n)
    if rows:
        a, b, w = np.array(rows), np.array(rhs), np.sqrt(np.array(wts))
        ln_s = np.linalg.lstsq(a * w[:, None], b * w, rcond=None)[0]  # minimum-norm: each group is centred on zero
        linked = np.abs(a).sum(axis=0) > 0
        ln_s[linked] -= np.median(ln_s[linked])
        ln_s[~linked] = 0.0
        if np.abs(a @ ln_s - b).max() > MAX_SCALE_RESIDUAL:
            flags.append("depth_scale_inconsistent")
    else:
        linked = np.zeros(n, bool)
    flags += [f"scale_unharmonised:{names[i]}" for i in range(n) if not linked[i]]
    return np.exp(ln_s), float(np.std(ln_s[linked])) if linked.any() else 0.0, flags


def _fan(points: np.ndarray, floor_y: float) -> np.ndarray:
    """The station and cells along each viewing direction up to just before the nearest wall points."""
    h = points[:, 1] - floor_y
    band = points[(h > BAND_LO) & (h < BAND_HI)][:, [0, 2]]
    out = [np.zeros((1, 2))]
    if len(band):
        az = np.degrees(np.arctan2(band[:, 0], band[:, 1]))
        r = np.hypot(band[:, 0], band[:, 1])
        b = np.floor((az + 180.0) / FAN_BIN_DEG).astype(int)
        for k in np.unique(b):
            sel = b == k
            if sel.sum() < FAN_MIN_POINTS:
                continue
            reach = float(np.percentile(r[sel], FAN_PERCENTILE)) - FAN_MARGIN
            if reach <= 0:
                continue
            a = np.radians((k + 0.5) * FAN_BIN_DEG - 180.0)
            s = np.arange(CELL, reach, CELL)
            out.append(np.column_stack([np.sin(a) * s, np.cos(a) * s]))
    return np.concatenate(out)


def build_scene(
    photos: PhotoSet,
    poses: Poses,
    depth: Callable[[np.ndarray], np.ndarray],
    seed: int = 0,
    voxel: float = VOXEL,
) -> RoomScene | None:
    """Fuse the registered photos of one room. ``depth`` maps an upright uint8 RGB image to float32 z-depth
    in metres at the same size. None, with the reason in a flag on the caller's side, when no floor is found."""
    used = [i for i, r in enumerate(poses.rotations) if r is not None]
    zs = [np.asarray(depth(photos.images[i].rgb), dtype=np.float32) for i in used]
    rots = [np.asarray(poses.rotations[i], float) for i in used]
    scales, scale_spread, flags = harmonise_scales(
        zs, [photos.images[i].f_px for i in used], rots, [photos.images[i].name for i in used])
    parts, chunk, ups = [], [], []
    for k, i in enumerate(used):
        im, z, r = photos.images[i], zs[k] / scales[k], rots[k]
        p = back_project(z, im.f_px)
        if len(p) < 0.01 * z.size / PIXEL_STRIDE**2:
            flags.append(f"image_without_depth:{im.name}")
        thin = voxel_downsample((p @ r.T).astype(np.float32), voxel)  # camera -> root; one point per voxel per image
        parts.append(thin)
        chunk.append(np.full(len(thin), k, dtype=np.int16))
        ups.append(r @ np.array([0.0, -1.0, 0.0]))  # camera up (OpenCV y is down) in the root frame
    if not parts or sum(len(p) for p in parts) < MIN_POINTS:
        return None
    pts, chunk = np.concatenate(parts), np.concatenate(chunk)
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)

    below = pts @ up < -FLOOR_BELOW
    plane = fit_plane(
        pts[below].astype(np.float64), thresh=FLOOR_THRESH, range_slope=FLOOR_RANGE_SLOPE,
        normal_hint=up, max_angle_deg=MAX_TILT_DEG, seed=seed,
    ) if below.sum() >= 3 else None
    support = 0.0 if plane is None else float(plane.inlier_mask.mean())
    if plane is None or support < MIN_FLOOR_FRAC:  # RANSAC always returns its best guess; a floor must be a large plane
        return None
    leveled, rot = level(pts, plane, up)
    floor_y = float(plane.offset)
    tilt = float(np.degrees(np.arccos(np.clip(plane.normal @ up, -1.0, 1.0))))
    if not CAMERA_HEIGHT_RANGE[0] <= -floor_y <= CAMERA_HEIGHT_RANGE[1]:
        flags.append("floor_plane_uncertain")
    cloud = Cloud(leveled.astype(np.float32), chunk, len(used))
    return RoomScene(cloud, _fan(cloud.points, floor_y), [photos.images[i].name for i in used],
                     floor_y, tilt, support, scales, scale_spread, rot, flags)
