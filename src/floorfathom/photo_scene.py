"""One room's registered stills to a gravity-levelled point cloud the shared estimator can read.

The estimator wants a cloud with +y up, a floor as its lowest large plane, and a trajectory
marking space that is known to be free. Stills give none of that: depth is per image, poses are
rotations only (every photo is taken from one spot), and gravity is unknown.

  1. each registered photo's depth is back-projected and rotated into the root photo's frame,
  2. gravity is the direction, within 35 degrees of the photos' mean camera-up, in which the walls
     are thinnest (a tilted "up" smears every wall in the top-down view); the cloud is rotated so
     it is +y, and the floor height is then read the way the estimator reads it,
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
from .planes import find_floor
from .points import Cloud, voxel_downsample
from .ransac import Plane, fit_plane, level

MIN_DEPTH_M, MAX_DEPTH_M = 0.3, 8.0  # depths trusted; a metric model is unreliable outside this
EDGE_JUMP = 0.05  # relative depth change per pixel above which a pixel counts as an object edge
PIXEL_STRIDE = 2
VOXEL = 0.02
FLOOR_BELOW = 0.5  # floor candidates lie at least this far below the camera (m)
GRAVITY_SEARCH_DEG = 35.0  # gravity is looked for this far from the photos' mean camera-up
GRAVITY_STAGES = ((3.0, GRAVITY_SEARCH_DEG), (1.0, 3.0), (0.25, 1.0))  # (step, half-width) in degrees, coarse to fine
GRAVITY_AT_LIMIT_DEG = 3.0  # a search result this close to the edge of the cone is not trusted
EYE_SLAB = (-0.6, 0.4)  # heights relative to the camera (m) whose points are scored as walls
WALL_CELL = 0.05  # top-down cell (m) in which wall thinness is measured
GRAVITY_MAX_POINTS = 150_000
FLOOR_TILT_DEG = 10.0  # the floor-evidence plane may differ this much from the searched gravity
FLOOR_THRESH, FLOOR_RANGE_SLOPE = 0.05, 0.03  # inlier distance (m) and its growth per metre of range
CAMERA_HEIGHT_RANGE = (0.8, 2.0)  # a hand-held phone is not outside this; else the "floor" is furniture
MIN_FLOOR_FRAC = 0.15  # share of the below-camera points the floor plane must carry (provisional, check on real photos)
MIN_AREA_SPREAD, MIN_AREA_ASPECT = 0.3, 0.12  # a floor spreads over an area: std (m) along its narrow axis, and narrow / wide
PAIR_STRIDE = 4  # pixel step when comparing two photos' depth in their overlap
MIN_OVERLAP_FRAC = 0.04  # share of an image a pair must overlap to say anything about relative scale
PAIR_SD_FLOOR = 0.03  # log-ratio noise every pair is assumed to have, so a lucky-tight pair does not dominate
MAX_SCALE_RESIDUAL = 0.12  # a pair disagreeing with the fitted scales by more than this (log units) is flagged
MAX_PAIR_SPREAD = 0.15  # links between well-registered photos measured at most 0.11, a wrong one 0.19 and up
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
    depths: list[np.ndarray] = field(default_factory=list)  # per fused image, its z-depth divided by its scale (cloud metres)


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


def pair_log_ratios(depths, focals, rotations) -> dict[tuple[int, int], tuple[float, float]]:
    """(median, spread) of the log depth ratio for every pair of images that overlap enough; ``rotations`` are camera -> root."""
    out = {}
    for i in range(len(depths)):
        for j in range(i + 1, len(depths)):
            got = _pair_log_ratio(depths[i], focals[i], depths[j], focals[j], rotations[j].T @ rotations[i])
            if got is not None:
                out[(i, j)] = got
    return out


def inconsistent_images(pairs: dict[tuple[int, int], tuple[float, float]], n: int) -> list[int]:
    """Images, worst first, that a wrong rotation has probably put in the wrong place.

    A photo turned wrongly sees different surfaces than its neighbours think it does, so its depth
    disagrees with theirs wherever they overlap: the pair's spread is above ``MAX_PAIR_SPREAD``. A bad pair
    says only that one of its two photos is wrong; the blame goes to the photo whose best *other* pair is
    worse (a photo with no other pair has none). With a tie, as for a lone link, nothing is dropped.
    Worst pair first, one photo at a time, since a bad photo also spoils the pairs of its good neighbours.
    """
    live, dropped = dict(pairs), []
    while True:
        bad = [(sd, k) for k, (_, sd) in live.items() if sd > MAX_PAIR_SPREAD]
        if not bad:
            return dropped
        _, (i, j) = max(bad)

        def best_other(x: int) -> float:
            return min((sd for k, (_, sd) in live.items() if x in k and k != (i, j)), default=np.inf)

        bi, bj = best_other(i), best_other(j)
        if bi == bj:
            live.pop((i, j))
            continue
        blame = i if bi > bj else j
        dropped.append(blame)
        live = {k: v for k, v in live.items() if blame not in k}


def harmonise_scales(depths, focals, rotations, names, pairs=None) -> tuple[np.ndarray, float, list[str]]:
    """Per-image scale factors (median 1) that make overlapping depth maps agree, their log spread, and flags.

    One equation per overlapping pair, ln s_i - ln s_j = the pair's median log ratio, weighted by the
    inverse variance of that ratio over the overlap (a tight pair, such as two views of a plain wall,
    counts more than one that sees a mirror). ``rotations`` are camera -> root; images that overlap no
    other image keep scale 1 and are flagged.
    """
    n = len(depths)
    pairs = pair_log_ratios(depths, focals, rotations) if pairs is None else pairs
    rows, rhs, wts = [], [], []
    for (i, j), (med, sd) in pairs.items():
        row = np.zeros(n)
        row[i], row[j] = 1.0, -1.0
        rows.append(row)
        rhs.append(med)
        wts.append(1.0 / (sd**2 + PAIR_SD_FLOOR**2))
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


def _is_area(p: np.ndarray) -> bool:
    """True when points spread over an area. A horizontal slab through a wall is a thin line of points and fails."""
    if len(p) < 10:
        return False
    s = np.linalg.svd(p - p.mean(axis=0), compute_uv=False) / np.sqrt(len(p))
    return bool(s[1] >= MIN_AREA_SPREAD and s[1] >= MIN_AREA_ASPECT * s[0])


def _plane_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def _wall_concentration(points: np.ndarray, up: np.ndarray) -> float:
    """How thin the walls are when ``up`` is taken as vertical: the average number of eye-level points sharing a point's
    5 cm top-down cell. Per point, not per pair: a tilt that pushes points out of the slab would otherwise look better."""
    s = points @ up
    p = points[(s > EYE_SLAB[0]) & (s < EYE_SLAB[1])]
    if len(p) == 0:
        return 0.0
    e1, e2 = _plane_basis(up)
    ij = np.floor(np.column_stack([p @ e1, p @ e2]) / WALL_CELL).astype(np.int64)
    ij -= ij.min(axis=0)
    counts = np.bincount(ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]).astype(float)
    return float((counts**2).sum() / counts.sum())


def _directions_around(u0: np.ndarray, half_deg: float, step_deg: float) -> np.ndarray:
    """Unit vectors on a grid of tilts (in two perpendicular directions) up to ``half_deg`` from ``u0``, including ``u0``."""
    e1, e2 = _plane_basis(u0)
    g = np.arange(-half_deg, half_deg + 1e-9, step_deg)
    tilts = [(a, b) for a in g for b in g if np.hypot(a, b) <= half_deg + 1e-9]
    d = np.array([u0 + np.tan(np.radians(a)) * e1 + np.tan(np.radians(b)) * e2 for a, b in tilts])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def gravity_from_walls(points: np.ndarray, prior_up: np.ndarray) -> tuple[np.ndarray, float]:
    """(up, angle from ``prior_up`` in degrees): the direction near the prior in which walls are thinnest.

    The photos are held upright, so their mean camera-up is a good first guess, but a phone pitched to
    see the ceiling or floor puts it 20 degrees or more off; tilting "up" wrongly turns every wall into a
    band as wide as the tilt times the wall height. Coarse-to-fine grid search, deterministic.
    """
    sub = points[:: max(1, len(points) // GRAVITY_MAX_POINTS)].astype(np.float64)
    best = np.asarray(prior_up, float) / np.linalg.norm(prior_up)
    for step, half in GRAVITY_STAGES:
        cands = _directions_around(best, half, step)
        best = cands[int(np.argmax([_wall_concentration(sub, u) for u in cands]))]
    return best, float(np.degrees(np.arccos(np.clip(best @ np.asarray(prior_up, float) / np.linalg.norm(prior_up), -1, 1))))


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
    focals = [photos.images[i].f_px for i in used]
    pairs = pair_log_ratios(zs, focals, rots)
    flags: list[str] = []
    bad = inconsistent_images(pairs, len(used))
    if bad:
        flags += [f"image_inconsistent:{photos.images[used[k]].name}" for k in bad]
        keep = [k for k in range(len(used)) if k not in bad]
        used, zs, rots, focals = ([x[k] for k in keep] for x in (used, zs, rots, focals))
        pairs = None  # indices changed
    scales, scale_spread, scale_flags = harmonise_scales(
        zs, focals, rots, [photos.images[i].name for i in used], pairs)
    flags += scale_flags
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
    prior = np.mean(ups, axis=0)
    prior /= np.linalg.norm(prior)
    up, tilt = gravity_from_walls(pts, prior)
    if tilt > GRAVITY_SEARCH_DEG - GRAVITY_AT_LIMIT_DEG:
        flags.append("gravity_at_search_limit")
    leveled, rot = level(pts, Plane(up, 0.0, np.zeros(0, bool)), up)

    # the floor height is read as the estimator will read it, so the fan and the estimator agree;
    # RANSAC only provides evidence that a floor was seen at all (a view of bare walls has none)
    floor = find_floor(leveled[:, 1])
    below = leveled[leveled[:, 1] < -FLOOR_BELOW]
    plane = fit_plane(
        below, thresh=FLOOR_THRESH, range_slope=FLOOR_RANGE_SLOPE, normal_hint=(0.0, 1.0, 0.0),
        max_angle_deg=FLOOR_TILT_DEG, seed=seed,
    ) if len(below) >= 3 else None
    support = 0.0 if plane is None else float(plane.inlier_mask.mean())
    # RANSAC always returns its best guess; a floor must be a large plane that spreads over an area
    if floor is None or plane is None or support < MIN_FLOOR_FRAC or not _is_area(below[plane.inlier_mask]):
        return None
    floor_y = floor.y
    if not CAMERA_HEIGHT_RANGE[0] <= -floor_y <= CAMERA_HEIGHT_RANGE[1]:
        flags.append("floor_plane_uncertain")
    cloud = Cloud(leveled.astype(np.float32), chunk, len(used))
    return RoomScene(cloud, _fan(cloud.points, floor_y), [photos.images[i].name for i in used],
                     floor_y, tilt, support, scales, scale_spread, rot, flags,
                     [(zs[k] / scales[k]).astype(np.float32) for k in range(len(used))])
