"""Metric scale from a reference object of known length that stays still in the room.

SfM gives the room in arbitrary units. An object whose true length is known and which does
not move gives the missing number: metres per SfM unit = true length / its length in SfM
units. Its two ends are seen in many frames; each sighting is a ray from the camera through
the end, and the end sits where the rays from all frames meet (closest point to all rays).

The precision of the result is measured, not assumed. Frames are cut into contiguous time
chunks, a few chunks are left out at a time (the same delete-d jackknife as
``uncertainty.py``) and the spread of the recovered length is the uncertainty. Rays that are
nearly parallel (the camera hardly moved sideways) pin the distance down badly, so the
largest angle between rays is reported and a small one is flagged.

The reference must be fixed in the room. An object carried with the camera, or seen while the
camera only turns on the spot, has no parallax and gives no scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .io_video import Keyframes
from .sfm import SfmResult
from .uncertainty import jackknife_scale

# The reference ruler of capture_protocol.md, shared by the video and photo tiers: the yellow body,
# end to end, measured with a tape on 2026-09-20 (the printed scale is 30 cm, the body is longer).
REFERENCE_LENGTH_M = 0.316
STRIP_COLOUR = "yellow"  # for stills (photo tier)
VIDEO_STRIP_COLOUR = "yellow_video"  # the same ruler as video frames render it

MIN_FRAMES = 8
MIN_RAY_ANGLE_DEG = 5.0
MAX_RESIDUAL_PX = 3.0
N_CHUNKS = 10


@dataclass
class SegmentObs:
    frame: int  # keyframe index
    top: tuple[float, float]  # pixel (u, v) of the end with the smaller v
    bottom: tuple[float, float]


@dataclass
class Anchor:
    length_sfm: float  # length of the reference in SfM units
    scale: float | None  # metres per SfM unit, None while the true length is not given
    rel_sigma: float  # standard error of the length, as a fraction of it
    n_frames: int  # frames that agree with the triangulated ends
    max_ray_angle_deg: float
    residual_px: float  # median reprojection error of the ends
    flags: list[str] = field(default_factory=list)


def _colour_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]  # OpenCV hue runs 0-179, so yellow is about 30
    if colour == "white":  # bright and unsaturated
        return (s < 60) & (v > 150)
    if colour == "yellow":  # stills: the ruler is hue 32-40, a wooden door 13-24
        return (h >= 28) & (h <= 42) & (s > 90) & (v > 150)
    if colour == "yellow_video":  # video renders the same ruler hue 23-27 and the door 11-15; stills' range would miss it
        return (h >= 20) & (h <= 45) & (s > 90) & (v > 140)
    raise ValueError(f"unknown strip colour {colour!r}")


def find_strip(bgr: np.ndarray, near: tuple[float, float] | None = None, min_length_px: float = 40.0, colour: str = "white"):
    """Two ends ((u, v) top, (u, v) bottom) of an elongated, roughly vertical strip of one colour, or None.

    White means bright and unsaturated, which separates it from a wooden door; yellow is picked by
    hue, which also leaves out white door frames and walls. Large patches of the colour fail on
    size and shape, and blobs touching the image border may be cut off.
    """
    h, w = bgr.shape[:2]
    mask = _colour_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), colour).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best, best_score = None, -np.inf
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 150 or area > 0.05 * h * w:
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        pts = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(float)  # (u, v)
        c = pts.mean(axis=0)
        _, sing, vt = np.linalg.svd(pts - c, full_matrices=False)
        axis = vt[0] if vt[0][1] >= 0 else -vt[0]
        if sing[1] < 1e-6 or sing[0] / sing[1] < 3.5 or abs(axis[1]) < 0.82:  # elongated, within 35 deg of vertical
            continue
        t = (pts - c) @ axis
        length = float(t.max() - t.min())
        if length < min_length_px:
            continue
        top, bottom = c + axis * t.min(), c + axis * t.max()
        if near is not None:
            dist = float(np.hypot(*(c - np.asarray(near))))
            if dist > 0.25 * w:
                continue
            score = -dist
        else:
            score = length
        if score > best_score:
            best, best_score = (tuple(top), tuple(bottom)), score
    return best


def track_strip(
    kf: Keyframes, sfm: SfmResult, max_misses: int = 4, until_s: float | None = None, colour: str = "white"
) -> list[SegmentObs]:
    """Follow the strip through the registered keyframes in time order.

    ``until_s`` stops the search after that many seconds: the protocol puts the strip move at the
    start of the clip, and a white door frame later in the walk must not be taken for the strip.
    """
    obs: list[SegmentObs] = []
    near, misses = None, 0
    for i in range(len(kf)):
        if until_s is not None and kf.time_s[i] > until_s:
            break
        if not sfm.registered[i]:
            continue
        found = find_strip(cv2.imread(str(kf.directory / kf.names[i])), near, colour=colour)
        if found is None:
            misses += 1
            if obs and misses > max_misses:
                break
            continue
        misses = 0
        obs.append(SegmentObs(i, found[0], found[1]))
        near = tuple(0.5 * (np.asarray(found[0]) + np.asarray(found[1])))
    return obs


def _rays(sfm: SfmResult, frames: np.ndarray, uv: np.ndarray):
    fx, fy, cx, cy = sfm.intrinsics
    cam = np.column_stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))])
    d = np.einsum("nij,nj->ni", sfm.cam_to_world[frames], cam)
    return sfm.centers[frames], d / np.linalg.norm(d, axis=1, keepdims=True)


def _closest_point(o: np.ndarray, d: np.ndarray) -> np.ndarray:
    proj = np.eye(3)[None] - d[:, :, None] * d[:, None, :]  # projector onto the plane normal to each ray
    rhs = (proj @ o[:, :, None]).sum(axis=0)[:, 0]
    return np.linalg.lstsq(proj.sum(axis=0), rhs, rcond=None)[0]


def _reproject_error(sfm: SfmResult, frames: np.ndarray, uv: np.ndarray, x: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = sfm.intrinsics
    cam = np.einsum("nji,nj->ni", sfm.cam_to_world[frames], x - sfm.centers[frames])  # R^T (x - centre)
    pred = np.column_stack([fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy])
    return np.linalg.norm(pred - uv, axis=1)


def _segment_length(sfm: SfmResult, obs: list[SegmentObs]):
    """(length, frames kept, residual px, max ray angle in degrees), or None."""
    frames = np.array([o.frame for o in obs])
    ends = [np.array([o.top for o in obs]), np.array([o.bottom for o in obs])]
    keep = np.ones(len(obs), dtype=bool)
    x = [None, None]
    for _ in range(3):
        if keep.sum() < 3:
            return None
        for k in (0, 1):
            o, d = _rays(sfm, frames[keep], ends[k][keep])
            x[k] = _closest_point(o, d)
        err = np.maximum(
            *(_reproject_error(sfm, frames, ends[k], x[k]) for k in (0, 1))
        )
        keep = err <= max(MAX_RESIDUAL_PX, 3.0 * float(np.median(err[keep])))
    o, d = _rays(sfm, frames[keep], ends[0][keep])
    angle = float(np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)).max()))
    return float(np.linalg.norm(x[0] - x[1])), int(keep.sum()), float(np.median(err[keep])), angle


def estimate_anchor(
    sfm: SfmResult,
    obs: list[SegmentObs],
    true_length_m: float | None = None,
    replicates: int = 30,
    seed: int = 0,
) -> Anchor | None:
    """Length of the reference in SfM units, its uncertainty and, given its true length, the scale."""
    full = _segment_length(sfm, obs) if len(obs) >= 3 else None
    if full is None:
        return None
    length, n_used, residual, angle = full
    chunks = np.array_split(np.arange(len(obs)), N_CHUNKS)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        drop = set(rng.choice(N_CHUNKS, size=2, replace=False).tolist())
        sub = [obs[j] for c, idx in enumerate(chunks) if c not in drop for j in idx]
        res = _segment_length(sfm, sub) if len(sub) >= 3 else None
        if res is not None:
            samples.append(res[0])
    if len(samples) >= 5:
        s = np.asarray(samples)
        sd = 1.4826 * float(np.median(np.abs(s - np.median(s))))
        rel_frames = jackknife_scale(N_CHUNKS, 2) * sd / length
    else:
        rel_frames = 1.0
    # Errors that every frame shares (a weak baseline) do not show up when frames are left out.
    # Add the pixel noise seen in the data and see how strongly the geometry magnifies it.
    sigma_px = max(residual, 0.5) / 1.177  # median of a 2-D Gaussian error is 1.177 sigma
    noisy = []
    for _ in range(replicates):
        jittered = [
            SegmentObs(o.frame, tuple(np.add(o.top, rng.normal(0, sigma_px, 2))), tuple(np.add(o.bottom, rng.normal(0, sigma_px, 2))))
            for o in obs
        ]
        res = _segment_length(sfm, jittered)
        if res is not None:
            noisy.append(res[0])
    rel_noise = 1.4826 * float(np.median(np.abs(np.asarray(noisy) - np.median(noisy)))) / length if len(noisy) >= 5 else 1.0
    rel = float(np.hypot(rel_frames, rel_noise))
    flags = []
    if n_used < MIN_FRAMES:
        flags.append("anchor_too_few_frames")
    if angle < MIN_RAY_ANGLE_DEG:
        flags.append("anchor_weak_baseline")
    if residual > MAX_RESIDUAL_PX:
        flags.append("anchor_large_residual")
    return Anchor(
        length_sfm=length,
        scale=None if true_length_m is None else true_length_m / length,
        rel_sigma=float(rel),
        n_frames=n_used,
        max_ray_angle_deg=angle,
        residual_px=residual,
        flags=flags,
    )
