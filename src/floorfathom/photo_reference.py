"""Metric scale of a photo room from the yellow reference ruler of the capture protocol, seen in one photo.

The photos turn on the spot, so there is no parallax and a size cannot be triangulated. A single view still
gives it: the ruler lies in a wall whose plane the layout has already fitted, and

  * the two ends of the ruler in the picture are two viewing rays,
  * the wall plane (normal from the fitted wall, distance unknown) meets each ray at a point proportional to
    that distance, so the ruler's length in the picture fixes the wall's true distance from the camera,
  * the depth model's distance to the same wall (the fitted plane's offset in the cloud) is what it believed.

Their ratio is metres per depth-model metre. Using the wall plane's normal corrects the foreshortening of a
ruler seen off head-on; the protocol keeps the view within 15 degrees so that error stays small.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .anchor import REFERENCE_LENGTH_M
from .io_photos import PhotoSet
from .photo_layout import WallSegment
from .photo_pose import Poses
from .photo_scene import RoomScene

# Colour of the ruler in the phone's photos: hue 23-25 on OpenCV's 0-179 scale (in the video clips it is 32-40), a
# wooden door 14-16. The shape test keeps a yellow gas pipe or a cable, much longer than wide, from passing.
RULER_HUE, RULER_MIN_SAT, RULER_MIN_VAL = (19, 34), 100, 140
RULER_ASPECT = (5.0, 13.0)  # length / width of the yellow body: 31.6 / 4 = 7.9, a little more when seen at an angle
RULER_MIN_MEDIAN_HUE = 22  # the ruler is a uniform 23-29; bright wood edges and window trim sit at the mask's low edge, 19-20
RULER_MIN_FILL = 0.7  # the ruler is a solid rectangle: blob area / (length x width); pipes with fittings and trim are sparse
MAX_HEIGHT_M = 0.6  # the protocol puts the ruler's centre at phone height: within this of the camera
MAX_VIEW_DISAGREE = 0.15  # views of one ruler that differ more than this in scale are not trusted
MIN_RULER_FRAC = 0.06  # the ruler is at least this fraction of the picture's long side (60 px of 1008)
MAX_VIEW_DEG = 45.0  # a ruler seen further off head-on than this is not used; the protocol asks for 15
END_PX_SIGMA = 1.5  # pixels of error at each end of the detected ruler
NORMAL_SIGMA_RAD = 0.09  # error of the wall normal taken from the fitted plane (about 5 degrees)
PLANE_REL_SIGMA = 0.02  # error of the model's distance to the wall (bowed depth)
LENGTH_REL_SIGMA = 0.003  # 1 mm on 31.6 cm
FLOOR_REL_SIGMA = 0.02  # no scale is claimed better than this, whatever one photo says
PLAUSIBLE_FACTOR = (0.4, 3.0)  # a metric depth model is not further off than this; outside it the detection is wrong


@dataclass
class RulerScale:
    factor: float  # metres per depth-model metre
    rel_sigma: float
    n_photos: int


def find_ruler(bgr: np.ndarray, min_length_px: float):
    """Ends ((u, v) top, (u, v) bottom) of the yellow ruler in a photo, or None: the elongated yellow blob, upright
    within 35 degrees, that does not touch the border and whose length over width is closest to the ruler's."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] >= RULER_HUE[0]) & (hsv[..., 0] <= RULER_HUE[1]) & (hsv[..., 1] > RULER_MIN_SAT)
            & (hsv[..., 2] > RULER_MIN_VAL)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best, best_err = None, np.inf
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 150 or area > 0.05 * h * w or x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        pts = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(float)  # (u, v)
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        axis = vt[0] if vt[0][1] >= 0 else -vt[0]
        along, across = (pts - c) @ axis, (pts - c) @ vt[1]
        length, width = float(along.max() - along.min()), float(across.max() - across.min())
        if abs(axis[1]) < 0.82 or length < min_length_px or width < 1 or not RULER_ASPECT[0] <= length / width <= RULER_ASPECT[1]:
            continue
        if area / (length * width) < RULER_MIN_FILL or np.median(hsv[..., 0][labels == i]) < RULER_MIN_MEDIAN_HUE:
            continue
        err = abs(np.log(length / width / 7.9))
        if err < best_err:
            best, best_err = (tuple(c + axis * along.min()), tuple(c + axis * along.max())), err
    return best


def _one_view(rgb: np.ndarray, f_px: float, cloud_from_cam: np.ndarray, segs: list[WallSegment], length: float):
    """(factor, relative sigma) from one photo, or a flag saying why not."""
    found = find_ruler(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), MIN_RULER_FRAC * max(rgb.shape[:2]))
    if found is None:
        return None, "ruler_not_seen"
    h, w = rgb.shape[:2]
    ends = [np.array([(u - w / 2) / f_px, (v - h / 2) / f_px, 1.0]) for u, v in found]
    ends = [r / np.linalg.norm(r) for r in ends]
    mid = cloud_from_cam @ (ends[0] + ends[1])
    ray = mid[[0, 2]] / np.linalg.norm(mid[[0, 2]])  # horizontal direction to the ruler, in the cloud frame
    best = None
    for s in segs:  # the wall the ruler is on: the nearest fitted wall along that direction
        den = float(ray @ s.normal)
        if den < 1e-3:
            continue
        t = s.offset / den
        if s.t0 - 0.5 <= float(t * ray @ s.tangent) <= s.t1 + 0.5 and (best is None or t < best[0]):
            best = (t, s)
    if best is None:
        return None, "ruler_wall_not_found"
    seg = best[1]
    n_cam = cloud_from_cam.T @ np.array([seg.normal[0], 0.0, seg.normal[1]])  # wall normal in the camera frame
    cos = [float(n_cam @ r) for r in ends]
    if min(cos) < np.cos(np.radians(MAX_VIEW_DEG)):
        return None, "ruler_seen_too_obliquely"
    span = float(np.linalg.norm(ends[0] / cos[0] - ends[1] / cos[1]))  # ruler length per unit wall distance
    d_true = length / span  # true distance of the wall from the camera, metres
    mid_true = d_true * 0.5 * (ends[0] / cos[0] + ends[1] / cos[1])  # the ruler's centre in the camera frame
    if abs(float((cloud_from_cam @ mid_true)[1])) > MAX_HEIGHT_M:  # cloud +y is up, camera at the origin
        return None, "ruler_height_implausible"
    factor = d_true / seg.offset
    length_px = float(np.hypot(found[0][0] - found[1][0], found[0][1] - found[1][1]))
    theta = float(np.arccos(np.clip(np.mean(cos), -1, 1)))
    rel = float(np.sqrt((np.sqrt(2) * END_PX_SIGMA / length_px) ** 2 + (np.tan(theta) * NORMAL_SIGMA_RAD) ** 2
                        + PLANE_REL_SIGMA**2 + LENGTH_REL_SIGMA**2))
    return (factor, rel), None


def ruler_scale(
    photos: PhotoSet, poses: Poses, scene: RoomScene, segs: list[WallSegment], length: float = REFERENCE_LENGTH_M
) -> tuple[RulerScale | None, list[str]]:
    """(scale from the ruler in the room's fused photos or None, flags for views that could not be used)."""
    index = {im.name: i for i, im in enumerate(photos.images)}
    views, flags = [], []
    for name in scene.used:
        i = index[name]
        got, why = _one_view(photos.images[i].rgb, photos.images[i].f_px,
                             scene.rotation @ np.asarray(poses.rotations[i], float), segs, length)
        if got is not None:
            views.append(got)
        elif why != "ruler_not_seen":
            flags.append(f"{why}:{name}")
    views = [v for v in views if PLAUSIBLE_FACTOR[0] <= v[0] <= PLAUSIBLE_FACTOR[1]]
    if len(views) >= 3:  # with three views or more an outlier is a false detection: keep those near the median
        med = float(np.median([v[0] for v in views]))
        kept = [v for v in views if abs(np.log(v[0] / med)) <= np.log(1 + MAX_VIEW_DISAGREE)]
        if len(kept) < len(views):
            flags.append("ruler_view_outlier_dropped")
        views = kept
    if not views:
        return None, flags
    if len(views) == 2 and abs(np.log(views[0][0] / views[1][0])) > np.log(1 + MAX_VIEW_DISAGREE):
        return None, [*flags, "ruler_views_disagree"]  # two views that disagree: no way to tell which is the ruler
    f = np.array([v[0] for v in views])
    w = 1.0 / np.array([v[1] for v in views]) ** 2
    factor = float(np.exp(np.sum(w * np.log(f)) / w.sum()))
    rel = float(np.sqrt(1.0 / w.sum()))  # weighted mean of independent views
    if len(views) > 1:
        spread = float(np.std(np.log(f), ddof=1))
        rel = float(np.hypot(rel, spread / np.sqrt(len(views))))
    return RulerScale(factor, float(max(rel, FLOOR_REL_SIGMA)), len(views)), flags
