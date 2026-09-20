"""From the arbitrary SfM frame to a gravity-aligned frame (world +y up, as in ``points.py``).

The .MOV files carry no gravity sensor data, so "up" is recovered from the geometry with two
independent cues:

1. The phone is held upright in portrait, so the image-up direction of the cameras,
   averaged over the walk, points roughly at world up. This also settles the sign.
2. Floors, ceilings and tabletops are horizontal planes, so along the true up axis the
   points pile into sharp layers. Tilting the axis away from the truth smears the layers.
   The tilt (within ``SEARCH_DEG`` of cue 1) that maximises the layering is kept.

The angle between the two cues is a sanity check: when they disagree, or the search runs
into its limit, the alignment is flagged as uncertain rather than trusted.

Limitation: cue 2 cannot tell up from down (a height histogram looks the same upside
down), so the sign rests on cue 1 alone. A capture made with the phone upside down would be
aligned with the floor on top and nothing here would notice. The capture protocol requires
the phone to be held upright in portrait.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sfm import SfmResult

SEARCH_DEG = 16.0
AGREE_DEG = 20.0


@dataclass
class Gravity:
    up: np.ndarray  # (3,) unit vector, in the SfM frame
    rotation: np.ndarray  # (3, 3): gravity_aligned = rotation @ sfm_point, world +y up
    camera_up: np.ndarray  # (3,) cue 1 alone
    tilt_from_camera_deg: float  # angle between the two cues
    layering: float  # layering score at ``up`` (Simpson index of the height histogram)
    layering_at_camera_up: float  # the same score at cue 1, to show what the search gained
    flags: list[str] = field(default_factory=list)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` to unit vector ``b`` (Rodrigues)."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def layering_score(points: np.ndarray, up: np.ndarray, bin_width: float) -> float:
    """Chance that two random points fall in the same height bin along ``up``."""
    h = points @ up
    counts = np.bincount(np.floor((h - h.min()) / bin_width).astype(np.int64))
    p = counts / counts.sum()
    return float((p * p).sum())


def estimate_gravity(sfm: SfmResult) -> Gravity:
    pts = sfm.points
    axes_y = sfm.cam_to_world[sfm.registered][:, :, 1]  # camera y (down) in the SfM frame
    camera_up = _unit(-axes_y.mean(axis=0))

    # bin width is fixed once, from the vertical extent along cue 1 (units are arbitrary here)
    h0 = pts @ camera_up
    lo, hi = np.percentile(h0, [1, 99])
    bin_width = (hi - lo) / 100.0

    e1 = _unit(np.cross(camera_up, [1.0, 0.0, 0.0] if abs(camera_up[0]) < 0.9 else [0.0, 1.0, 0.0]))
    e2 = np.cross(camera_up, e1)

    def candidate(a: float, b: float) -> np.ndarray:
        return _unit(camera_up + np.tan(np.radians(a)) * e1 + np.tan(np.radians(b)) * e2)

    def search(centre: tuple[float, float], radius: float, step: float):
        best = (-1.0, centre)
        for a in np.arange(centre[0] - radius, centre[0] + radius + 1e-9, step):
            for b in np.arange(centre[1] - radius, centre[1] + radius + 1e-9, step):
                if a * a + b * b > SEARCH_DEG**2 + 1e-9:
                    continue
                s = layering_score(pts, candidate(a, b), bin_width)
                if s > best[0]:
                    best = (s, (float(a), float(b)))
        return best

    coarse_score, coarse = search((0.0, 0.0), SEARCH_DEG, 2.0)
    score, (a, b) = search(coarse, 2.0, 0.5)
    up = candidate(a, b)
    tilt = float(np.degrees(np.arccos(np.clip(up @ camera_up, -1.0, 1.0))))

    flags: list[str] = []
    if tilt > AGREE_DEG:
        flags.append("gravity_uncertain")
    elif np.hypot(a, b) >= SEARCH_DEG - 0.6:
        flags.append("gravity_uncertain")  # the best tilt sits on the edge of the search area
    return Gravity(
        up=up,
        rotation=rotation_between(up, np.array([0.0, 1.0, 0.0])),
        camera_up=camera_up,
        tilt_from_camera_deg=tilt,
        layering=score,
        layering_at_camera_up=layering_score(pts, camera_up, bin_width),
        flags=flags,
    )
