"""Drift correction on a walked-around room with drift injected by hand (truth is known)."""

import numpy as np

from floorfathom.drift import correct_points, estimate_drift
from floorfathom.points import Cloud

W, D, K = 6.0, 5.0, 8
YAW_DEG_PER_CHUNK = 0.1


def _room_walls(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Wall points of a W x D room as (x, height above floor, z), and each point's distance
    s along the perimeter."""
    per = 2 * (W + D)
    s0 = np.arange(0.0, per, 0.02)
    heights = np.arange(0.3, 2.0, 0.05)
    x = np.select([s0 < W, s0 < W + D, s0 < 2 * W + D], [s0, W, 2 * W + D - s0], 0.0)
    z = np.select([s0 < W, s0 < W + D, s0 < 2 * W + D], [0.0, s0 - W, D], per - s0)
    n = len(heights)
    pts = np.column_stack([np.repeat(x, n), np.tile(heights, len(s0)), np.repeat(z, n)])
    return pts + rng.normal(0.0, 0.004, pts.shape), np.repeat(s0, n)


def _walk(rng, drift_per_chunk):
    """A cloud of K chunks; each chunk sees 3/8 of the perimeter, chunk c starting at c/K of it."""
    pts, s = _room_walls(rng)
    per = 2 * (W + D)
    truth, chunks, moved = [], [], []
    centre = np.array([W / 2, D / 2])
    for c in range(K):
        start = (c / K - 1 / 16) * per
        seen = ((s - start) % per) < 3 / 8 * per
        p = pts[seen]
        truth.append(p)
        chunks.append(np.full(len(p), c, np.int16))
        dx, dz = drift_per_chunk * c * np.array([1.0, -0.75])
        yaw = np.radians(YAW_DEG_PER_CHUNK) * c * (drift_per_chunk > 0)
        rel = p[:, [0, 2]] - centre
        cs, sn = np.cos(yaw), np.sin(yaw)
        q = np.stack([cs * rel[:, 0] - sn * rel[:, 1], sn * rel[:, 0] + cs * rel[:, 1]], axis=1) + centre + [dx, dz]
        m = p.copy()
        m[:, 0], m[:, 2] = q[:, 0], q[:, 1]
        moved.append(m)
    return (
        Cloud(np.concatenate(moved).astype(np.float32), np.concatenate(chunks), K),
        np.concatenate(truth),
    )


def _wall_error(points: np.ndarray) -> float:
    """Largest distance of a wall point from the true wall line (the error a plan shows).

    Sliding a wall along itself is invisible in a plan and not observable from wall
    geometry, so the along-wall component is not part of the error.
    """
    x, z = points[:, 0], points[:, 2]
    return float(np.minimum.reduce([np.abs(x), np.abs(x - W), np.abs(z), np.abs(z - D)]).max())


def test_recovers_injected_drift():
    cloud, _ = _walk(np.random.default_rng(1), 0.02)  # chunk 7 is off by 14 cm and 0.7 degrees
    drift = estimate_drift(cloud, floor_y=0.0)
    assert _wall_error(cloud.points) > 0.15
    assert _wall_error(correct_points(cloud, drift).points) < 0.02
    assert drift.chi2_after < 0.05 * drift.chi2_before


def test_no_drift_no_correction():
    cloud, _ = _walk(np.random.default_rng(2), 0.0)
    drift = estimate_drift(cloud, floor_y=0.0)
    assert drift.max_shift < 0.005
    assert drift.max_yaw_deg < 0.05
