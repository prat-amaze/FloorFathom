"""Repeatability across two captures of the same property (real data, no ground truth).

    uv run python scripts/cross_scan_repeatability.py CAPTURE_A CAPTURE_B

Each capture has its own world frame, so the wall maps are first aligned rigidly
(rotation by brute force, translation by FFT cross-correlation). Rooms are then matched
by overlap and compared: area, and the length of every long wall that lies on the same
line in both. This does not measure accuracy, only how well two walks agree, and part of
any disagreement is real (the two walks did not see the same things).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import fftconvolve

from floorfathom.estimate import estimate, make_grid
from floorfathom.io_lidar import load_scan
from floorfathom.points import build_cloud

CELL = 0.05
SIZE = 700  # alignment canvas, 35 m


def _run(path: Path):
    scan = load_scan(path)
    cloud = build_cloud(scan, target_frames=800, n_chunks=20)
    traj = scan.positions[:, [0, 2]]
    return estimate(cloud.points, traj, grid=make_grid(cloud.points, traj))


def _wall_points(est) -> np.ndarray:
    rows, cols = np.nonzero(est.barrier)
    return est.grid.to_xz(cols, rows)


def _raster(pts: np.ndarray, origin: np.ndarray) -> np.ndarray:
    ij = np.floor((pts - origin) / CELL).astype(int)
    ok = ((ij >= 0) & (ij < SIZE)).all(axis=1)
    m = np.zeros((SIZE, SIZE), np.float32)
    m[ij[ok, 1], ij[ok, 0]] = 1
    return m


def align(pts_a: np.ndarray, pts_b: np.ndarray):
    """Rigid transform taking B's wall points onto A's: returns a function on (N, 2) arrays."""
    origin = pts_a.mean(axis=0) - SIZE * CELL / 2
    map_a = cv2.dilate(_raster(pts_a, origin), np.ones((3, 3), np.uint8))
    cb, ca = pts_b.mean(axis=0), pts_a.mean(axis=0)
    best = (-1.0, 0.0, np.zeros(2))
    for deg in np.arange(0.0, 360.0, 1.0):
        a = np.radians(deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        map_b = _raster((pts_b - cb) @ rot.T + ca, origin)
        corr = fftconvolve(map_a, map_b[::-1, ::-1], mode="same")
        j = np.unravel_index(corr.argmax(), corr.shape)
        if corr[j] > best[0]:
            best = (float(corr[j]), deg, (np.array(j)[::-1] - SIZE // 2) * CELL)
    score, deg, shift = best
    a = np.radians(deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def overlap(sign: int) -> float:
        return float((map_a * _raster((pts_b - cb) @ rot.T + ca + sign * shift, origin)).sum())

    sign = 1 if overlap(1) >= overlap(-1) else -1
    n_b = float(_raster((pts_b - cb) @ rot.T + ca + sign * shift, origin).sum())
    return (lambda p: (p - cb) @ rot.T + ca + sign * shift), deg, overlap(sign) / max(n_b, 1.0), origin


def _poly_mask(poly: np.ndarray, origin: np.ndarray) -> np.ndarray:
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m, [np.floor((poly - origin) / CELL).astype(np.int32)], 1)
    return m.astype(bool)


def main(a_path: Path, b_path: Path) -> None:
    est_a, est_b = _run(a_path), _run(b_path)
    tf, deg, frac, origin = align(_wall_points(est_a), _wall_points(est_b))
    print(f"aligned {b_path.name} onto {a_path.name}: rotation {deg:.0f} deg, {frac:.0%} of its wall cells overlap\n")
    print(f"| room in {b_path.name} | matched room in {a_path.name} | IoU | area B (m2) | area A (m2) | area diff |")
    print("|---|---|---|---|---|---|")
    walls = []
    for kb, rb in enumerate(est_b.rooms):
        mb = _poly_mask(tf(rb.outline.polygon), origin)
        scores = [
            (float((mb & (ma := _poly_mask(ra.outline.polygon, origin))).sum() / max((mb | ma).sum(), 1)), ka)
            for ka, ra in enumerate(est_a.rooms)
        ]
        iou, ka = max(scores)
        ra = est_a.rooms[ka]
        print(f"| room_{kb} | room_{ka} | {iou:.2f} | {rb.area:.2f} | {ra.area:.2f} | {rb.area - ra.area:+.2f} ({(rb.area - ra.area) / ra.area:+.0%}) |")
        if iou < 0.5:
            continue
        for eb in rb.outline.edges:
            p0, p1 = tf(eb.p0[None])[0], tf(eb.p1[None])[0]
            tb = (p1 - p0) / np.linalg.norm(p1 - p0)
            nb = np.array([-tb[1], tb[0]])
            cands = []
            for ea in ra.outline.edges:
                ta = (ea.p1 - ea.p0) / ea.length
                off = abs(nb @ (0.5 * (ea.p0 + ea.p1) - p0))
                if abs(ta @ tb) >= 0.98 and off <= 0.25:
                    cands.append((off, ea.length))
            if cands and eb.length > 1.0:
                walls.append((eb.length, min(cands)[1]))
    d = np.array([b - a for b, a in walls])
    print(f"\nwalls > 1 m that lie on the same line in both captures: {len(walls)}")
    print(f"length difference: median |d| = {np.median(np.abs(d)) * 100:.0f} cm, "
          f"within 5 cm: {np.mean(np.abs(d) <= 0.05):.0%}, within 1 cm or 0.5%: "
          f"{np.mean([abs(b - a) <= max(0.01, 0.005 * a) for b, a in walls]):.0%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
