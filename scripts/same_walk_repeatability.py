"""Estimator repeatability on one walk: two disjoint frame subsets of the same scan (no ground truth).

    uv run python scripts/same_walk_repeatability.py SCAN_FOLDER [--drift]

The scan's frames are split into two interleaved subsets (even and odd picks of the usual
subsampling), each becomes a cloud and a plan, and the two plans are compared in the same frame,
so coverage is identical and any disagreement is the estimator's own. It says nothing about
device noise or about two separate walks (see cross_scan_repeatability.py for those, where part of
the difference is real). Reported: room areas, wall line positions (long walls that overlap),
wall corner positions along the wall (both neighbours supported) versus wall ends next to an
unsupported edge (a doorway or an unseen stretch), and the length of every long wall that lies on
the same line in both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from floorfathom.drift import correct_points, correct_trajectory, estimate_drift
from floorfathom.estimate import estimate, make_grid
from floorfathom.io_lidar import load_scan
from floorfathom.planes import find_floor
from floorfathom.points import Cloud, frame_points, voxel_downsample

N_CHUNKS, TARGET_FRAMES, MIN_WALL = 20, 800, 1.0


def run(scan, offset: int, correct_drift: bool):
    n = len(scan)
    rows = np.arange(offset, n, max(1, n // TARGET_FRAMES))
    chunk_of_row = np.minimum(np.arange(len(rows)) * N_CHUNKS // len(rows), N_CHUNKS - 1)
    pts, chunks = [], []
    for c in range(N_CHUNKS):
        sel = rows[chunk_of_row == c]
        if len(sel):
            p = voxel_downsample(np.concatenate([frame_points(scan, int(i)) for i in sel]), 0.02)
            pts.append(p)
            chunks.append(np.full(len(p), c, dtype=np.int16))
    cloud, traj = Cloud(np.concatenate(pts), np.concatenate(chunks), N_CHUNKS), scan.positions[:, [0, 2]]
    if correct_drift:
        drift = estimate_drift(cloud, find_floor(cloud.points[:, 1]).y)
        cloud, traj = correct_points(cloud, drift), correct_trajectory(traj, drift)
    return estimate(cloud.points, traj, grid=make_grid(cloud.points, traj))


def _mask(poly: np.ndarray, size: int = 1200, cell: float = 0.05) -> np.ndarray:
    import cv2

    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.floor(poly / cell + size // 2).astype(np.int32)], 1)
    return m.astype(bool)


def _stat(name: str, v: list[float]) -> None:
    a = np.abs(np.asarray(v))
    print(f"{name}: n={len(a)}, median {np.median(a):.1f} cm, within 1 cm {np.mean(a <= 1):.0%}, "
          f"within 3 cm {np.mean(a <= 3):.0%}, within 10 cm {np.mean(a <= 10):.0%}" if len(a) else f"{name}: none")


def main(path: Path, correct_drift: bool) -> None:
    scan = load_scan(path)
    ea, eb = run(scan, 0, correct_drift), run(scan, max(1, len(scan) // TARGET_FRAMES) // 2, correct_drift)
    print(f"{path.name}, drift correction {'on' if correct_drift else 'off'}: {len(ea.rooms)} and {len(eb.rooms)} rooms\n")
    print("| room in B | matched room in A | IoU | area B (m2) | area A (m2) | area difference |")
    print("|---|---|---|---|---|---|")
    pos, corner, opened, length, lengths = [], [], [], [], []
    for kb, rb in enumerate(eb.rooms):
        mb = _mask(rb.outline.polygon)
        iou, ka = max((float((mb & (ma := _mask(ra.outline.polygon))).sum() / max((mb | ma).sum(), 1)), k) for k, ra in enumerate(ea.rooms))
        ra = ea.rooms[ka]
        print(f"| room_{kb} | room_{ka} | {iou:.2f} | {rb.area:.2f} | {ra.area:.2f} | {(rb.area / ra.area - 1) * 100:+.1f}% |")
        E, EA = rb.outline.edges, ra.outline.edges
        for i, e in enumerate(E):
            if not e.supported or e.length < MIN_WALL:
                continue
            t = (e.p1 - e.p0) / e.length
            n = np.array([-t[1], t[0]])
            partners = [(j, a) for j, a in enumerate(EA) if a.supported and a.length >= MIN_WALL and abs(((a.p1 - a.p0) / a.length) @ t) >= np.cos(np.radians(3))]
            best = None
            for j, a in partners:
                lo, hi = sorted([(a.p0 - e.p0) @ t, (a.p1 - e.p0) @ t])
                ov = min(hi, e.length) - max(lo, 0.0)
                if ov >= 0.5 * min(a.length, e.length):
                    mid = e.p0 + t * (max(lo, 0.0) + min(hi, e.length)) / 2
                    off = abs(n @ (a.p0 + ((a.p1 - a.p0) / a.length) * ((mid - a.p0) @ ((a.p1 - a.p0) / a.length)) - mid))
                    if off < 0.25 and (best is None or off < best[0]):
                        best = (off, j, a)
            if best is None:
                continue
            off, j, a = best
            pos.append(off * 100)
            if off <= 0.03:
                length.append((e.length - a.length) * 100)
                lengths.append(a.length)
                for pe, pa, ne, na in ((e.p0, a.p0, E[i - 1], EA[j - 1]), (e.p1, a.p1, E[(i + 1) % len(E)], EA[(j + 1) % len(EA)])):
                    d = (pa - pe) @ t * 100
                    if abs(d) < 50:
                        (corner if ne.supported and na.supported else opened).append(d)
    print()
    _stat("wall line position difference", pos)
    _stat("wall end at a corner (both neighbours supported)", corner)
    _stat("wall end next to an unsupported edge (doorway or unseen stretch)", opened)
    _stat("wall length difference (walls on the same line)", length)
    ok = [abs(d) <= max(1.0, 0.5 * L) for d, L in zip(length, lengths)]  # 0.5% of an L m wall is 0.5 * L cm
    print(f"walls within the repeatability gate (1 cm or 0.5%): {np.mean(ok):.0%} of {len(ok)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--drift"]
    if len(args) != 1:
        raise SystemExit(__doc__)
    main(Path(args[0]), "--drift" in sys.argv)
