"""Audit the rotation registration of each photo room (read-only, numbers only, no fixes).

    uv run python scripts/audit_photo_registration.py [ROOM ...] [--data Data] [--cache out/depth_cache]

For every registered pair of photos it prints the RANSAC inlier count (``-`` when the pair was not a
link), the rotation between the two cameras that the registration implies, and the dense depth check:
every photo is taken from one spot, so a ray seen by both photos has the same length in both, and the
log ratio of the two depth maps over the overlap is constant when the rotation is right. The median
is the pair's relative depth scale and the spread (robust sigma) is large when the two photos are not
looking at the same surface, that is, when the rotation, and so the link, is wrong. Pairs are sorted
by spread, worst first, then each photo gets its mean spread over the pairs it is in. Depth comes
only from the cache (no model runs); a photo with no cached depth is listed and left out of the check.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from floorfathom.io_photos import discover_rooms, load_photo_set
from floorfathom.photo_pose import register_rotations
from floorfathom.photo_scene import _pair_log_ratio

LONG_SIDE = 924  # the depth model's input size in the cache file names


def cached_depth(cache: Path, rgb: np.ndarray) -> np.ndarray | None:
    f = cache / f"{hashlib.sha256(rgb.tobytes()).hexdigest()[:16]}_{LONG_SIDE}.npy"
    return np.load(f) if f.exists() else None


def rotation_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def audit(room: str, paths: list[Path], cache: Path) -> None:
    photos = load_photo_set(room, paths)
    poses = register_rotations(photos.images)
    names = [Path(im.name).stem for im in photos.images]
    ok = [i for i, r in enumerate(poses.rotations) if r is not None]
    print(f"\n## {room}: {len(names)} photos, {len(ok)} registered, {len(poses.pair_inliers)} links, "
          f"root {names[poses.root] if poses.root is not None else None}, "
          f"loop error {poses.loop_error_deg if poses.loop_error_deg is None else round(poses.loop_error_deg, 1)} deg, flags {poses.flags or 'none'}")
    depth = {i: cached_depth(cache, photos.images[i].rgb) for i in ok}
    missing = [names[i] for i in ok if depth[i] is None]
    if missing:
        print(f"no cached depth for {', '.join(missing)}: left out of the depth check")
    rows = []
    for a, i in enumerate(ok):
        for j in ok[a + 1 :]:
            inl = poses.pair_inliers.get((i, j))
            rij = poses.rotations[j].T @ poses.rotations[i]
            got = None
            if depth[i] is not None and depth[j] is not None:
                zi, zj = depth[i], depth[j]
                fi = photos.images[i].f_px * zi.shape[1] / photos.images[i].rgb.shape[1]
                fj = photos.images[j].f_px * zj.shape[1] / photos.images[j].rgb.shape[1]
                got = _pair_log_ratio(zi, fi, zj, fj, rij)
            rows.append((i, j, inl, rotation_deg(rij), got))
    rows.sort(key=lambda r: (r[4] is None, -(r[4][1] if r[4] else 0.0)))
    print("\n| pair | inliers | camera rotation (deg) | depth log-ratio median (scale) | spread |")
    print("|---|---|---|---|---|")
    for i, j, inl, deg, got in rows:
        d = "no overlap or no depth | " if got is None else f"{got[0]:+.3f} (x{np.exp(got[0]):.2f}) | {got[1]:.3f}"
        print(f"| {names[i]}-{names[j]} | {'-' if inl is None else inl} | {deg:.0f} | {d} |")
    print("\n| photo | links | pairs with overlap | mean spread | worst spread |")
    print("|---|---|---|---|---|")
    for i in ok:
        mine = [r for r in rows if i in (r[0], r[1]) and r[4] is not None]
        links = sum(1 for k in poses.pair_inliers if i in k)
        sp = [r[4][1] for r in mine]
        print(f"| {names[i]} | {links} | {len(mine)} | {np.mean(sp):.3f} | {np.max(sp):.3f} |" if sp
              else f"| {names[i]} | {links} | 0 | | |")
    for i in range(len(names)):
        if i not in ok:
            print(f"| {names[i]} | unregistered | | | |")


def main(argv: list[str]) -> None:
    opts = {"--data": "Data", "--cache": "out/depth_cache"}
    rest = []
    it = iter(argv)
    for a in it:
        if a in opts:
            opts[a] = next(it)
        else:
            rest.append(a)
    rooms = discover_rooms(Path(opts["--data"]))
    chosen = [r for r in rooms if not rest or r.lower() in {x.lower() for x in rest}]
    if not chosen:
        raise SystemExit(f"no such room; rooms are {sorted(rooms)}")
    for room in chosen:
        audit(room, rooms[room], Path(opts["--cache"]))


if __name__ == "__main__":
    main(sys.argv[1:])
