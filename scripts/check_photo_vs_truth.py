"""Run the photo tier's stages on the real stills of each room and print them next to the tape measure.

Registration, metric depth, fusion into a levelled cloud, then the shared estimator. Depth is the slow
step (seconds per image on CPU), so it is cached per image under ``--cache`` and reused; a cache hit
gives exactly the depth the model would give again, the model is deterministic.

    uv run python scripts/check_photo_vs_truth.py [--rooms b2 BR] [--long-side 924] [--out out/photo_check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from floorfathom.depth import DepthEstimator
from floorfathom.estimate import Params, estimate
from floorfathom.io_photos import discover_rooms, load_photo_set
from floorfathom.layout import BAND_HI, BAND_LO
from floorfathom.photo_pose import register_rotations
from floorfathom.photo_scene import build_scene

ROOT = Path(__file__).resolve().parents[1]
TRUTH_ROOM = {"b1": "bedroom_b1", "b2": "master_bedroom_b2", "BR": "bathroom", "Hall": "hall_kitchen"}


def truth_for(room: str) -> dict:
    t = json.loads((ROOT / "ground_truth.json").read_text())["rooms"][TRUTH_ROOM[room]]
    if "walls" in t:
        lengths = sorted(w["length"] / 100 for w in t["walls"])
    else:
        lengths = sorted([t["length"] / 100] * 2 + [t["width"] / 100] * 2)
    return {"ceiling": t["ceiling_height"] / 100, "walls": lengths}


class CachedDepth:
    """Depth model with a disk cache keyed on the pixels and the input size."""

    def __init__(self, cache: Path, long_side: int):
        self.cache, self.long_side, self.model = cache, long_side, None
        cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        f = self.cache / f"{hashlib.sha256(rgb.tobytes()).hexdigest()[:16]}_{self.long_side}.npy"
        if f.exists():
            return np.load(f)
        self.model = self.model or DepthEstimator(threads=1)
        z = self.model(rgb, long_side=self.long_side)
        np.save(f, z)
        return z


def plot(room: str, scene, est, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    p = scene.cloud.points
    h = p[:, 1] - scene.floor_y
    band = p[(h > BAND_LO) & (h < BAND_HI)]
    ax.scatter(band[:, 0], band[:, 2], s=0.3, c="0.55", label="wall band")
    ax.scatter(scene.traj_xz[:, 0], scene.traj_xz[:, 1], s=1, c="tab:blue", label="fan")
    for r in est.rooms:
        poly = np.vstack([r.outline.polygon, r.outline.polygon[:1]])
        ax.plot(poly[:, 0], poly[:, 1], "r-", lw=1.5)
    ax.plot(0, 0, "k^")
    ax.set_aspect("equal")
    ax.set_title(f"{room}: {len(scene.used)} photos fused")
    ax.legend(loc="upper right", markerscale=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="*", default=list(TRUTH_ROOM))
    ap.add_argument("--long-side", type=int, default=924)
    ap.add_argument("--data", type=Path, default=ROOT / "Data")
    ap.add_argument("--cache", type=Path, default=ROOT / "out" / "depth_cache")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "photo_check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    found = discover_rooms(args.data)
    cache = CachedDepth(args.cache, args.long_side)
    for room in args.rooms:
        t0 = time.perf_counter()
        photos = load_photo_set(room, found[room])
        poses = register_rotations(photos.images)
        scene = build_scene(photos, poses, cache)
        truth = truth_for(room)
        n_reg = sum(r is not None for r in poses.rotations)
        print(f"\n== {room}: {n_reg}/{len(photos.images)} photos registered, {time.perf_counter() - t0:.0f} s")
        print(f"   flags: {photos.flags + poses.flags}")
        if scene is None:
            print("   no scene (no floor plane, or too few points)")
            continue
        print(f"   camera height {-scene.floor_y:.2f} m, tilt {scene.tilt_deg:.1f} deg, floor support {scene.floor_support:.2f},"
              f" {len(scene.cloud)} points, scene flags {scene.flags}")
        print(f"   depth scale per photo {np.round(scene.scales, 2).tolist()} (spread {scene.scale_spread:.2f})")
        est = estimate(scene.cloud.points, scene.traj_xz, Params())
        print(f"   rooms found: {len(est.rooms)}")
        for r in est.rooms:
            ch = r.ceiling_height
            ce = "none" if ch is None else f"{ch:.2f} m ({(ch / truth['ceiling'] - 1) * 100:+.1f}% vs {truth['ceiling']:.2f})"
            lengths = sorted(e.length for e in r.outline.edges)
            print(f"   - area {r.area:.2f} m2, ceiling {ce}, {len(r.outline.edges)} walls, {len(r.openings)} openings, flags {r.flags}")
            print(f"     wall lengths {[round(x, 2) for x in lengths]}")
            print(f"     tape        {truth['walls']}")
        plot(room, scene, est, args.out / f"{room}.png")


if __name__ == "__main__":
    main()
