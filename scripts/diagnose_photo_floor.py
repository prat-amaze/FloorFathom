"""Does the photo tier suffer the same glossy-floor reflection as the video tier?

Builds one room's gravity-levelled cloud (rotation-only path — the densest, most complete cloud),
prints a coarse height histogram, and shows where each floor/ceiling estimator lands. A reflection
shows up as a second cluster of points below the true floor (camera sits at y=0, so a hand-held
phone's floor is ~-1.4 m and ceiling ~+1.4 m).
"""
import sys
import pathlib

import numpy as np
import pillow_heif

pillow_heif.register_heif_opener()

from floorfathom.io_photos import load_photo_set, discover_rooms
from floorfathom.photo_pose import register_rotations
from floorfathom.photo_scene import build_scene
from floorfathom.photo_layout import floor_and_ceiling
from floorfathom.planes import find_floor
from floorfathom.photo_pipeline import cached_depth


def diagnose(folder: str):
    name = folder.rstrip("/").split("/")[-1]
    rooms = discover_rooms(pathlib.Path(folder))
    photos = load_photo_set(name, list(rooms.values())[0])
    depth = cached_depth(pathlib.Path("out") / f"{name}_diag" / "depth")
    poses = register_rotations(photos.images, seed=0)
    scene = build_scene(photos, poses, depth, seed=0)
    if scene is None:
        print(f"{folder}: build_scene returned None")
        return
    y = scene.cloud.points[:, 1]
    print(f"\n=== {folder}: {len(y)} cloud points (camera at y=0, +y up) ===")

    # coarse height histogram in 20 cm bins
    lo, hi = float(y.min()), float(y.max())
    edges = np.arange(np.floor(lo * 5) / 5, hi + 0.2, 0.2)
    hist, _ = np.histogram(y, bins=edges)
    peak = hist.max()
    print(f"height range {lo:+.2f} .. {hi:+.2f} m")
    print("height histogram (20 cm bins, bar length ~ point count):")
    for k in range(len(hist)):
        if hist[k] == 0:
            continue
        h0 = edges[k]
        bar = "#" * int(60 * hist[k] / peak)
        mark = ""
        if -0.1 <= h0 <= 0.1:
            mark = " <- camera height (y=0)"
        print(f"  {h0:+5.2f} m | {hist[k]:6d} {bar}{mark}")

    # where each estimator lands
    ff = find_floor(y)
    fl, ce = floor_and_ceiling(scene.cloud.points, seed=0)
    print()
    print(f"planes.find_floor         -> {ff.y:+.2f} m (camera height {-ff.y:.2f} m)" if ff else "planes.find_floor -> None")
    print(f"layout.floor_and_ceiling  -> floor {fl.y:+.2f} m, ceiling {ce.y if ce else None}")
    if fl and ce:
        print(f"   => ceiling height = {ce.y - fl.y:.2f} m")
    print(f"scene.floor_y (used)      -> {scene.floor_y:+.2f} m  flags={scene.flags}")
    # A hand-held phone floor is ~-1.4 m. Anything near -2.5..-3 m is the reflection.
    if ff and -ff.y > 2.1:
        print("   !! floor is >2.1 m below camera: REFLECTION LOCK likely (true floor ~1.4 m)")
    # reflection check: points below the found floor (the mirror image lives here)
    if ff:
        below = int((y < ff.y - 0.1).sum())
        print(f"   points below the found floor: {below} ({100*below/len(y):.1f}% - reflection tail if large)")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["Data/Hall"]:
        diagnose(f)
