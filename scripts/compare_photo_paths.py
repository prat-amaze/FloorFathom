"""Head-to-head: SfM path vs rotation-only path on the same room, so the SfM regression is measurable.

Prints walls, area, ceiling for each path. Mirrors how the video tier measured its fix.
"""
import sys
import pathlib
import tempfile

import numpy as np
import pillow_heif

pillow_heif.register_heif_opener()

from floorfathom.io_photos import load_photo_set, discover_rooms
from floorfathom.photo_pose import register_rotations, run_photo_sfm
from floorfathom.photo_scene import build_scene, build_scene_sfm
from floorfathom.photo_layout import wall_segments, outline_from_segments, floor_and_ceiling
from floorfathom.photo_pipeline import cached_depth
from floorfathom.estimate import make_grid
from floorfathom import layout as L


def summarise(tag, scene):
    if scene is None:
        print(f"  {tag}: scene is None")
        return
    pts = scene.cloud.points
    grid = make_grid(pts, scene.traj_xz)
    segs = wall_segments(pts, seed=0)
    outline = outline_from_segments(segs, grid)
    fl, ce = floor_and_ceiling(pts, seed=0)
    n_walls = len(outline.edges) if outline else 0
    n_sup = sum(1 for e in outline.edges if e.supported) if outline else 0
    area = abs(L.signed_area(outline.polygon)) if outline else None
    ch = (ce.y - fl.y) if (fl and ce) else None
    print(f"  {tag}: points={len(pts):6d} used={len(scene.used)} walls={n_walls} (supported {n_sup}) "
          f"area={area if area is None else round(area,1)} ceiling={ch if ch is None else round(ch,2)}")


def compare(folder: str):
    name = folder.rstrip("/").split("/")[-1]
    rooms = discover_rooms(pathlib.Path(folder))
    photos = load_photo_set(name, list(rooms.values())[0])
    depth = cached_depth(pathlib.Path("out") / f"{name}_diag" / "depth")
    print(f"\n=== {folder}: {len(photos.images)} images ===")

    sfm = run_photo_sfm(photos, pathlib.Path(tempfile.mkdtemp()), seed=0)
    if sfm is not None:
        frac = sfm.registered_fraction
        print(f"  SfM: {sfm.registered.sum()}/{len(photos.images)} registered ({frac:.0%}), flags={sfm.flags}")
        summarise("SfM path        ", build_scene_sfm(photos, sfm, depth, seed=0, work=pathlib.Path(tempfile.mkdtemp())))
    else:
        print("  SfM: None")

    poses = register_rotations(photos.images, seed=0)
    summarise("rotation-only   ", build_scene(photos, poses, depth, seed=0))


if __name__ == "__main__":
    for f in sys.argv[1:] or ["Data/Hall"]:
        compare(f)
