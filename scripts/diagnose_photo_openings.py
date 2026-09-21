"""Can the photo tier recover any doorway on the Hall (which physically has 3: to B1, B2, bathroom)?

Rebuilds the Hall room's scaled cloud the way build_photo_room does, then prints every fitted wall
segment with its dense runs and the gaps between them, and runs find_openings. This tells us whether
the stitch fails because doorways are undetectable (capture) or because detection misses them (code).
"""
import pathlib
import numpy as np
import pillow_heif

pillow_heif.register_heif_opener()

from floorfathom.io_photos import load_photo_set, discover_rooms
from floorfathom.photo_pose import run_photo_sfm, register_rotations
from floorfathom.photo_scene import build_scene_sfm, build_scene
from floorfathom.photo_reference import ruler_scale as ruler_v1
from floorfathom.photo_layout import wall_segments, outline_from_segments, find_openings, OPENING_WIDTH
from floorfathom.photo_pipeline import cached_depth, MIN_SFM_FRAC
from floorfathom.estimate import make_grid

FOLDER = "Data/Hall"
depth = cached_depth(pathlib.Path("out/photo/work/depth"))
rooms = discover_rooms(pathlib.Path(FOLDER))
photos = load_photo_set("Hall", list(rooms.values())[0])
work = pathlib.Path("out/hall_openings_diag")

sfm = run_photo_sfm(photos, work / "sfm", 0)
use_sfm = sfm is not None and sfm.registered_fraction >= MIN_SFM_FRAC
if use_sfm:
    scene = build_scene_sfm(photos, sfm, depth, 0, work)
    scale = None  # ruler path differs; use raw for structure inspection
else:
    poses = register_rotations(photos.images, seed=0)
    scene = build_scene(photos, poses, depth, 0)
    ruler, _ = ruler_v1(photos, poses, scene, wall_segments(scene.cloud.points, 0), 0.316)
    scale = ruler.factor if ruler else None

factor = scale or 1.0
pts = (scene.cloud.points * factor).astype(np.float32)
print(f"path={'SfM' if use_sfm else 'rotation-only'}  scale={'ruler' if scale else 'none(raw)'}  points={len(pts)}")

segs = wall_segments(pts, 0)
print(f"\nfitted wall segments: {len(segs)}  (doorway width window {OPENING_WIDTH[0]}-{OPENING_WIDTH[1]} m)")
for i, s in enumerate(segs):
    runs = s.runs
    gaps = [(round(b, 2), round(a, 2), round(a - b, 2)) for (_, b), (a, _) in zip(runs, runs[1:])]
    dw = [g for g in gaps if OPENING_WIDTH[0] <= g[2] <= OPENING_WIDTH[1]]
    print(f"  seg{i}: len={s.length:.2f} m, support={s.support}, {len(runs)} run(s); "
          f"gaps={gaps}  doorway-width gaps={dw}")

grid = make_grid(pts, scene.traj_xz * factor)
outline = outline_from_segments(segs, grid)
if outline is None:
    print("\noutline: None (fewer than 2 visible walls)")
else:
    open_edges = sum(1 for e in outline.edges if not e.supported)
    print(f"\noutline: {len(outline.edges)} edges, {open_edges} unsupported (open) — closed={open_edges == 0}")
    ops = find_openings(segs, outline, pts)
    print(f"find_openings: {len(ops)} doorway(s) detected")
    for o in ops:
        print(f"   opening centre={np.round(o.centre, 2)} on edge {o.edge}")
