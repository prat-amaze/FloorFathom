"""Compare room estimators on a finished video run's caches, without SfM, depth or bootstrap.

    uv run python scripts/eval_video_estimator.py out/h1 hall_kitchen [--variants points,rays] [--png DIR] [param=value ...]

Reads ``work/sfm.pkl``, ``dense.pkl``, ``alignment.json`` and ``depth924/`` of a run made by
``floorfathom plan`` (so rotation, scale and depth maps are the run's own), runs the room estimator with
each variant of free-space evidence and scores the largest room against the tape in ``ground_truth.json``:
wall lengths (one plan edge per tape wall, nearest length), area, ceiling. A dev tool: the user command
is ``floorfathom plan``. Variants: ``points`` (the LiDAR-tier estimator on the cloud alone) and ``rays``
(the same plus camera-ray free space, ``video_rays.py``), ``walls`` (cut behind fitted wall lines,
``video_walls.py``) and ``rays+walls``.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_video import assign, truth_room  # noqa: E402

from floorfathom.damage_video import cached_depth  # noqa: E402
from floorfathom.estimate import Params, estimate, make_grid  # noqa: E402
from floorfathom.planes import find_floor  # noqa: E402
from floorfathom.points_video import to_cloud  # noqa: E402
from floorfathom.video_walls import wall_finder  # noqa: E402
from floorfathom.video_rays import build_rays, carve, seen_from_votes  # noqa: E402


def load(run: Path):
    work = run / "work"
    sfm = pickle.loads((work / "sfm.pkl").read_bytes())[1]
    dense = pickle.loads((work / "dense.pkl").read_bytes())[1]
    al = json.loads((work / "alignment.json").read_text())
    rotation, scale = np.array(al["rotation"]), float(al["scale"])
    names = sorted(p.name for p in (work / "frames").glob("*.jpg"))
    kf = SimpleNamespace(directory=work / "frames", names=names)

    def never(_rgb):
        raise RuntimeError("depth map missing from the cache: run floorfathom plan first")

    depth = cached_depth(never, work / "depth924")
    return sfm, dense, rotation, scale, kf, depth


def rays_for(run: Path, sfm, dense, kf, depth) -> object:
    f = run / "work" / "rays.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())[1]  # (stamp, RayBundle), as run_clip's cache writes it
    b = build_rays(kf, sfm, dense, depth)
    f.write_bytes(pickle.dumps(("dev", b)))
    return b


def score(room, truth: dict) -> str:
    lengths = [e.length * 100 for e in room.outline.edges]
    tape = [v for _, v in truth["walls"]]
    got = dict(assign(tape, lengths))
    cells = []
    ok = 0
    for i, t in enumerate(tape):
        if i in got:
            err = lengths[got[i]] / t - 1
            ok += abs(err) <= 0.03
            cells.append(f"{lengths[got[i]]:.0f}({err * 100:+.0f}%)")
        else:
            cells.append("none")
    c = room.ceiling_height
    ceil = "none" if c is None else f"{c * 100:.1f} ({c * 100 - truth['ceiling']:+.1f} cm)"
    return (
        f"area {room.area:.2f} m2 | edges {len(lengths)} | walls {ok}/{len(tape)} within 3%: {' '.join(cells)} | "
        f"ceiling {ceil} | doors {len(room.openings)}: {[round(o.width * 100) for o in room.openings]}"
    )


def draw(est, room, path: Path, traj: np.ndarray) -> None:
    g = est.grid
    img = np.full((g.nz, g.nx, 3), 255, np.uint8)
    if est.free is not None:
        img[est.free] = (235, 225, 200)
    if est.barrier is not None:
        img[est.barrier] = (60, 60, 60)
    img[room.outline.mask & ~(est.barrier if est.barrier is not None else False)] = (200, 230, 200)
    s = 6
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    pts = ((room.outline.polygon - [g.x0, g.z0]) / g.cell * s).astype(np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 255), 2)
    for e in room.outline.edges:
        a, b = ((np.array([e.p0, e.p1]) - [g.x0, g.z0]) / g.cell * s).astype(np.int32)
        cv2.putText(img, f"{e.length:.2f}", tuple(((a + b) // 2).tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0) if e.supported else (0, 120, 0), 2)
    for p in ((traj - [g.x0, g.z0]) / g.cell * s).astype(np.int32)[::3]:
        cv2.circle(img, tuple(p.tolist()), 3, (255, 0, 255), -1)
    cv2.imwrite(str(path), img)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    run, key = Path(argv[0]), argv[1]
    variants = argv[argv.index("--variants") + 1].split(",") if "--variants" in argv else ["points", "rays"]
    png = Path(argv[argv.index("--png") + 1]) if "--png" in argv else None
    overrides = {k: float(v) if "." in v else int(v) for k, v in (a.split("=") for a in argv if "=" in a and not a.startswith("--"))}
    sfm, dense, rotation, scale, kf, depth = load(run)
    cloud = to_cloud(dense, rotation, scale=scale)
    traj = (sfm.centers[sfm.registered] @ rotation.T * scale)[:, [0, 2]]
    grid = make_grid(cloud.points, traj)
    floor = find_floor(cloud.points[:, 1])
    cam_y = float(np.median(sfm.centers[sfm.registered] @ rotation.T * scale, axis=0)[1])
    truth = truth_room(key)
    print(f"{run} vs {key}: tape walls {[v for _, v in truth['walls']]}, ceiling {truth['ceiling']}")
    for v in variants:
        hint = None
        if "rays" in v:
            votes = carve(rays_for(run, sfm, dense, kf, depth), rotation, scale, grid, float(floor.y))
            hint = seen_from_votes(votes)
        walls = wall_finder(cam_y) if "walls" in v else None
        est = estimate(cloud.points, traj, Params(**overrides), grid=grid, free_hint=hint, walls=walls)
        if not est.rooms:
            print(f"[{v}] no room")
            continue
        room = max(est.rooms, key=lambda r: r.area)
        print(f"[{v} {overrides or ''}] {score(room, truth)}")
        if png:
            png.mkdir(parents=True, exist_ok=True)
            draw(est, room, png / f"{run.name}_{v.replace('+', '_')}{'_' + '_'.join(f'{k}{x}' for k, x in overrides.items()) if overrides else ''}.png", traj)


if __name__ == "__main__":
    main(sys.argv[1:])
