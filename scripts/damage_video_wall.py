"""Judge one wall of a video run for damage, with the wall plane fitted here from the run's dense cloud.

    uv run python scripts/damage_video_wall.py out/<name> [--chunks 0,1] [--depth] [--all] [--debug]

The plan's wall planes come from the room estimator and on some clips (``H1``) most of them are chords across gaps,
so the wall that carries the damage is not judged at all. This script separates the two questions: it takes the dense
cloud points of the chosen frame chunks (``work/dense.pkl``, in time order, tenths of the clip by default the first
two), fits the largest vertical plane to them (RANSAC, normal horizontal), makes a one-wall room from it, and runs
``damage_video.assess_video`` on that wall alone. It says nothing about the room estimator, only about what the
detector and the video adapter find on a wall whose plane is right. ``--all`` also prints every plane found.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

import damage_video_clip as C
from floorfathom import damage_video as D
from floorfathom import surfaces as S
from floorfathom.ransac import fit_planes
from floorfathom.schema import Measurement, RoomPlan, Wall


def _m(v: float) -> Measurement:
    return Measurement(value=v, lo=v, hi=v, unit="m", method="fitted here")


def wall_room(out: Path, sfm, chunks: list[int], rotation, scale, floor_y, ceiling: float, which: int = 0):
    _stamp, dense = pickle.loads((out / "work" / "dense.pkl").read_bytes())
    pts = (dense.points[np.isin(dense.chunk, chunks)] @ rotation.T) * scale
    pts = pts[(pts[:, 1] > floor_y + 0.2) & (pts[:, 1] < floor_y + ceiling - 0.2)]
    planes = fit_planes(pts, max_planes=4, thresh=0.03, normal_hint=(0, 1, 0), hint_mode="perpendicular", max_angle_deg=8, range_slope=0.02, min_inlier_frac=0.05)
    if not planes:
        raise SystemExit("no vertical plane found in those chunks")
    pl = planes[which]
    n = pl.normal * np.array([1, 0, 1])
    n /= np.linalg.norm(n)
    d = np.array([-n[2], 0, n[0]])
    inl = pts[pl.inlier_mask]
    along = inl @ d
    lo, hi = np.percentile(along, [1, 99])
    foot = inl.mean(axis=0)
    foot = foot - ((foot - n * pl.offset) @ n) * n  # a point on the plane
    a3, b3 = foot + (lo - foot @ d) * d, foot + (hi - foot @ d) * d
    # the room is on the camera's side
    n_frames, n_chunks = len(sfm.registered), dense.n_chunks
    span = np.concatenate([np.arange(c * n_frames // n_chunks, (c + 1) * n_frames // n_chunks) for c in chunks])
    span = span[sfm.registered[span]]
    cam = (sfm.centers[span] @ rotation.T * scale).mean(axis=0)  # the room is on the side the camera was
    inside = n if (cam - foot) @ n > 0 else -n
    a, b = (a3[0], -a3[2]), (b3[0], -b3[2])
    ln = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    room_dir = np.array([inside[0], -inside[2]])
    # orient the wall so that the room lies to its left (counter-clockwise polygon), which is what surface_planes expects
    if (b[0] - a[0]) * room_dir[1] - (b[1] - a[1]) * room_dir[0] < 0:
        a, b = b, a
    polygon = [a, b, (b[0] + 3 * room_dir[0], b[1] + 3 * room_dir[1]), (a[0] + 3 * room_dir[0], a[1] + 3 * room_dir[1])]
    room = RoomPlan(
        id="wall", polygon=polygon, walls=[Wall(id="w", start=a, end=b, length=_m(ln), evidence="wall_points")],
        ceiling_height=_m(ceiling), floor_area=Measurement(value=9.0, lo=9.0, hi=9.0, unit="m2", method="fitted here"), openings=[],
    )
    return room, planes, pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--chunks", default="0,1")
    ap.add_argument("--plane", type=int, default=0, help="which fitted plane, largest first")
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    plan, align, sfm, kf = C.load(args.out)
    rotation, scale, floor_y = np.array(align["rotation"]), float(align["scale"]), float(align["floor_y"])
    rel = plan.diagnostics.scale_rel_sigma or 0.15
    ceiling = plan.rooms[0].ceiling_height.value or 2.4
    room, planes, pts = wall_room(args.out, sfm, [int(c) for c in args.chunks.split(",")], rotation, scale, floor_y, ceiling, args.plane)
    if args.all:
        for k, p in enumerate(planes):
            print(f"plane {k}: normal {np.round(p.normal, 2)}, {p.inlier_mask.sum()} of {len(pts)} points")
    w = room.walls[0]
    print(f"wall {w.length.value:.2f} m long, from {np.round(w.start, 2)} to {np.round(w.end, 2)}, floor at {floor_y:.2f}, ceiling {ceiling:.2f} m")
    depth = C.make_depth(args.out) if args.depth else None
    notes = D.assess_video(room, floor_y, kf, sfm, rotation, scale, rel, depth)
    print(f"{len(room.damage)} regions")
    for d in room.damage:
        print(f"  {d.id} {d.damage_class} conf {d.class_confidence:.2f} {d.width.value * 100:.0f} x {d.height.value * 100:.0f} cm "
              f"[{d.width.lo * 100:.0f}-{d.width.hi * 100:.0f}] x [{d.height.lo * 100:.0f}-{d.height.hi * 100:.0f}] at s={d.centre[0]:.2f} h={d.centre[1]:.2f}")
    for n in notes:
        print("  note:", n)
    if args.debug:
        C.debug_images(room, D.Views(kf, sfm, D.world_poses(sfm, rotation, scale), scale, depth), floor_y, floor_y + ceiling, args.out / "damage_debug_wall")


if __name__ == "__main__":
    main()
