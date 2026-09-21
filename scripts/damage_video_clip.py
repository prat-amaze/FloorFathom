"""Run damage detection on a finished video run, from its cached files only (no SfM, no dense pass).

    uv run python scripts/damage_video_clip.py out/<name> [--depth] [--room room_0] [--debug]

Reads ``plan.json``, ``work/frames/*.jpg``, ``work/sfm.pkl`` and ``work/alignment.json`` of a run of ``floorfathom
plan`` on a clip, judges the walls and ceiling of each room (or the one named) with ``damage_video.assess_video``
and writes ``damage.json`` (the rooms with their damage, concealed-damage flags and scope items) next to the plan.
``--depth`` loads the depth model to leave out what stands in front of a wall (slow, needs the models and about a
gigabyte of memory); without it a pixel is dropped only when its views disagree in lightness. ``--debug`` also
writes ``damage_debug/<wall>.png``: each judged surface unrolled with its regions outlined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from floorfathom import damage_video as D
from floorfathom import surfaces as S
from floorfathom.io_video import Keyframes
from floorfathom.schema import CapturePlan


def load(out: Path):
    plan = CapturePlan.model_validate_json((out / "plan.json").read_text())
    align = json.loads((out / "work" / "alignment.json").read_text())
    _stamp, sfm = pickle.loads((out / "work" / "sfm.pkl").read_bytes())
    frames = out / "work" / "frames"
    names = sorted(p.name for p in frames.glob("*.jpg"))
    n = len(names)
    zeros = np.zeros(n)
    kf = Keyframes(frames, names, zeros.astype(int), zeros, zeros, tuple(align["image_size"]), 30.0, out)
    return plan, align, sfm, kf


def surface_patches(room, views: D.Views, floor_y: float, ceiling_y):
    """(surface, plane, patch) of every surface the video tier judges, unrolled from the frames the adapter picks."""
    for ref, plane in S.surface_planes(room, floor_y, ceiling_y):
        if ref.kind not in D.KINDS:
            continue
        acc = S.Accumulator(plane, 2, False)
        for fr in views.frames_for(plane):
            acc.add(fr)
        yield ref, plane, acc.patch()


def save_patches(room, views: D.Views, floor_y: float, ceiling_y, dest: Path, prefix: str) -> None:
    """One small .npz per surface (rgb as float16, valid, m_per_px): a test set for the detector."""
    dest.mkdir(parents=True, exist_ok=True)
    for ref, _plane, patch in surface_patches(room, views, floor_y, ceiling_y):
        np.savez_compressed(dest / f"{prefix}_{ref.id}.npz", rgb=patch.rgb.astype(np.float16), valid=patch.valid, m_per_px=patch.m_per_px)


def debug_images(room, views: D.Views, floor_y: float, ceiling_y, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for ref, plane, patch in surface_patches(room, views, floor_y, ceiling_y):
        img = (np.clip(patch.rgb, 0, 1) * 255).astype(np.uint8)
        img[~patch.valid] = (img[~patch.valid] * 0.25 + np.array([60, 0, 60]) * 0.75).astype(np.uint8)
        ox, oy = (0.0, 0.0) if ref.kind == "wall" else (plane.origin[0], -plane.origin[2])
        h = img.shape[0]
        if ref.kind == "wall":
            img = np.ascontiguousarray(img[::-1])  # row 0 of the patch is the floor
        for d in room.damage:
            if d.surface.id != ref.id:
                continue
            pts = np.array([[(x - ox) / S.MPP, (y - oy) / S.MPP] for x, y in d.polygon])
            if ref.kind == "wall":
                pts[:, 1] = h - pts[:, 1]
            pts = pts.astype(np.int32)
            cv2.polylines(img, [pts], True, (255, 255, 0), 2)
            cv2.putText(img, d.damage_class, tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imwrite(str(dest / f"{ref.id}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def make_depth(out: Path):
    """The depth model as ``rgb -> metres``, at the same input size as the SfM pass (its scale depends on it), with the
    maps kept in ``work/damage_depth`` so that a rerun does not repeat the model."""
    from floorfathom.depth import DepthEstimator

    est = DepthEstimator(threads=2)
    cache = out / "work" / "damage_depth"
    cache.mkdir(exist_ok=True)

    def depth(rgb):
        f = cache / f"{hashlib.md5(rgb.tobytes()).hexdigest()}.npy"
        if f.exists():
            return np.load(f).astype(np.float32)
        d = est(rgb, long_side=924)
        np.save(f, d.astype(np.float16))
        return d

    return depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--room")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save-patches", type=Path, help="also save every judged surface as <dir>/<run name>_<surface>.npz")
    ap.add_argument("--closure-walls", action="store_true", help="diagnostic: judge walls that are only chords across gaps too")
    args = ap.parse_args()
    plan, align, sfm, kf = load(args.out)
    rotation, scale, floor_y = np.array(align["rotation"]), float(align["scale"]), float(align["floor_y"])
    rel = plan.diagnostics.scale_rel_sigma or 0.15
    depth = make_depth(args.out) if args.depth else None
    for room in plan.rooms:
        if args.room and room.id != args.room:
            continue
        if args.closure_walls:
            for w in room.walls:
                w.evidence = "wall_points"
        notes = D.assess_video(room, floor_y, kf, sfm, rotation, scale, rel, depth)
        print(f"{room.id}: {len(room.damage)} regions, {len(room.concealed_flags)} flags, {len(room.scope)} scope items")
        for d in room.damage:
            print(f"  {d.id} {d.surface.id} {d.damage_class} conf {d.class_confidence:.2f} {d.width.value * 100:.0f} x {d.height.value * 100:.0f} cm at {d.centre}")
        for n in notes:
            print("  note:", n)
        ceiling = None if room.ceiling_height.value is None else floor_y + room.ceiling_height.value
        if args.save_patches:
            save_patches(room, D.Views(kf, sfm, D.world_poses(sfm, rotation, scale), scale, depth), floor_y, ceiling, args.save_patches, args.out.name)
        if args.debug:
            debug_images(room, D.Views(kf, sfm, D.world_poses(sfm, rotation, scale), scale, depth), floor_y, ceiling, args.out / "damage_debug")
    (args.out / "damage.json").write_text(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
