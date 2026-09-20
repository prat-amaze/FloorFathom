"""One video clip in, one plan out: the video tier of ``floorfathom plan``.

keyframes -> SfM poses -> rough gravity -> dense cloud (depth model fitted to SfM) -> refined
gravity -> the same room estimator and leave-chunks-out intervals as the LiDAR tier.

Metric scale comes from the reference ruler of the capture protocol (a ruler with a 31.6 cm yellow body) when it is found at the
start of the clip and is plausible (``anchor.py``). Otherwise it comes from the depth model's
median, which alone was off by -1%, +7% and +67% on three clips: its scale then carries a wide
relative uncertainty that is added to every interval, and every room is flagged
``scale_from_depth_model_only`` with the reason the strip was not used. A failed reconstruction
gives a plan with no rooms and the reason, never an invented number.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .anchor import REFERENCE_LENGTH_M, VIDEO_STRIP_COLOUR, estimate_anchor, track_strip
from .damage_video import assess_video, cached_depth, renumber_damage
from .estimate import Params, estimate, make_grid
from .io_video import extract_keyframes
from .points_video import build_dense_cloud, to_cloud
from .render import render_plan
from .report import capture_plan
from .schema import CapturePlan, Diagnostics, Measurement, RoomPlan
from .sfm import MIN_REGISTERED, run_sfm
from .stitch import stitch_plans
from .uncertainty import bootstrap, jackknife_scale
from .world import estimate_gravity, refine_gravity

DEPTH_LONG_SIDE = 924  # the depth model's scale depends on its input size; never mix sizes
SCALE_FLOOR_REL = 0.15  # relative uncertainty of a depth-model-only scale: an assumption, to be calibrated
STRIP_WINDOW_S = 15.0  # the protocol puts the strip move at the start of the clip
STRIP_SCALE_RATIO_OK = (0.5, 2.0)  # strip scale / depth-model scale outside this: not the strip
STRIP_LENGTH_REL = 0.004  # 1 mm on 30 cm plus the edges of the yellow part: how well the ruler itself is known
KEYFRAME_STEP_S = 0.2  # one keyframe per this many seconds; SfM time grows faster than the keyframe count (0.15 s: 25-35 min per clip, 0.3 s: 11 min but lost the H1 ceiling)
TARGET_DENSE_FRAMES = 50
N_CHUNKS = 10
CONVENTIONS = "SfM (pycolmap) poses; depth-model depth fitted to SfM; world +y up from floor and ceiling planes"

Depth = Callable[[np.ndarray], np.ndarray]


def _default_depth() -> Depth:
    from .depth import DepthEstimator

    est = DepthEstimator(threads=max(1, (os.cpu_count() or 2) // 2))
    return lambda rgb: est(rgb, long_side=DEPTH_LONG_SIDE)


def _widen(m: Measurement, rel: float) -> None:
    """Grow the interval of a measurement to cover a relative error ``rel`` on top of the existing half-width."""
    if m.value is None or m.lo is None or m.hi is None:
        return
    half = float(np.hypot(max(m.hi - m.value, m.value - m.lo), rel * abs(m.value)))
    m.lo, m.hi = round(max(0.0, m.value - half), 4), round(m.value + half, 4)


def _scale_rel_sigma(ratios: dict[int, float]) -> float:
    r = np.array(list(ratios.values()))
    med = float(np.median(r))
    se = 1.2533 * 1.4826 * float(np.median(np.abs(r - med))) / med / np.sqrt(len(r))  # standard error of the median
    return float(np.hypot(SCALE_FLOOR_REL, se))


def _choose_scale(
    kf, sfm, depth_scale: float, depth_rel: float, reference_length_m: float = REFERENCE_LENGTH_M
) -> tuple[float, float, str, list[str]]:
    """(metres per SfM unit, relative sigma, method, flags): the reference strip when it is found
    and plausible, else the depth model's scale with the reason the strip was not used."""
    obs = track_strip(kf, sfm, until_s=STRIP_WINDOW_S, colour=VIDEO_STRIP_COLOUR)
    anchor = estimate_anchor(sfm, obs, reference_length_m) if len(obs) >= 3 else None
    if anchor is None:
        return depth_scale, depth_rel, "monocular_depth", ["reference_strip_not_found"]
    bad = list(anchor.flags)
    if not STRIP_SCALE_RATIO_OK[0] <= anchor.scale / depth_scale <= STRIP_SCALE_RATIO_OK[1]:
        bad.append("reference_strip_implausible")  # far from any depth-model error seen: probably another object
    if bad:
        return depth_scale, depth_rel, "monocular_depth", ["reference_strip_rejected", *bad]
    return anchor.scale, float(np.hypot(anchor.rel_sigma, STRIP_LENGTH_REL)), "reference_object", []


def _no_plan(name: str, seconds: float, seed: int, notes: list[str], sfm=None, n_keyframes: int = 0) -> CapturePlan:
    return CapturePlan(
        capture=name,
        tier="video",
        rooms=[],
        diagnostics=Diagnostics(
            frames_used=n_keyframes,
            bootstrap_replicates=0,
            seed=seed,
            seconds=round(seconds, 2),
            conventions=CONVENTIONS,
            notes=notes,
        ),
    )


def _load_or_run(cache: Path, stamp: str, compute):
    """Result of ``compute()``, replayed from ``cache`` when it was made for the same ``stamp``.

    SfM and the depth pass take minutes, so a rerun (or a run that died on a full disk or memory)
    must not repeat them. A missing, corrupt or stale cache is simply recomputed; a ``None`` result
    (a failed step) is never cached.
    """
    try:
        saved_stamp, value = pickle.loads(cache.read_bytes())
        if saved_stamp == stamp:
            return value
    except Exception:
        pass
    value = compute()
    if value is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps((stamp, value)))
    return value


def write_alignment(path: Path, rotation: np.ndarray, scale: float, floor_y: float | None, size: tuple[int, int]) -> None:
    """Save how SfM coordinates map into the plan's frame, so posed keyframes can be placed on the plan's walls.

    A SfM point ``p`` is ``q = (p @ rotation.T) * scale`` in the plan frame: x and z are the plan's axes (the
    polygons and walls use (x, z)), y is up, in metres. A keyframe's centre is placed the same way and its
    camera-to-world rotation is ``rotation @ cam_to_world``; wall heights are ``y - floor_y``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "rotation": np.asarray(rotation).tolist(), "scale": float(scale), "floor_y": floor_y, "image_size": list(size),
        "frame": "q = (p @ rotation.T) * scale; plan (x, z), y up; camera-to-world in this frame = rotation @ cam_to_world",
    }, indent=2))


def _write(plan: CapturePlan, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, out / "plan.png")


def _renumber(room: RoomPlan, k: int) -> None:
    """Ids unique across a property: every clip's room is ``room_0`` with ``r0_`` walls and openings."""
    room.id = f"room_{k}"
    for w in room.walls:
        w.id = w.id.replace("r0_", f"r{k}_", 1)
    for o in room.openings:
        o.id, o.wall_id = o.id.replace("r0_", f"r{k}_", 1), o.wall_id.replace("r0_", f"r{k}_", 1)
    renumber_damage(room, k)


def run_video(capture: str | Path, out: str | Path, **kw) -> CapturePlan:
    """One plan for a clip, or one stitched property plan for a folder of clips (each clip is one room).

    Every clip gets its own plan and cache under ``out/rooms/<clip>/``; the rooms are placed in one
    frame by gluing their doorways (``stitch.py``). A clip that gave no room is listed in the notes, and
    a room that could not be placed stays in its own frame (``stitching.unplaced``).
    """
    t0 = time.perf_counter()
    capture, out = Path(capture), Path(out)
    clips = [capture] if capture.is_file() else sorted([*capture.glob("*.MOV"), *capture.glob("*.mp4")])
    if not clips:
        raise ValueError(f"no video clips found in {capture}")
    if len(clips) == 1:
        return run_clip(clips[0], out, **kw)
    plans = [run_clip(clip, out / "rooms" / clip.stem, **kw) for clip in clips]
    built = [p for p in plans if p.rooms and p.rooms[0].polygon]
    for k, p in enumerate(built):
        _renumber(p.rooms[0], k)
    seed = kw.get("seed", 0)
    if not built:
        result = _no_plan(capture.name, time.perf_counter() - t0, seed, ["no_clip_gave_a_room"] + [f"{p.capture}: {n}" for p in plans for n in p.diagnostics.notes])
    else:
        result = stitch_plans(built, capture.name)
        kept = {r.id for r in result.rooms}
        result.rooms += [p.rooms[0] for p in built if p.rooms[0].id not in kept]
        result.diagnostics.notes += [f"{p.capture}: no room ({', '.join(p.diagnostics.notes)})" for p in plans if p not in built]
    _write(result, out)
    return result


def run_clip(
    clip: Path,
    out: Path,
    replicates: int = 10,
    seed: int = 0,
    scale: float | None = None,
    scale_rel_sigma: float | None = None,
    reference_length_m: float | None = None,
    depth: Depth | None = None,
    params: Params | None = None,
    keyframe_step_s: float = KEYFRAME_STEP_S,
    assess_damage: bool = True,
    **_ignored,
) -> CapturePlan:
    """Plan for one clip. ``scale`` (metres per SfM unit) and its relative sigma override the depth model's;
    ``reference_length_m`` is the tape-measured length of the ruler's yellow body (default: ours)."""
    t0 = time.perf_counter()
    name = clip.stem
    params = params or Params()

    work = out / "work"
    st = clip.stat()
    sfm_stamp = f"v2:{clip.name}:{st.st_size}:{int(st.st_mtime)}:seed{seed}"
    if keyframe_step_s != KEYFRAME_STEP_S:  # caches made at the default spacing keep their stamp
        sfm_stamp += f":step{keyframe_step_s:g}"
    kf = extract_keyframes(clip, work / "frames", step_s=keyframe_step_s)
    sfm = _load_or_run(work / "sfm.pkl", sfm_stamp, lambda: run_sfm(kf, work / "colmap", seed=seed))
    if sfm is None or sfm.registered_fraction < MIN_REGISTERED:
        notes = ["sfm_failed" if sfm is None else "sfm_registered_too_few_frames"] + ([] if sfm is None else sfm.flags)
        plan = _no_plan(name, time.perf_counter() - t0, seed, notes, n_keyframes=len(kf))
        _write(plan, out)
        return plan

    stride = max(1, int(sfm.registered.sum()) // TARGET_DENSE_FRAMES)
    dense_stamp = f"{sfm_stamp}:depth{DEPTH_LONG_SIDE}:stride{stride}:chunks{N_CHUNKS}:depthcache"
    # the dense pass and the damage step share one set of depth maps on disk; a caller-supplied depth function
    # is never cached: it may differ between calls
    dm = cached_depth(_default_depth(), work / f"depth{DEPTH_LONG_SIDE}") if depth is None else depth
    if depth is None:
        dense = _load_or_run(
            work / "dense.pkl", dense_stamp,
            lambda: build_dense_cloud(kf, sfm, dm, n_chunks=N_CHUNKS, frame_stride=stride),
        )
    else:
        dense = build_dense_cloud(kf, sfm, dm, n_chunks=N_CHUNKS, frame_stride=stride)
    if dense is None:
        plan = _no_plan(name, time.perf_counter() - t0, seed, ["dense_cloud_failed"] + sfm.flags, n_keyframes=len(kf))
        _write(plan, out)
        return plan

    if scale is None:
        scale, rel, method, scale_flags = _choose_scale(
            kf, sfm, dense.scale_from_depth_model, _scale_rel_sigma(dense.frame_ratio), reference_length_m or REFERENCE_LENGTH_M
        )
    else:
        rel, method, scale_flags = scale_rel_sigma if scale_rel_sigma is not None else 0.03, "reference_object", []

    rough = estimate_gravity(sfm)
    refined = refine_gravity(to_cloud(dense, rough.rotation, scale=scale).points)
    rotation = refined.rotation @ rough.rotation
    cloud = to_cloud(dense, rotation, scale=scale)
    traj = (sfm.centers[sfm.registered] @ rotation.T * scale)[:, [0, 2]]

    ref = estimate(cloud.points, traj, params, grid=make_grid(cloud.points, traj))
    write_alignment(work / "alignment.json", rotation, scale, None if ref.floor is None else float(ref.floor.y), sfm.image_size)
    samples = bootstrap(cloud, traj, ref, params, replicates=replicates, seed=seed)
    plan = capture_plan(
        name, "video", ref, samples, replicates, seed, len(dense.frame_ratio), len(cloud),
        time.perf_counter() - t0, None if ref.floor is None else ref.floor.sharpness, jackknife_scale(N_CHUNKS, 2),
    )

    flags = list(dict.fromkeys(sfm.flags + dense.flags + rough.flags + refined.flags + scale_flags))
    if method == "monocular_depth":
        flags.append("scale_from_depth_model_only")
    for room in plan.rooms:
        room.name, room.frame = name, "room"
        for w in room.walls:
            _widen(w.length, rel)
        _widen(room.ceiling_height, rel)
        _widen(room.floor_area, 2 * rel)
        for o in room.openings:
            _widen(o.width, rel)
        room.flags = list(dict.fromkeys(room.flags + flags))
    d = plan.diagnostics
    d.conventions, d.scale_method, d.scale_factor, d.scale_rel_sigma = CONVENTIONS, method, round(scale, 5), round(rel, 4)
    d.notes = flags + ([] if plan.rooms else ["no_room_found"])
    if depth is None:  # the default depth model builds the dense cloud whatever gave the scale
        from .models import DEPTH, REGISTRY

        d.models = [f"{DEPTH}@{REGISTRY[DEPTH].revision[:10]}"]
    if assess_damage and ref.floor is not None and plan.rooms:  # before stitching: damage is in each clip's own room frame
        for room in plan.rooms:
            d.notes += [f"{room.id}: {n}" for n in assess_video(room, float(ref.floor.y), kf, sfm, rotation, scale, rel, dm, debug_dir=out / "debug" / "damage", frames=set(dense.frame_ratio))]
        d.notes.append("Damage classes, concealed-damage rules and scope actions are our own definitions, tuned on synthetic surfaces; no real damage was available to check them (README).")
    _write(plan, out)
    return plan
