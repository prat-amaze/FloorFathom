"""One video clip in, one plan out: the video tier of ``floorfathom plan``.

keyframes -> SfM poses -> rough gravity -> dense cloud (depth model fitted to SfM) -> refined
gravity -> the same room estimator and leave-chunks-out intervals as the LiDAR tier.

Metric scale comes from the depth model's median unless a scale is passed in (a reference object,
see ``anchor.py``). The depth model alone was off by -1%, +7% and +67% on three clips, so its scale
carries a wide relative uncertainty that is added to every interval, and every room is flagged
``scale_from_depth_model_only``. A failed reconstruction gives a plan with no rooms and the reason,
never an invented number.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .estimate import Params, estimate, make_grid
from .io_video import extract_keyframes
from .points_video import build_dense_cloud, to_cloud
from .render import render_plan
from .report import capture_plan
from .schema import CapturePlan, Diagnostics, Measurement
from .sfm import MIN_REGISTERED, run_sfm
from .uncertainty import bootstrap, jackknife_scale
from .world import estimate_gravity, refine_gravity

DEPTH_LONG_SIDE = 924  # the depth model's scale depends on its input size; never mix sizes
SCALE_FLOOR_REL = 0.15  # relative uncertainty of a depth-model-only scale: an assumption, to be calibrated
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


def _write(plan: CapturePlan, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, out / "plan.png")


def run_video(
    capture: str | Path,
    out: str | Path,
    replicates: int = 10,
    seed: int = 0,
    scale: float | None = None,
    scale_rel_sigma: float | None = None,
    depth: Depth | None = None,
    params: Params | None = None,
    **_ignored,
) -> CapturePlan:
    """Plan for one clip. ``scale`` (metres per SfM unit) and its relative sigma override the depth model's."""
    t0 = time.perf_counter()
    capture, out = Path(capture), Path(out)
    clips = [capture] if capture.is_file() else sorted([*capture.glob("*.MOV"), *capture.glob("*.mp4")])
    if len(clips) != 1:
        raise NotImplementedError(f"{capture} holds {len(clips)} clips; pass one clip file (several rooms are stitched later)")
    clip = clips[0]
    name = clip.stem
    params = params or Params()

    work = out / "work"
    st = clip.stat()
    sfm_stamp = f"v1:{clip.name}:{st.st_size}:{int(st.st_mtime)}:seed{seed}"
    kf = extract_keyframes(clip, work / "frames")
    sfm = _load_or_run(work / "sfm.pkl", sfm_stamp, lambda: run_sfm(kf, work / "colmap", seed=seed))
    if sfm is None or sfm.registered_fraction < MIN_REGISTERED:
        notes = ["sfm_failed" if sfm is None else "sfm_registered_too_few_frames"] + ([] if sfm is None else sfm.flags)
        plan = _no_plan(name, time.perf_counter() - t0, seed, notes, n_keyframes=len(kf))
        _write(plan, out)
        return plan

    stride = max(1, int(sfm.registered.sum()) // TARGET_DENSE_FRAMES)
    dense_stamp = f"{sfm_stamp}:depth{DEPTH_LONG_SIDE}:stride{stride}:chunks{N_CHUNKS}"
    if depth is None:  # a caller-supplied depth function is never cached: it may differ between calls
        dense = _load_or_run(
            work / "dense.pkl", dense_stamp,
            lambda: build_dense_cloud(kf, sfm, _default_depth(), n_chunks=N_CHUNKS, frame_stride=stride),
        )
    else:
        dense = build_dense_cloud(kf, sfm, depth, n_chunks=N_CHUNKS, frame_stride=stride)
    if dense is None:
        plan = _no_plan(name, time.perf_counter() - t0, seed, ["dense_cloud_failed"] + sfm.flags, n_keyframes=len(kf))
        _write(plan, out)
        return plan

    if scale is None:
        scale, rel = dense.scale_from_depth_model, _scale_rel_sigma(dense.frame_ratio)
        method = "monocular_depth"
    else:
        rel, method = scale_rel_sigma if scale_rel_sigma is not None else 0.03, "reference_object"

    rough = estimate_gravity(sfm)
    refined = refine_gravity(to_cloud(dense, rough.rotation, scale=scale).points)
    rotation = refined.rotation @ rough.rotation
    cloud = to_cloud(dense, rotation, scale=scale)
    traj = (sfm.centers[sfm.registered] @ rotation.T * scale)[:, [0, 2]]

    ref = estimate(cloud.points, traj, params, grid=make_grid(cloud.points, traj))
    samples = bootstrap(cloud, traj, ref, params, replicates=replicates, seed=seed)
    plan = capture_plan(
        name, "video", ref, samples, replicates, seed, len(dense.frame_ratio), len(cloud),
        time.perf_counter() - t0, None if ref.floor is None else ref.floor.sharpness, jackknife_scale(N_CHUNKS, 2),
    )

    flags = list(dict.fromkeys(sfm.flags + dense.flags + rough.flags + refined.flags))
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
    if method == "monocular_depth":
        from .models import DEPTH, REGISTRY

        d.models = [f"{DEPTH}@{REGISTRY[DEPTH].revision[:10]}"]
    _write(plan, out)
    return plan
