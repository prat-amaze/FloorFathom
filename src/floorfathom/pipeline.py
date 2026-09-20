"""One capture in, one plan out."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .drift import correct_points, correct_trajectory, estimate_drift
from .estimate import Params, estimate, make_grid
from .io_lidar import load_scan
from .planes import find_floor
from .points import build_cloud
from .render import render_plan
from .report import capture_plan
from .schema import CapturePlan, Stitching
from .stitch import describe, union_area
from .uncertainty import bootstrap, jackknife_scale


def detect_tier(capture: Path) -> str:
    if capture.is_file() and capture.suffix.lower() in (".mov", ".mp4"):
        return "video"
    if (capture / "depth").is_dir() and (capture / "odometry.csv").is_file():
        return "lidar"
    if any(capture.glob("*.MOV")) or any(capture.glob("*.mp4")):
        return "video"
    if (capture / "images").is_dir() or any(p.is_dir() for p in capture.iterdir()):
        return "photo"
    raise ValueError(f"cannot tell the input tier of {capture}")


def _lidar_stitching(raw, raw_traj, plan: CapturePlan, params: Params, drift, ablate: bool, why_not: str) -> Stitching:
    """Adjacency, overlaps and footprint of the rooms, and what was done about pose drift.

    The plan comes from the drift-corrected walk. The uncorrected walk is estimated too, so every
    plan carries its own on/off footprint comparison.
    """
    if drift is None:
        return describe(plan.rooms, [], [], f"Drift was not corrected: {why_not}.")
    if drift.n_pairs == 0:
        text = "Drift could not be corrected: no two chunks of the walk overlapped enough to register. "
    else:
        text = (
            f"Drift is corrected by a pose graph (drift.py): walls of {drift.n_pairs} chunk pairs of the walk are "
            f"registered against each other and one (x, z, yaw) correction per chunk is solved by weighted least "
            f"squares; the largest is {drift.max_shift * 100:.0f} cm and {drift.max_yaw_deg:.1f} deg. "
        )
    if ablate:
        off = estimate(raw.points, raw_traj, params, grid=make_grid(raw.points, raw_traj))
        off_area = union_area([r.outline.polygon for r in off.rooms])
        on_area = describe(plan.rooms, [], [], "").footprint.value
        text += (
            f"Ablation: footprint {on_area:.1f} m2 ({len(plan.rooms)} rooms) with the correction, {off_area:.1f} m2 "
            f"({len(off.rooms)} rooms) without ({(on_area / max(off_area, 1e-9) - 1) * 100:+.1f}%). "
        )
    text += (
        "The tolerances of the estimate are assumptions, not calibrated, and on two repeat walks of one flat "
        "the correction did not make them agree better (README)."
    )
    return describe(plan.rooms, [], [], text)


def _lidar_damage(scan, traj, drift, plan: CapturePlan) -> list[str]:
    """Damage, concealed-damage flags and scope of every room, from the scan's own RGB and depth frames."""
    from .assess import assess_room
    from .lidar_frames import LidarFrames

    floor_y = plan.diagnostics.floor_height_world
    if floor_y is None:
        return ["damage was not assessed: no floor plane was found"]
    frames = LidarFrames(scan, traj, drift)
    notes: list[str] = []
    try:
        for room in plan.rooms:
            ch = room.ceiling_height.value
            ceiling_y = None if ch is None else floor_y + ch
            others = [r for r in plan.rooms if r is not room]
            notes += [f"{room.id}: {n}" for n in assess_room(room, floor_y, ceiling_y, frames.for_room(room), others)]
    finally:
        frames.close()
    return notes


def run_lidar(
    capture: Path,
    out: Path,
    replicates: int = 20,
    seed: int = 0,
    target_frames: int = 800,
    n_chunks: int = 20,
    debug: bool = True,
    params: Params | None = None,
    correct_drift: bool = True,
    drift_ablation: bool = True,
    assess_damage: bool = True,
) -> CapturePlan:
    t0 = time.perf_counter()
    params = params or Params()
    scan = load_scan(capture)
    cloud = build_cloud(scan, target_frames=target_frames, n_chunks=n_chunks)
    traj = scan.positions[:, [0, 2]]
    raw, raw_traj, drift, why_not = cloud, traj, None, "switched off"
    floor = find_floor(cloud.points[:, 1]) if correct_drift else None
    if floor is not None:
        drift = estimate_drift(cloud, floor.y)
        cloud, traj = correct_points(cloud, drift), correct_trajectory(traj, drift)
    elif correct_drift:
        why_not = "no floor plane was found to anchor the drift estimate"
    grid = make_grid(cloud.points, traj)
    ref = estimate(cloud.points, traj, params, grid=grid)
    samples = bootstrap(cloud, traj, ref, params, replicates=replicates, seed=seed)
    frames_used = min(len(scan), target_frames)
    sharp = None if ref.floor is None else ref.floor.sharpness
    plan = capture_plan(
        capture.name,
        "lidar",
        ref,
        samples,
        replicates,
        seed,
        frames_used,
        len(cloud),
        time.perf_counter() - t0,
        sharp,
        jackknife_scale(n_chunks, 2),
    )
    plan.stitching = _lidar_stitching(raw, raw_traj, plan, params, drift, drift_ablation, why_not)
    if assess_damage:
        plan.diagnostics.notes += _lidar_damage(scan, traj, drift, plan)
        plan.diagnostics.notes.append(
            "Damage classes, concealed-damage rules and scope actions are our own definitions, tuned on synthetic surfaces; "
            "no real damage was available to check them (README)."
        )
        plan.diagnostics.seconds = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, out / "plan.png")
    if debug:
        from .debug import write_debug

        write_debug(out / "debug", cloud, ref, scan)
    return plan


def run(capture: str | Path, out: str | Path, tier: str | None = None, **kw) -> CapturePlan:
    capture, out = Path(capture), Path(out)
    tier = tier or detect_tier(capture)
    if tier == "lidar":
        return run_lidar(capture, out, **kw)
    if tier == "video":
        from .video_pipeline import run_video

        return run_video(capture, out, **kw)
    if tier == "photo":
        from .photo_pipeline import run_photo

        return run_photo(capture, out, **kw)
    raise NotImplementedError(f"the {tier} tier is not implemented yet; only lidar is")


def schema_json() -> str:
    from .schema import json_schema

    return json.dumps(json_schema(), indent=2)
