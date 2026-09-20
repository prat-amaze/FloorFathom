"""One capture in, one plan out."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .drift import correct_points, correct_trajectory, estimate_drift
from .estimate import Params, estimate, make_grid
from .io_lidar import load_scan
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


def _lidar_stitching(cloud, traj, ref, plan: CapturePlan, params: Params, ablate: bool) -> Stitching:
    """Adjacency, overlaps and footprint of the rooms, and what was done about pose drift.

    The plan uses the poses as they are. Drift is estimated anyway and the footprint is
    recomputed with it corrected, so every plan carries its own on/off comparison.
    """
    text = "Poses are used as recorded (ARKit visual-inertial odometry) and drift is not corrected. "
    if not ablate or ref.floor is None:
        return describe(plan.rooms, [], [], text + "The drift ablation was not run.")
    drift = estimate_drift(cloud, ref.floor.y)
    fixed = correct_points(cloud, drift)
    on = estimate(fixed.points, correct_trajectory(traj, drift), params, grid=make_grid(fixed.points, traj))
    off_area = describe(plan.rooms, [], [], "").footprint.value
    on_area = union_area([r.outline.polygon for r in on.rooms])
    text += (
        f"Ablation: the drift estimate (walls of {drift.n_pairs} chunk pairs registered) puts chunks up to "
        f"{drift.max_shift * 100:.0f} cm and {drift.max_yaw_deg:.1f} deg apart; applying it gives a footprint of "
        f"{on_area:.1f} m2 ({len(on.rooms)} rooms) against {off_area:.1f} m2 ({len(plan.rooms)} rooms) without "
        f"({(on_area / off_area - 1) * 100:+.1f}%). It is not applied: on two repeat walks of one flat it did not "
        "make them agree better, and the tolerances of the estimate are assumptions, not calibrated."
    )
    return describe(plan.rooms, [], [], text)


def run_lidar(
    capture: Path,
    out: Path,
    replicates: int = 20,
    seed: int = 0,
    target_frames: int = 800,
    n_chunks: int = 20,
    debug: bool = True,
    params: Params | None = None,
    drift_ablation: bool = True,
) -> CapturePlan:
    t0 = time.perf_counter()
    params = params or Params()
    scan = load_scan(capture)
    cloud = build_cloud(scan, target_frames=target_frames, n_chunks=n_chunks)
    traj = scan.positions[:, [0, 2]]
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
    plan.stitching = _lidar_stitching(cloud, traj, ref, plan, params, drift_ablation)
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
    raise NotImplementedError(f"the {tier} tier is not implemented yet; only lidar is")


def schema_json() -> str:
    from .schema import json_schema

    return json.dumps(json_schema(), indent=2)
