"""Photo tier: one plan per room folder, then the rooms stitched into a property plan.

Per room: rotations between the stills (photo_pose), metric depth per still (depth), one gravity-levelled
cloud (photo_scene), walls / outline / heights / doorways from planes (photo_layout), and intervals from
leaving one photo out at a time plus assumed systematic terms. Thin input gives an explicit null room with
flags, never a confident guess.

Absolute scale is the weak point of monocular depth: on the four rooms of Data/ the depth model's metres were
right within a few percent in one room and 45-65% too long in two others. With no reference in the capture the
scale is taken from the model and its uncertainty is set to ``MONO_SCALE_REL_SIGMA``; a caller that knows the
scale (a reference object) passes ``scale`` and ``scale_rel_sigma``, as in the video tier.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import layout as L
from .estimate import Estimate, RoomEst, make_grid
from .io_photos import PhotoSet, discover_rooms, load_photo_set
from .photo_layout import CEILING_RANGE, OPENING_WIDTH, find_openings, floor_and_ceiling, outline_from_segments, wall_segments
from .photo_pose import Poses, register_rotations
from .photo_scene import RoomScene, build_scene
from .render import render_plan
from .report import capture_plan
from .schema import CapturePlan, Diagnostics, Measurement, RoomPlan, Station
from .stitch import stitch_plans
from .uncertainty import RoomSamples, _iou, _match_edge, jackknife_scale

Z95 = 1.96
DEPTH_LONG_SIDE = 924
MONO_SCALE_REL_SIGMA = 0.30  # assumed 1-sigma of the depth model's room scale with no reference (four rooms spread 0-65%)
# 95% half-widths added in quadrature to the sampling spread: (absolute m, relative). Assumed, not calibrated:
# four rooms cannot calibrate anything. They cover registration by rotation only, arm swing and depth bowing.
SYS_LENGTH, SYS_HEIGHT, SYS_AREA, SYS_OPENING = (0.05, 0.10), (0.05, 0.09), (0.10, 0.19), (0.06, 0.10)
MAX_UNSUPPORTED_FRAC = 0.25  # more of the perimeter unobserved than this and the area is not reported
CONVENTIONS = "rotation-only registration of stills from one station; monocular metric depth; world +y up from wall thinness"

Depth = Callable[[np.ndarray], np.ndarray]


def cached_depth(cache_dir: Path, long_side: int = DEPTH_LONG_SIDE) -> Depth:
    """The depth model behind a disk cache keyed on the pixels, the model revision and the input size.

    A cache hit is exactly what the model returns again (it is deterministic), so replay and a live run agree.
    """
    from .models import DEPTH, REGISTRY

    rev, model = REGISTRY[DEPTH].revision[:10], []

    def depth(rgb: np.ndarray) -> np.ndarray:
        f = cache_dir / f"{hashlib.sha256(rgb.tobytes()).hexdigest()[:16]}_{rev}_{long_side}.npy"
        if f.exists():
            return np.load(f)
        if not model:
            from .depth import DepthEstimator

            model.append(DepthEstimator(threads=1))
        z = model[0](rgb, long_side=long_side)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(f, z)
        return z

    return depth


def _room_from_points(points: np.ndarray, grid: L.Grid, seed: int) -> RoomEst | None:
    segs = wall_segments(points, seed)
    outline = outline_from_segments(segs, grid)
    if outline is None:
        return None
    floor, ceiling = floor_and_ceiling(points, seed)
    flags: list[str] = []
    if floor is None or ceiling is None:
        flags.append("ceiling_not_observed")
        ceiling = None
    elif not CEILING_RANGE[0] <= ceiling.y - floor.y <= CEILING_RANGE[1]:
        flags.append("ceiling_height_implausible")
        ceiling = None
    if any((not e.supported) and e.length > OPENING_WIDTH[1] for e in outline.edges):
        flags.append("open_boundary")
    return RoomEst(outline, abs(L.signed_area(outline.polygon)), floor, ceiling, find_openings(segs, outline, points), flags)


def _leave_one_out(points: np.ndarray, chunk: np.ndarray, n: int, grid: L.Grid, ref: RoomEst, seed: int) -> RoomSamples:
    """The reference quantities recomputed with each photo left out in turn, matched back to the reference room."""
    out = RoomSamples()
    for k in range(n):
        got = _room_from_points(points[chunk != k], grid, seed)
        if got is None or _iou(ref.outline.mask, got.outline.mask) <= 0.5:
            continue
        out.matched += 1
        out.area.append(got.area)
        if got.ceiling_height is not None:
            out.ceiling_height.append(got.ceiling_height)
        for i, e in enumerate(ref.outline.edges):
            length = _match_edge(e, got.outline.edges)
            if length is not None:
                out.wall_length.setdefault(i, []).append(length)
        for j, o in enumerate(ref.openings):
            near = [c for c in got.openings if np.linalg.norm(c.centre - o.centre) <= 0.4]
            if near:
                out.opening_width.setdefault(j, []).append(min(near, key=lambda c: np.linalg.norm(c.centre - o.centre)).width)
    return out


def _widen(m: Measurement, sys: tuple[float, float], scale_rel: float) -> None:
    """Grow the interval of ``m`` by the systematic term and the scale uncertainty, added in quadrature."""
    if m.value is None or m.lo is None or m.hi is None:
        return
    half = max(m.hi - m.value, m.value - m.lo)
    half = float(np.sqrt(half**2 + sys[0] ** 2 + (sys[1] * m.value) ** 2 + (Z95 * scale_rel * m.value) ** 2))
    m.lo, m.hi = round(max(0.0, m.value - half), 4), round(m.value + half, 4)


def _null(name: str, seconds: float, seed: int, flags: list[str], models: list[str]) -> CapturePlan:
    def none(unit: str) -> Measurement:
        return Measurement(value=None, lo=None, hi=None, unit=unit, method="not estimated",
                           note="no room was reconstructed from these photos")

    room = RoomPlan(
        id="room_0", name=name, frame="room", polygon=[], walls=[], openings=[], flags=flags,
        ceiling_height=none("m"), floor_area=none("m2"),
    )
    return CapturePlan(
        capture=name, tier="photo", rooms=[room],
        diagnostics=Diagnostics(bootstrap_replicates=0, seed=seed, seconds=round(seconds, 2), conventions=CONVENTIONS,
                                models=models, notes=flags),
    )


def plan_photo_room(
    name: str,
    photos: PhotoSet,
    poses: Poses,
    depth: Depth,
    seed: int = 0,
    scale: float | None = None,
    scale_rel_sigma: float | None = None,
    models: list[str] | None = None,
) -> CapturePlan:
    """The plan of one room from its stills. ``scale`` (metres per depth-model metre) and its relative sigma
    override the monocular default when a reference gives the scale."""
    t0 = time.perf_counter()
    models = models or []
    flags = list(dict.fromkeys(photos.flags + poses.flags))
    scene: RoomScene | None = build_scene(photos, poses, depth, seed)
    if scene is None:
        return _null(name, time.perf_counter() - t0, seed, [*flags, "insufficient_views"], models)
    flags += scene.flags
    factor = 1.0 if scale is None else scale
    method, rel = ("monocular_depth", MONO_SCALE_REL_SIGMA) if scale is None else (
        "reference_object", 0.03 if scale_rel_sigma is None else scale_rel_sigma)
    if scale is None:
        flags.append("scale_from_depth_model_only")
    points = (scene.cloud.points * factor).astype(np.float32)
    grid = make_grid(points, scene.traj_xz * factor)
    ref = _room_from_points(points, grid, seed)
    if ref is None:
        return _null(name, time.perf_counter() - t0, seed, [*flags, "room_outline_not_found"], models)

    n = scene.cloud.n_chunks
    samples = _leave_one_out(points, scene.cloud.chunk, n, grid, ref, seed)
    plan = capture_plan(
        name, "photo", Estimate([ref], grid, ref.floor), [samples], n, seed, n, len(points),
        time.perf_counter() - t0, None if ref.floor is None else ref.floor.sharpness, jackknife_scale(n, 1),
    )
    room = plan.rooms[0]
    room.name, room.frame = name, "room"
    room.stations = [Station(position=(0.0, 0.0), height_above_floor=None if ref.floor is None else round(-ref.floor.y, 3))]
    for w in room.walls:
        _widen(w.length, SYS_LENGTH, rel)
    _widen(room.ceiling_height, SYS_HEIGHT, rel)
    _widen(room.floor_area, SYS_AREA, 2 * rel)
    for o in room.openings:
        _widen(o.width, SYS_OPENING, rel)

    total = sum(e.length for e in ref.outline.edges)
    unseen = sum(e.length for e in ref.outline.edges if not e.supported)
    for w in room.walls:
        if w.evidence == "closure":  # a straight chord across an unobserved stretch is not a measurement of a wall
            w.length = Measurement(value=None, lo=None, hi=None, unit="m", method=w.length.method,
                                   note="wall not observed: the outline is closed by a straight chord")
    if total > 0 and unseen / total > MAX_UNSUPPORTED_FRAC:
        room.floor_area = Measurement(
            value=None, lo=None, hi=None, unit="m2", method=room.floor_area.method,
            note=f"outline incomplete: {100 * unseen / total:.0f}% of the perimeter was not observed",
        )
        flags.append("room_outline_incomplete")
    room.flags = list(dict.fromkeys(flags + room.flags))
    d = plan.diagnostics
    d.frames_used = None
    d.conventions, d.scale_method, d.scale_factor, d.scale_rel_sigma = CONVENTIONS, method, round(factor, 5), round(rel, 4)
    d.models, d.notes = models, list(room.flags)
    return plan


def _write(plan: CapturePlan, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, out / "plan.png")


def run_photo(
    capture: str | Path,
    out: str | Path,
    seed: int = 0,
    scale: float | None = None,
    scale_rel_sigma: float | None = None,
    depth: Depth | None = None,
    **_ignored,
) -> CapturePlan:
    """One plan per room folder of ``capture`` (written to ``out/rooms/<room>.json``) and the stitched property plan."""
    capture, out = Path(capture), Path(out)
    rooms = discover_rooms(capture)
    if not rooms:
        raise ValueError(f"no photos found under {capture}")
    models: list[str] = []
    if depth is None:
        from .models import DEPTH, REGISTRY

        depth = cached_depth(out / "work" / "depth")
        models = [f"{DEPTH}@{REGISTRY[DEPTH].revision[:10]}"]
    plans = []
    for name, paths in rooms.items():
        photos = load_photo_set(name, paths)
        poses = register_rotations(photos.images, seed=seed) if len(photos.images) >= 2 else Poses(
            [None] * len(photos.images), None, {}, None, ["insufficient_registration"])
        plan = plan_photo_room(name, photos, poses, depth, seed, scale, scale_rel_sigma, models)
        (out / "rooms").mkdir(parents=True, exist_ok=True)
        (out / "rooms" / f"{name}.json").write_text(plan.model_dump_json(indent=2))
        plans.append(plan)
    for k, plan in enumerate(plans):  # ids unique across the capture: room_k, r{k}_w*, r{k}_o*
        _renumber(plan.rooms[0], k)
    placed = [p for p in plans if p.rooms[0].polygon]
    result = stitch_plans(placed, capture.name) if placed else plans[0]
    # every room folder keeps its own plan: a room the stitcher could not place stays in its own frame, and a
    # room that could not be built stays as a null room
    kept = {r.id for r in result.rooms}
    result.rooms += [p.rooms[0] for p in plans if p.rooms[0].id not in kept]
    _write(result, out)
    return result


def _renumber(room: RoomPlan, k: int) -> None:
    room.id = f"room_{k}"
    for w in room.walls:
        w.id = w.id.replace("r0_", f"r{k}_", 1)
    for o in room.openings:
        o.id, o.wall_id = o.id.replace("r0_", f"r{k}_", 1), o.wall_id.replace("r0_", f"r{k}_", 1)
