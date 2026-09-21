"""Assemble the JSON contract from an estimate and its bootstrap samples."""

from __future__ import annotations

import numpy as np

from . import layout as L
from .estimate import Estimate, RoomEst
from .schema import CapturePlan, Diagnostics, Measurement, Opening, RoomPlan, Wall
from .uncertainty import (
    SYS_AREA_REL,
    SYS_HEIGHT,
    SYS_LENGTH,
    SYS_OPENING,
    RoomSamples,
    interval,
)


def _plan(p: np.ndarray) -> tuple[float, float]:
    """World (x, z) to plan (x, y): y is flipped so the drawing is not mirrored."""
    return (round(float(p[0]), 4), round(float(-p[1]), 4))


def _measure(value: float, samples: list[float], sys: tuple[float, float], unit: str, method: str, scale: float):
    lo, hi, stable = interval(value, samples, sys[0], sys[1], scale)
    return Measurement(value=round(value, 4), lo=round(lo, 4), hi=round(hi, 4), unit=unit, method=method), stable


def room_plan(idx: int, room: RoomEst, samples: RoomSamples, replicates: int, scale: float) -> RoomPlan:
    flags = list(room.flags)
    unstable = set()
    edges = room.outline.edges
    n = len(edges)

    # final order is counter-clockwise in plan coordinates (world order is clockwise there)
    walls: list[Wall] = []
    final_of_edge: dict[int, int] = {}
    for f in range(n):
        i = n - 1 - f
        e = edges[i]
        final_of_edge[i] = f
        m, ok = _measure(
            e.length,
            samples.wall_length.get(i, []),
            SYS_LENGTH,
            "m",
            "distance between corners; walls fitted by total least squares to wall points",
            scale,
        )
        if not ok:
            unstable.add("walls")
        walls.append(
            Wall(
                id=f"r{idx}_w{f}",
                start=_plan(e.p1),
                end=_plan(e.p0),
                length=m,
                evidence="wall_points" if e.supported else "closure",
            )
        )
    polygon = [w.start for w in walls]

    area_m, ok = _measure(
        room.area,
        samples.area,
        (0.05, SYS_AREA_REL),
        "m2",
        "shoelace formula on the room polygon",
        scale,
    )
    if not ok:
        unstable.add("area")

    ch = room.ceiling_height
    if ch is None:
        ceiling = Measurement(
            value=None,
            lo=None,
            hi=None,
            unit="m",
            method="ceiling plane minus floor plane",
            note="no ceiling plane observed in this capture; not estimated",
        )
    else:
        ceiling, ok = _measure(
            ch, samples.ceiling_height, SYS_HEIGHT, "m", "ceiling plane minus floor plane, from height-histogram peaks", scale
        )
        if not ok:
            unstable.add("ceiling")

    openings = []
    for j, o in enumerate(room.openings):
        m, ok = _measure(
            o.width, samples.opening_width.get(j, []), SYS_OPENING, "m", "gap between the two wall ends of a doorway", scale
        )
        if not ok:
            unstable.add("openings")
        openings.append(
            Opening(
                id=f"r{idx}_o{j}",
                kind="doorway",
                wall_id=f"r{idx}_w{final_of_edge[o.edge_index]}",
                start=_plan(o.p1),
                end=_plan(o.p0),
                centre=_plan(o.centre),
                width=m,
                note="floor-level gap in wall evidence; door height and windows are not measured at this tier",
            )
        )
    for what in sorted(unstable):
        flags.append(f"unstable_interval_{what}")
    if samples.matched < max(3, int(0.6 * replicates)):
        flags.append("room_not_reproduced_in_bootstrap")
    return RoomPlan(
        id=f"room_{idx}",
        polygon=polygon,
        walls=walls,
        ceiling_height=ceiling,
        floor_area=area_m,
        openings=openings,
        flags=flags,
    )


def capture_plan(
    name: str,
    tier: str,
    est: Estimate,
    samples: list[RoomSamples],
    replicates: int,
    seed: int,
    frames_used: int,
    n_points: int,
    seconds: float,
    floor_sharpness: float | None,
    scale: float,
) -> CapturePlan:
    rooms = [room_plan(k, r, samples[k], replicates, scale) for k, r in enumerate(est.rooms)]
    return CapturePlan(
        capture=name,
        tier=tier,  # type: ignore[arg-type]
        rooms=rooms,
        diagnostics=Diagnostics(
            frames_used=frames_used,
            points=n_points,
            floor_height_world=None if est.floor is None else round(est.floor.y, 4),
            floor_sharpness=None if floor_sharpness is None else round(floor_sharpness, 4),
            bootstrap_replicates=replicates,
            seed=seed,
            seconds=round(seconds, 2),
            conventions="depth as-is; OpenCV camera axes; quaternion camera-to-world; world +y up",
        ),
    )


def polygon_area(poly) -> float:
    return abs(L.signed_area(np.asarray(poly, dtype=float)))
