"""Damage, concealed-damage flags and scope for one room, from posed frames: the driver shared by every tier.

A tier supplies the room's plan (in some frame), the floor and ceiling heights in that frame, and posed RGB frames
in that frame (``surfaces.Frame``). This module unrolls each surface, finds damage on it, converts the regions to
the schema in surface coordinates, and fills ``room.damage``, ``room.concealed_flags`` and ``room.scope``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

from .concealed import concealed_flags
from .damage import Found, Patch, detect
from .schema import DamageRegion, Measurement, RoomPlan, SurfaceRef
from .scope import scope_items
from .surfaces import MPP, Accumulator, Frame, Plane, surface_planes

Frames = Iterable[Frame] | Callable[[Plane], Iterable[Frame]]
MIN_SEEN = 0.30  # a surface with less than this share seen face on is reported as poorly covered
EXCLUDED_CLASSES: frozenset[str] = frozenset({"mould", "other_anomaly"})  # too false-positive-prone to report by default


def _meas(v: tuple[float, float, float], unit: str, method: str) -> Measurement:
    return Measurement(value=float(v[0]), lo=float(v[1]), hi=float(v[2]), unit=unit, method=method)


def _surface_xy(plane: Plane, kind: str, cols: np.ndarray, rows: np.ndarray, mpp: float) -> np.ndarray:
    """Pixel (col, row) to surface coordinates: wall (s, h); ceiling and floor plan (x, y)."""
    xy = np.stack([cols, rows], axis=-1).astype(float) * mpp
    if kind != "wall":
        xy = xy + np.array([plane.origin[0], -plane.origin[2]])
    return xy


def to_region(found: Found, ref: SurfaceRef, plane: Plane, room_id: str, index: int, mpp: float) -> DamageRegion:
    rows, cols = np.nonzero(found.mask)
    centre = _surface_xy(plane, ref.kind, np.array([cols.mean() + 0.5]), np.array([rows.mean() + 0.5]), mpp)[0]
    outline = _surface_xy(plane, ref.kind, np.array([p[0] for p in found.outline]), np.array([p[1] for p in found.outline]), mpp)
    method = "region cut at half its peak deviation; interval from the cut fraction, one pixel and the patch scale error"
    return DamageRegion(
        id=f"{room_id}_d{index}",
        surface=ref,
        damage_class=found.damage_class,  # type: ignore[arg-type]
        class_confidence=found.confidence,
        polygon=[(float(x), float(y)) for x, y in outline],
        centre=(float(centre[0]), float(centre[1])),
        width=_meas(found.width, "m", method),
        height=_meas(found.height, "m", method),
        length=_meas(found.length, "m", method),
        area=_meas(found.area, "m2", method),
        evidence=found.evidence,
    )


def write_debug(patch: Patch, found: list[Found], ref: SurfaceRef, path: Path) -> None:
    """One picture per surface: the unrolled colour patch (purple: not judged) beside its relief, regions outlined."""
    h, w = patch.valid.shape
    rgb = (np.clip(patch.rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    rgb[~patch.valid] = (rgb[~patch.valid] * 0.25 + np.array([120, 0, 80]) * 0.75).astype(np.uint8)
    if patch.relief is not None:
        rel = np.clip((np.nan_to_num(patch.relief) + 0.06) / 0.12, 0, 1)
        rel = cv2.applyColorMap((rel * 255).astype(np.uint8), cv2.COLORMAP_JET)
        rel[~np.isfinite(patch.relief)] = 0
    else:
        rel = np.zeros_like(rgb)
    img = cv2.flip(np.hstack([rgb, rel]), 0)  # wall height upward
    for f in found:
        pts = np.array([(x, h - 1 - y) for x, y in f.outline], np.int32)
        for off in (0, w):
            cv2.polylines(img, [pts + [off, 0]], True, (0, 255, 255), 2)
        x0, y0 = pts.min(axis=0)
        cv2.putText(img, f.damage_class.replace("_", " "), (int(x0), max(12, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    scale = min(1.0, 1600 / img.shape[1])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def assess_room(
    room: RoomPlan,
    floor_y: float,
    ceiling_y: float | None,
    frames: Frames,
    others: Sequence[RoomPlan] = (),
    kinds: Sequence[str] = ("wall", "ceiling", "floor"),
    mpp: float = MPP,
    min_views: int = 2,
    use_relief: bool = True,
    scale_rel_sigma: float | None = None,
    report_unclassified: bool = True,
    debug_dir: Path | None = None,
    exclude_classes: Iterable[str] = EXCLUDED_CLASSES,
) -> list[str]:
    """Fill ``room.damage``, ``room.concealed_flags`` and ``room.scope``; returns notes on what could not be judged.

    ``frames`` is either an iterable of frames (one pass feeds every surface) or a function from a surface's plane
    to the frames to use for it. ``kinds`` names the surface kinds to assess. ``scale_rel_sigma`` is the relative
    error of the frames' metric scale (a LiDAR walk: the default of a few percent). ``report_unclassified=False``
    drops regions that fit no damage class, for tiers without depth to tell furniture from the wall.
    ``debug_dir`` gets one picture per surface (``write_debug``), drawn after ``exclude_classes`` is applied, so an
    excluded class never appears there, in ``room.damage``, or (via it) on the rendered plan.
    """
    notes: list[str] = []
    planes = [(ref, pl) for ref, pl in surface_planes(room, floor_y, ceiling_y, mpp) if ref.kind in kinds]
    if ceiling_y is None:
        notes.append("no ceiling was observed: walls are judged up to 2.0 m and the ceiling is not judged")
    damage: list[DamageRegion] = []

    def finish(ref: SurfaceRef, plane: Plane, acc: Accumulator) -> None:
        patch = acc.patch()
        patch.kind = ref.kind
        seen = float(patch.valid.sum() / max(1, plane.inside.sum() if plane.inside is not None else patch.valid.size))
        if seen < MIN_SEEN:
            notes.append(f"{ref.id}: only {seen * 100:.0f}% of the surface was seen face on and unoccluded, damage elsewhere on it is unknown")
        found = [f for f in detect(patch, scale_rel_sigma, report_unclassified) if f.damage_class not in exclude_classes]
        if debug_dir is not None:
            write_debug(patch, found, ref, Path(debug_dir) / f"{ref.id}.png")
        for f in found:
            damage.append(to_region(f, ref, plane, room.id, len(damage), mpp))

    if callable(frames):
        for ref, plane in planes:
            acc = Accumulator(plane, min_views, use_relief)
            for fr in frames(plane):
                acc.add(fr)
            finish(ref, plane, acc)
    else:
        accs = [Accumulator(pl, min_views, use_relief) for _r, pl in planes]
        for fr in frames:
            for acc in accs:
                acc.add(fr)
        for (ref, plane), acc in zip(planes, accs):
            finish(ref, plane, acc)
    room.damage = damage
    room.concealed_flags = concealed_flags(room, damage, list(others))
    room.scope = scope_items(room, damage, room.concealed_flags)
    return notes
