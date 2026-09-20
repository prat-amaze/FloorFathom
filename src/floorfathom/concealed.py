"""Concealed-damage flags: damage that probably continues out of sight, and the named rule that says so.

The rules are our own assumptions about how damage behaves (moisture travels, cracks continue behind finishes),
not a published standard. Each flag carries the rule that fired, so an inspector can see why it is there and
disagree with it.
"""

from __future__ import annotations

import numpy as np

from .schema import ConcealedFlag, DamageRegion, RoomPlan, SurfaceRef

LONG_CRACK = 0.5  # m: a crack this long is more than a surface flaw
LOW_ON_WALL = 0.6  # m above the floor: damage this low points to moisture rising or pooling
LEAK_PATH = 0.8  # m in plan between a ceiling stain and a wall stain that share a leak path
SHARED_WALL = 0.4  # m between a wall and another room's outline


def _point_on_wall(room: RoomPlan, d: DamageRegion) -> np.ndarray | None:
    w = next((w for w in room.walls if w.id == d.surface.id), None)
    if w is None:
        return None
    a, b = np.array(w.start), np.array(w.end)
    ln = np.linalg.norm(b - a)
    return a + (b - a) / max(ln, 1e-9) * d.centre[0]


def _dist_to_outline(p: np.ndarray, polygon: list[tuple[float, float]]) -> float:
    pts = np.array(polygon)
    best = np.inf
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        ab = b - a
        t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def concealed_flags(room: RoomPlan, damage: list[DamageRegion], others: list[RoomPlan]) -> list[ConcealedFlag]:
    """Flags for one room's damage. ``others`` are the rooms next to it, used for the shared-wall rule."""
    flags: list[tuple[str, SurfaceRef, list[str], str]] = []

    def add(rule: str, d: DamageRegion, reason: str, extra: list[DamageRegion] | None = None) -> None:
        ids = [d.id] + [e.id for e in extra or []]
        flags.append((rule, d.surface, ids, reason))

    ceiling_stains = [d for d in damage if d.surface.kind == "ceiling" and d.damage_class == "water_stain"]
    for d in damage:
        c = d.damage_class
        if c == "water_stain" and d.surface.kind == "ceiling":
            add("ceiling_stain_source_above", d, "a water stain on a ceiling comes from the ceiling void or the floor above")
        if c == "water_stain" and d.surface.kind == "wall":
            p = _point_on_wall(room, d)
            if p is not None:
                near = [s for s in ceiling_stains if np.linalg.norm(np.array(s.centre) - p) <= LEAK_PATH]
                if near:
                    add("wall_stain_under_ceiling_stain", d, "the wall stain lies under a ceiling stain, so water has run down inside the wall", near)
                for o in others:
                    if o.polygon and _dist_to_outline(p, o.polygon) <= SHARED_WALL:
                        add("stain_on_shared_wall", d, f"the stained wall is shared with {o.id}, so the source may be behind it")
                        break
        if c in ("water_stain", "mould", "efflorescence") and d.surface.kind == "wall" and d.centre[1] <= LOW_ON_WALL:
            add("moisture_low_on_wall", d, f"{c.replace('_', ' ')} within {LOW_ON_WALL:.1f} m of the floor points to moisture rising or pooling behind the skirting")
        if c == "mould":
            add("mould_moisture_behind_surface", d, "mould needs sustained moisture, which is likely still present behind the surface")
        if c in ("efflorescence", "peeling_paint"):
            add("moisture_migrating_through_coating", d, "salts or lifting paint show moisture moving through the wall behind the coating")
        if c == "structural_crack" and d.length.value is not None and d.length.value >= LONG_CRACK:
            add("long_crack_continues_behind_finish", d, f"a crack of {d.length.value * 100:.0f} cm or more usually continues behind plaster or a finish")
        if c == "sagging_or_bulging" and d.surface.kind == "ceiling":
            add("ceiling_sag_water_or_load", d, "a sagging ceiling is carrying water or load above it")
        if c == "sagging_or_bulging" and d.surface.kind == "wall":
            add("wall_bulge_moisture_or_movement", d, "a bulging wall hides moisture or movement behind the surface")
        if c == "soot_or_fire":
            add("soot_char_behind_surface", d, "soot on a surface means the material behind it may be charred or contaminated")
    out, seen = [], set()
    for rule, surface, ids, reason in flags:
        key = (rule, tuple(ids))
        if key in seen:
            continue
        seen.add(key)
        out.append(ConcealedFlag(id=f"{room.id}_c{len(out)}", rule=rule, surface=surface, damage_ids=ids, reason=reason))
    return out
