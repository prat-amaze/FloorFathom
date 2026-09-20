"""Scope line items: one repair action per damage region, plus an opening-up item where damage may be concealed.

The action for each class and the quantity rule are our own table (the brief gives none). Quantities are the
damaged area in m2, or the length in m for a crack, and keep the damage measurement's interval.
"""

from __future__ import annotations

from .schema import ConcealedFlag, DamageRegion, Measurement, RoomPlan, ScopeItem

ACTIONS = {
    "water_stain": ("Find and stop the moisture source, dry, seal and repaint", "m2"),
    "mould": ("Treat and clean the mould, then seal and repaint", "m2"),
    "structural_crack": ("Rout, fill and repaint the crack, and check its cause", "m"),
    "peeling_paint": ("Scrape back to sound paint, prime and repaint", "m2"),
    "soot_or_fire": ("Clean or seal the soot, then repaint", "m2"),
    "efflorescence": ("Brush off the salts, treat the wall and repaint", "m2"),
    "hole_or_impact": ("Patch the hole and repaint", "m2"),
    "sagging_or_bulging": ("Repair or replace the board or plaster and refinish", "m2"),
    "other_anomaly": ("Inspect the mark on site", "m2"),
}
INSPECT = "Open up the surface and inspect behind it"


def _quantity(d: DamageRegion, unit: str) -> Measurement:
    src = d.length if unit == "m" else d.area
    return Measurement(
        value=src.value,
        lo=src.lo,
        hi=src.hi,
        unit=unit,
        method="length of the region's oriented box" if unit == "m" else "area of the damage region",
    )


def scope_items(room: RoomPlan, damage: list[DamageRegion], flags: list[ConcealedFlag]) -> list[ScopeItem]:
    items: list[ScopeItem] = []
    for d in damage:
        action, unit = ACTIONS[d.damage_class]
        items.append(ScopeItem(id=f"{room.id}_s{len(items)}", surface=d.surface, damage_ids=[d.id], action=action, quantity=_quantity(d, unit)))
    by_id = {d.id: d for d in damage}
    flagged: dict[tuple[str, str], list[str]] = {}
    for f in flags:
        flagged.setdefault((f.surface.kind, f.surface.id), []).extend(f.damage_ids)
    for (kind, sid), ids in flagged.items():
        ids = sorted(set(ids))
        regions = [by_id[i] for i in ids if i in by_id and by_id[i].surface.id == sid]
        if not regions:
            continue
        vals = [r.area for r in regions]
        qty = Measurement(
            value=sum(v.value for v in vals if v.value is not None),
            lo=sum(v.lo for v in vals if v.lo is not None),
            hi=sum(v.hi for v in vals if v.hi is not None),
            unit="m2",
            method="summed area of the damage regions with a concealed-damage flag on this surface",
        )
        items.append(ScopeItem(id=f"{room.id}_s{len(items)}", surface=regions[0].surface, damage_ids=[r.id for r in regions], action=INSPECT, quantity=qty))
    return items
