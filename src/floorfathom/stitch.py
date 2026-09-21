"""Whole-property checks on placed rooms: which room each doorway opens onto, whether two
rooms occupy the same floor, and how much floor the property covers.

These work on the room polygons and openings of a ``CapturePlan`` whatever tier made them,
as long as the rooms share one frame (``frame="capture"``). Placing rooms that come in their
own frames is a separate step.

A doorway opens onto the nearest other room, so the distance from the doorway to that room's
outline is the evidence for the link: a wall's thickness (10 to 30 cm) is what a real link
looks like, a metre is a doubtful one. Polygons are the inner faces of the walls, so two
neighbouring rooms do not overlap; the wall between them is a gap. Areas are rasterised (no
shapely needed) by testing cell centres, so the boundary is off by at most half a cell, 1 cm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from .schema import CapturePlan, Diagnostics, Link, Measurement, Overlap, Placement, RoomPlan, Stitching

MAX_GAP = 1.5  # metres: a doorway further than this from every other room leads outside
MUTUAL = 0.5  # metres: the other room has its own doorway this close to ours
CELL = 0.02  # metres, raster cell for areas


def _dist_to_outline(p: np.ndarray, poly: np.ndarray) -> float:
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(axis=1) / (ab**2).sum(axis=1), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min())


def adjacency(rooms: list[RoomPlan], max_gap: float = MAX_GAP) -> list[Link]:
    """One link per doorway: the nearest other room outline within ``max_gap``."""
    polys = {r.id: np.asarray(r.polygon, float) for r in rooms}
    links = []
    for room in rooms:
        for op in room.openings:
            c = np.asarray(op.centre, float)
            near = sorted((_dist_to_outline(c, p), rid) for rid, p in polys.items() if rid != room.id)
            if not near or near[0][0] > max_gap:
                links.append(Link(room=room.id, opening=op.id, other=None, gap=None, mutual=False))
                continue
            gap, other = near[0]
            theirs = next(r for r in rooms if r.id == other)
            mutual = any(np.linalg.norm(np.asarray(o.centre, float) - c) <= MUTUAL for o in theirs.openings)
            links.append(Link(room=room.id, opening=op.id, other=other, gap=gap, mutual=mutual))
    return links


def _mask(polys: list[np.ndarray], origin: np.ndarray, shape: tuple[int, int], cell: float) -> np.ndarray:
    """Cells whose centre lies inside a polygon (unbiased, unlike filling boundary pixels)."""
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    centres = origin + (np.stack([cols, rows], axis=-1).reshape(-1, 2) + 0.5) * cell
    m = np.zeros(len(centres), bool)
    for p in polys:
        m |= Path(p).contains_points(centres)
    return m.reshape(shape)


def _canvas(polys: list[np.ndarray], cell: float) -> tuple[np.ndarray, tuple[int, int]]:
    allp = np.concatenate(polys)
    origin = allp.min(axis=0) - cell
    size = np.ceil((allp.max(axis=0) - origin) / cell).astype(int) + 2
    return origin, (int(size[1]), int(size[0]))


def overlap_area(a: RoomPlan, b: RoomPlan, cell: float = CELL) -> float:
    """Square metres of floor that both rooms claim."""
    pa, pb = np.asarray(a.polygon, float), np.asarray(b.polygon, float)
    origin, shape = _canvas([pa, pb], cell)
    return float((_mask([pa], origin, shape, cell) & _mask([pb], origin, shape, cell)).sum() * cell**2)


def overlaps(rooms: list[RoomPlan], min_area: float = 0.1) -> list[tuple[str, str, float]]:
    """Room pairs that share more than ``min_area`` m2 of floor, largest first."""
    found = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            area = overlap_area(a, b)
            if area > min_area:
                found.append((a.id, b.id, area))
    return sorted(found, key=lambda t: -t[2])


def union_area(polys: list[np.ndarray]) -> float:
    """Square metres covered by the polygons together, overlaps counted once."""
    if not polys:
        return 0.0
    origin, shape = _canvas(polys, CELL)
    return float(_mask(polys, origin, shape, CELL).sum() * CELL**2)


def footprint_area(rooms: list[RoomPlan]) -> float:
    """Square metres of floor covered by the rooms together, overlaps counted once."""
    return union_area([np.asarray(r.polygon, float) for r in rooms])


# ---- placing rooms that arrive in their own frames ------------------------------------

WALL = 0.15  # metres, assumed thickness of the wall between two rooms (not measured)
MAX_OVERLAP = 0.25  # m2 a placed room may share with the rooms already down before it is rejected
MAX_WIDTH_DIFF = 0.3  # metres: the two sides of one doorway differ by more than this
AMBIGUOUS = 0.1  # cost margin below which the runner-up placement is about as good
SEARCH_CELL = 0.05  # metres, raster cell while searching (coarser than CELL, much faster)


@dataclass
class Stitched:
    rooms: list[RoomPlan]  # placed rooms, all in one frame
    placements: list[Placement]
    unplaced: list[str]  # rooms with no doorway that fits anywhere


def _inward_normal(room: RoomPlan, c: np.ndarray) -> np.ndarray:
    """Unit vector into the room, perpendicular to the polygon edge nearest to ``c``.

    The polygon is counter-clockwise, so the interior is on the left of every edge.
    """
    a = np.asarray(room.polygon, float)
    ab = np.roll(a, -1, axis=0) - a
    t = np.clip(((c - a) * ab).sum(axis=1) / (ab**2).sum(axis=1), 0.0, 1.0)
    i = int(np.linalg.norm(a + t[:, None] * ab - c, axis=1).argmin())
    d = ab[i] / np.linalg.norm(ab[i])
    return np.array([-d[1], d[0]])


def _rot(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def transform_room(room: RoomPlan, angle: float, shift: np.ndarray) -> RoomPlan:
    """The room rotated by ``angle`` about the origin and then moved by ``shift``, in the capture frame."""
    r = _rot(angle)

    def pt(p) -> tuple[float, float]:
        q = r @ np.asarray(p, float) + shift
        return float(q[0]), float(q[1])

    return room.model_copy(
        update={
            "frame": "capture",
            "polygon": [pt(p) for p in room.polygon],
            "walls": [w.model_copy(update={"start": pt(w.start), "end": pt(w.end)}) for w in room.walls],
            "openings": [
                o.model_copy(update={"start": pt(o.start), "end": pt(o.end), "centre": pt(o.centre)})
                for o in room.openings
            ],
            "stations": [s.model_copy(update={"position": pt(s.position)}) for s in room.stations],
        }
    )


def _candidates(new: RoomPlan, placed: dict[str, RoomPlan], used: set[tuple[str, str]], wall: float):
    """Every way to glue one of ``new``'s doorways onto a free doorway of a placed room.

    Gluing fixes the pose completely: the doorway's inward normals must be opposite (the rooms lie
    on either side of the shared wall) and its centre sits one wall thickness beyond the host's.
    Yields (cost, placement, moved room); cost is overlap plus width mismatch, lower is better.
    """
    for host in placed.values():
        for ho in host.openings:
            if (host.id, ho.id) in used:
                continue
            c_host = np.asarray(ho.centre, float)
            n_host = _inward_normal(host, c_host)
            for own in new.openings:
                c_own = np.asarray(own.centre, float)
                n_own = _inward_normal(new, c_own)
                angle = np.arctan2(-n_host[1], -n_host[0]) - np.arctan2(n_own[1], n_own[0])
                shift = c_host - n_host * wall - _rot(angle) @ c_own
                moved = transform_room(new, angle, shift)
                overlap = sum(overlap_area(moved, p, SEARCH_CELL) for p in placed.values())
                wd = None
                if own.width.value is not None and ho.width.value is not None:
                    wd = abs(own.width.value - ho.width.value)
                if overlap > MAX_OVERLAP or (wd is not None and wd > MAX_WIDTH_DIFF):
                    continue
                cost = overlap + (wd or 0.0)
                yield cost, Placement(
                    room=new.id, angle=float(angle), shift=(float(shift[0]), float(shift[1])), host=host.id,
                    host_opening=ho.id, opening=own.id, overlap=overlap, width_diff=wd, margin=None,
                ), moved


def stitch(rooms: list[RoomPlan], wall: float = WALL) -> Stitched:
    """Place every room in one frame by gluing doorways together, most certain room first.

    Rooms already in the capture frame stay where they are. If there are none, the room with
    the most doorways is the root and keeps its own frame. Each round, every unplaced room
    finds its best doorway match against the rooms placed so far; the room whose best match
    beats its runner-up by the most goes down next, so a confident placement constrains the
    ambiguous ones. A room that cannot be placed is reported, not guessed.
    """
    fixed = [r for r in rooms if r.frame == "capture"]
    todo = [r for r in rooms if r.frame != "capture"]
    placed: dict[str, RoomPlan] = {r.id: r for r in fixed}
    placements: list[Placement] = []
    if not placed and todo:
        root = max(todo, key=lambda r: (len(r.openings), r.floor_area.value or 0.0))
        todo.remove(root)
        placed[root.id] = root.model_copy(update={"frame": "capture"})
        placements.append(
            Placement(room=root.id, angle=0.0, shift=(0.0, 0.0), host=None, host_opening=None, opening=None,
                      overlap=0.0, width_diff=None, margin=None)
        )
    used: set[tuple[str, str]] = set()
    unplaced: list[str] = []
    while todo:
        best = None  # (margin, room, cost, placement, moved)
        for room in todo:
            found = sorted(_candidates(room, placed, used, wall), key=lambda t: t[0])
            if not found:
                continue
            margin = found[1][0] - found[0][0] if len(found) > 1 else np.inf
            if best is None or margin > best[0]:
                best = (margin, room, *found[0])
        if best is None:
            unplaced = [r.id for r in todo]
            break
        margin, room, _, pl, moved = best
        pl.margin = None if np.isinf(margin) else float(margin)
        if pl.margin is not None and pl.margin < AMBIGUOUS:
            moved = moved.model_copy(update={"flags": [*moved.flags, "placement_ambiguous"]})
        placed[room.id] = moved
        used.add((pl.host, pl.host_opening))
        used.add((room.id, pl.opening))
        placements.append(pl)
        todo.remove(room)
    order = {r.id: i for i, r in enumerate(rooms)}
    return Stitched(sorted(placed.values(), key=lambda r: order[r.id]), placements, unplaced)


DRIFT_PLACED = (
    "No pose chain links the rooms: each room is reconstructed in its own frame and placed by its "
    "doorways, so drift cannot accumulate from room to room; a placement error enters only through "
    "the doorway position and the assumed wall thickness. Drift within one room's own walk is not corrected."
)

CORR_CENTRE_TOL, CORR_NORMAL_TOL_DEG = 0.15, 8.0


def _corr_candidates(rooms):
    """Every pair of doorways on two different placed rooms whose centres and opposed normals nearly agree."""
    out = []
    for i, ra in enumerate(rooms):
        for oa in ra.openings:
            na = _inward_normal(ra, np.asarray(oa.centre, float))
            for rb in rooms[i + 1:]:
                for ob in rb.openings:
                    nb = _inward_normal(rb, np.asarray(ob.centre, float))
                    d = float(np.linalg.norm(np.asarray(oa.centre, float) - np.asarray(ob.centre, float)))
                    ang = float(np.degrees(np.arccos(np.clip(-na @ nb, -1, 1))))
                    if d <= CORR_CENTRE_TOL and ang <= CORR_NORMAL_TOL_DEG:
                        out.append((ra.id, rb.id, np.asarray(oa.centre, float), np.asarray(ob.centre, float)))
    return out


def refine_global(rooms, placements, wall: float = WALL):
    """Jointly refine every room's (angle, shift) over all doorway correspondences at once, initialized from
    the sequential solve. Falls back to ``placements`` unchanged when there are not enough non-outlier
    correspondences to over-determine the system."""
    from scipy.optimize import least_squares

    by_id = {p.room: p for p in placements}
    ids = [r.id for r in rooms if r.id in by_id]
    if len(ids) < 2:
        return placements
    corr = _corr_candidates(rooms)
    dof = 3 * (len(ids) - 1)
    if len(corr) * 2 < dof + 2:
        return placements

    root = ids[0]
    var_ids = [i for i in ids if i != root]
    x0 = np.concatenate([[by_id[i].angle, *by_id[i].shift] for i in var_ids])
    index = {i: k for k, i in enumerate(var_ids)}

    def pose(i, x):
        if i == root:
            return 0.0, np.zeros(2)
        a, sx, sy = x[3 * index[i]:3 * index[i] + 3]
        return a, np.array([sx, sy])

    def residuals(x):
        out = []
        for ida, idb, ca, cb in corr:
            aa, sa = pose(ida, x)
            ab, sb = pose(idb, x)
            wa = _rot(aa) @ ca + sa
            wb = _rot(ab) @ cb + sb
            out.extend((wa - wb).tolist())
        return np.array(out)

    fit = least_squares(residuals, x0, loss="soft_l1", f_scale=CORR_CENTRE_TOL)
    resid1 = fit.fun.reshape(-1, 2)
    kept_mask = np.linalg.norm(resid1, axis=1) <= max(CORR_CENTRE_TOL, 3 * np.median(np.linalg.norm(resid1, axis=1)))
    if not kept_mask.all():
        corr = [c for c, k in zip(corr, kept_mask) if k]
        if len(corr) * 2 < dof + 2:
            return placements
        fit = least_squares(residuals, x0, loss="soft_l1", f_scale=CORR_CENTRE_TOL)

    out = []
    for p in placements:
        if p.room == root or p.room not in index:
            out.append(p)
            continue
        a, s = pose(p.room, fit.x)
        out.append(p.model_copy(update={"angle": float(a), "shift": (float(s[0]), float(s[1]))}))
    return out


def describe(rooms: list[RoomPlan], placements: list[Placement], unplaced: list[str], drift: str) -> Stitching:
    """Adjacency, overlaps and footprint of rooms that are already in one frame."""
    area = footprint_area(rooms)
    with_area = [r.floor_area for r in rooms if r.floor_area.value]
    total = sum(m.value for m in with_area)
    # room errors are taken as shared (one scale for all), the conservative choice, so the union scales together
    lo = area * sum(m.lo for m in with_area) / total if with_area and all(m.lo is not None for m in with_area) else None
    hi = area * sum(m.hi for m in with_area) / total if with_area and all(m.hi is not None for m in with_area) else None
    footprint = Measurement(
        value=area, lo=lo, hi=hi, unit="m2",
        method="union of the room polygons, overlaps counted once; interval scales it by the rooms' summed area interval",
        note="inner faces of the walls: wall thickness between rooms is not included",
    )
    return Stitching(
        footprint=footprint,
        links=adjacency(rooms),
        placements=placements,
        overlaps=[Overlap(a=a, b=b, area=v) for a, b, v in overlaps(rooms)],
        unplaced=unplaced,
        drift=drift,
    )


def stitch_plans(plans: list[CapturePlan], name: str) -> CapturePlan:
    """One property plan from per-room plans (one capture of one room each), placed by their doorways.

    All plans must come from one tier: a property is stitched from one kind of capture, never a mix.
    """
    if len({p.tier for p in plans}) != 1:
        raise ValueError(f"plans from different tiers cannot be stitched together: {sorted({p.tier for p in plans})}")
    rooms: list[RoomPlan] = []
    for plan in plans:
        for r in plan.rooms:
            rid = r.id if all(r.id != o.id for o in rooms) else f"{plan.capture}_{r.id}"
            rooms.append(r.model_copy(update={"id": rid, "name": r.name or plan.capture}))
    out = stitch(rooms)
    placements = refine_global(out.rooms, out.placements)
    out = Stitched(out.rooms, placements, out.unplaced)
    diag = plans[0].diagnostics
    notes = [n for p in plans for n in p.diagnostics.notes]
    return CapturePlan(
        capture=name,
        tier=plans[0].tier,
        rooms=out.rooms,
        stitching=describe(out.rooms, out.placements, out.unplaced, DRIFT_PLACED),
        diagnostics=Diagnostics(
            bootstrap_replicates=diag.bootstrap_replicates,
            seed=diag.seed,
            seconds=sum(p.diagnostics.seconds for p in plans),
            conventions=diag.conventions,
            models=sorted({m for p in plans for m in p.diagnostics.models}),
            notes=[*notes, *[f"room {u} could not be placed: no doorway fits" for u in out.unplaced]],
        ),
    )
