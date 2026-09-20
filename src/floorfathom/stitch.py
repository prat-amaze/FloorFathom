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

from .schema import RoomPlan

MAX_GAP = 1.5  # metres: a doorway further than this from every other room leads outside
MUTUAL = 0.5  # metres: the other room has its own doorway this close to ours
CELL = 0.02  # metres, raster cell for areas


@dataclass
class Link:
    room: str
    opening: str
    other: str | None  # None: no other room within MAX_GAP (outside, or a room not captured)
    gap: float | None  # metres from the opening's centre to the other room's outline
    mutual: bool  # the other room has a doorway of its own next to this one


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
                links.append(Link(room.id, op.id, None, None, False))
                continue
            gap, other = near[0]
            theirs = next(r for r in rooms if r.id == other)
            mutual = any(np.linalg.norm(np.asarray(o.centre, float) - c) <= MUTUAL for o in theirs.openings)
            links.append(Link(room.id, op.id, other, gap, mutual))
    return links


def _mask(polys: list[np.ndarray], origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Cells whose centre lies inside a polygon (unbiased, unlike filling boundary pixels)."""
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    centres = origin + (np.stack([cols, rows], axis=-1).reshape(-1, 2) + 0.5) * CELL
    m = np.zeros(len(centres), bool)
    for p in polys:
        m |= Path(p).contains_points(centres)
    return m.reshape(shape)


def _canvas(polys: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
    allp = np.concatenate(polys)
    origin = allp.min(axis=0) - CELL
    size = np.ceil((allp.max(axis=0) - origin) / CELL).astype(int) + 2
    return origin, (int(size[1]), int(size[0]))


def overlap_area(a: RoomPlan, b: RoomPlan) -> float:
    """Square metres of floor that both rooms claim."""
    pa, pb = np.asarray(a.polygon, float), np.asarray(b.polygon, float)
    origin, shape = _canvas([pa, pb])
    return float((_mask([pa], origin, shape) & _mask([pb], origin, shape)).sum() * CELL**2)


def overlaps(rooms: list[RoomPlan], min_area: float = 0.1) -> list[tuple[str, str, float]]:
    """Room pairs that share more than ``min_area`` m2 of floor, largest first."""
    found = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            area = overlap_area(a, b)
            if area > min_area:
                found.append((a.id, b.id, area))
    return sorted(found, key=lambda t: -t[2])


def footprint_area(rooms: list[RoomPlan]) -> float:
    """Square metres of floor covered by the rooms together, overlaps counted once."""
    polys = [np.asarray(r.polygon, float) for r in rooms]
    origin, shape = _canvas(polys)
    return float(_mask(polys, origin, shape).sum() * CELL**2)
