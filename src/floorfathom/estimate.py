"""Point cloud to per-room estimates (no intervals yet, see uncertainty.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import layout as L
from .planes import HeightPlane, find_ceiling, find_floor


@dataclass
class Params:
    cov_fraction: float = 0.58  # a wall cell is occupied in this fraction of the height bands
    max_gap: float = 1.1  # widest doorway closed by extending walls (m)
    min_traj_cells: int = 10  # camera cells a piece of free space needs to count as a room
    min_room_area: float = 2.0
    min_opening: float = 0.55
    max_opening: float = 1.8
    ceiling_footprint: float = 0.15  # fraction of the room a ceiling candidate must cover
    outline_eps: float = 0.12  # polygon simplification tolerance (m)


@dataclass
class OpeningEst:
    p0: np.ndarray  # metres, world (x, z)
    p1: np.ndarray
    edge_index: int  # index of the room edge that carries the opening

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)


@dataclass
class RoomEst:
    outline: L.RoomOutline
    area: float
    floor: HeightPlane | None
    ceiling: HeightPlane | None
    openings: list[OpeningEst] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ceiling_height(self) -> float | None:
        if self.floor is None or self.ceiling is None:
            return None
        return self.ceiling.y - self.floor.y


@dataclass
class Estimate:
    rooms: list[RoomEst]
    grid: L.Grid
    floor: HeightPlane | None
    closures: list = field(default_factory=list)
    barrier: np.ndarray | None = None
    free: np.ndarray | None = None


def make_grid(points: np.ndarray, traj_xz: np.ndarray) -> L.Grid:
    lo = np.percentile(points[:, [0, 2]], 0.1, axis=0)
    hi = np.percentile(points[:, [0, 2]], 99.9, axis=0)
    allxz = np.vstack([lo, hi, traj_xz.min(axis=0), traj_xz.max(axis=0)])
    return L.Grid.around(allxz, margin=0.5)


def _room_ceiling(points: np.ndarray, grid: L.Grid, mask: np.ndarray, floor: HeightPlane, min_frac: float):
    m_d = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ix, iz = grid.index(points[:, [0, 2]])
    ok = grid.inside(ix, iz)
    sel = np.zeros(len(points), dtype=bool)
    sel[ok] = m_d[iz[ok], ix[ok]]
    pts = points[sel]
    n_cells = max(1, int(mask.sum()))

    def footprint_ok(y0: float) -> bool:
        near = np.abs(pts[:, 1] - y0) <= 0.05
        jx, jz = grid.index(pts[near][:, [0, 2]])
        ok2 = grid.inside(jx, jz)
        cells = np.zeros(mask.shape, dtype=np.int32)
        np.add.at(cells, (jz[ok2], jx[ok2]), 1)
        return float(((cells >= 2) & mask).sum()) / n_cells >= min_frac

    room_floor = find_floor(pts[:, 1]) if len(pts) > 500 else None
    # a room's own floor spike is trusted only if it is close to the global floor
    if room_floor is None or abs(room_floor.y - floor.y) > 0.3:
        room_floor = floor
    ceiling = find_ceiling(pts[:, 1], room_floor.y, footprint_ok) if len(pts) > 500 else None
    return room_floor, ceiling


def estimate(
    points: np.ndarray,
    traj_xz: np.ndarray,
    params: Params | None = None,
    grid: L.Grid | None = None,
    free_hint: np.ndarray | None = None,
    walls: Callable[[np.ndarray], list[L.WallLine]] | None = None,
) -> Estimate:
    """``free_hint``: optional boolean mask on ``grid`` of cells known to be open floor (video tier: camera rays).
    ``walls``: optional function from the points to fitted wall lines; the free space is cut off behind them."""
    params = params or Params()
    floor = find_floor(points[:, 1])
    if floor is None:
        return Estimate([], grid or make_grid(points, traj_xz), None)
    grid = grid or make_grid(points, traj_xz)
    cov = L.wall_coverage(points, floor.y, grid)
    nb = int(round((L.BAND_HI - L.BAND_LO) / L.BAND_H))
    thresh = int(round(params.cov_fraction * nb))
    free, barrier = L.free_space(points, floor.y, grid, traj_xz, cov, thresh, seen_extra=free_hint)
    lines = walls(points) if walls is not None else []
    if lines:
        free = L.clip_to_walls(free, grid, lines, traj_xz)
    closed, closures = L.close_gaps(barrier, grid, max_gap=params.max_gap)
    closed_d = cv2.dilate(closed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    free = free & ~closed_d
    pieces = L.split_rooms(free, traj_xz, grid, min_traj_cells=params.min_traj_cells, min_area=params.min_room_area)

    h = points[:, 1] - floor.y
    band = (h > L.BAND_LO) & (h < L.BAND_HI)
    bx, bz = grid.index(points[band][:, [0, 2]])
    ok = grid.inside(bx, bz)
    band_xz = points[band][ok][:, [0, 2]]
    band_h = h[band][ok]
    band_ix, band_iz = bx[ok], bz[ok]
    on_wall = barrier[band_iz, band_ix]
    reach = 2 * int(round(0.5 / grid.cell)) + 1  # snapping only looks 0.5 m around a room

    rooms: list[RoomEst] = []
    for mask, visits in pieces:
        near_room = cv2.dilate(mask.astype(np.uint8), np.ones((reach, reach), np.uint8)).astype(bool)
        near = near_room[band_iz, band_ix]
        room_wall_pts = band_xz[near & on_wall]
        # doorway gaps are measured on raw points (not 5 cm cells) in the upper band, where
        # furniture is rare, so the jambs are not quantised to the grid
        upper_pts = band_xz[near & (band_h >= 0.9)]
        outline = L.build_outline(mask, grid, room_wall_pts, visits, epsilon=params.outline_eps)
        if outline is None:
            continue
        area = abs(L.signed_area(outline.polygon))
        rfloor, ceiling = _room_ceiling(points, grid, mask, floor, params.ceiling_footprint)
        est = RoomEst(outline, area, rfloor, ceiling)
        if ceiling is None:
            est.flags.append("ceiling_not_observed")
        elif not (2.0 <= ceiling.y - rfloor.y <= 5.0):
            est.flags.append("ceiling_height_implausible")
        # Doorways are a gap in the wall evidence that a closure was drawn across. Either
        # a gap inside a supported wall, or a whole unsupported edge (a door at a corner).
        def closure_near(mid: np.ndarray) -> bool:
            return any(np.linalg.norm(0.5 * (c0 + c1) - mid) <= 0.35 for c0, c1 in closures)

        for i, e in enumerate(outline.edges):
            if e.supported:
                for a, b in L.find_gaps(e, upper_pts, params.min_opening, params.max_opening):
                    if closure_near(0.5 * (a + b)):
                        est.openings.append(OpeningEst(a, b, i))
            elif params.min_opening <= e.length <= params.max_opening and closure_near(0.5 * (e.p0 + e.p1)):
                est.openings.append(OpeningEst(e.p0.copy(), e.p1.copy(), i))
        if any((not e.supported) and e.length > params.max_opening for e in outline.edges):
            est.flags.append("open_boundary")
        rooms.append(est)
    return Estimate(rooms, grid, floor, closures, barrier, free)
