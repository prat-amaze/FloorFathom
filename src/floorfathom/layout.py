"""Top-down room layout from a gravity-aligned point cloud.

Pipeline (all in a 5 cm top-down grid):
  1. wall evidence: walls are occupied over most of the height range, furniture is not,
  2. free space: observed floor plus everything the phone walked through, minus walls,
  3. doorways: wall segments are extended along their own direction across gaps up to
     doorway width until they meet another wall, which closes the room,
  4. rooms: connected free space that the camera actually visited,
  5. outline: simplify each room's outer contour to a polygon,
  6. snap: refit every polygon edge to the wall points next to it; edges with no wall
     evidence are kept as straight chords, and gaps inside a wall are doorway candidates.

Right angles are never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CELL = 0.05
BAND_H = 0.15
BAND_LO = 0.25
BAND_HI = 2.05


@dataclass
class Grid:
    x0: float
    z0: float
    nx: int
    nz: int
    cell: float = CELL

    @classmethod
    def around(cls, xz: np.ndarray, margin: float = 0.5, cell: float = CELL) -> "Grid":
        lo = xz.min(axis=0) - margin
        hi = xz.max(axis=0) + margin
        return cls(float(lo[0]), float(lo[1]), int(np.ceil((hi[0] - lo[0]) / cell)) + 1, int(np.ceil((hi[1] - lo[1]) / cell)) + 1, cell)

    def index(self, xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((xz[:, 0] - self.x0) / self.cell).astype(np.int64)
        iz = np.floor((xz[:, 1] - self.z0) / self.cell).astype(np.int64)
        return ix, iz

    def inside(self, ix: np.ndarray, iz: np.ndarray) -> np.ndarray:
        return (ix >= 0) & (ix < self.nx) & (iz >= 0) & (iz < self.nz)

    def to_xz(self, col: np.ndarray, row: np.ndarray) -> np.ndarray:
        """Cell centres (col = x index, row = z index) to metres."""
        return np.stack([self.x0 + (col + 0.5) * self.cell, self.z0 + (row + 0.5) * self.cell], axis=-1)


def _count_grid(grid: Grid, ix, iz) -> np.ndarray:
    ok = grid.inside(ix, iz)
    flat = iz[ok] * grid.nx + ix[ok]
    return np.bincount(flat, minlength=grid.nx * grid.nz).reshape(grid.nz, grid.nx)


def wall_coverage(points: np.ndarray, floor_y: float, grid: Grid, min_pts: int = 3) -> np.ndarray:
    """Number of 15 cm height bands (0.25-2.05 m above the floor) occupied in each cell."""
    h = points[:, 1] - floor_y
    nb = int(round((BAND_HI - BAND_LO) / BAND_H))
    b = np.floor((h - BAND_LO) / BAND_H).astype(np.int64)
    ok = (b >= 0) & (b < nb)
    ix, iz = grid.index(points[ok][:, [0, 2]])
    ok2 = grid.inside(ix, iz)
    flat = (b[ok][ok2] * grid.nz + iz[ok2]) * grid.nx + ix[ok2]
    counts = np.bincount(flat, minlength=nb * grid.nz * grid.nx).reshape(nb, grid.nz, grid.nx)
    return (counts >= min_pts).sum(axis=0).astype(np.int16)


def _ellipse(radius_cells: int) -> np.ndarray:
    d = 2 * radius_cells + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


@dataclass
class Edge:
    p0: np.ndarray  # metres (x, z)
    p1: np.ndarray
    supported: bool  # True when wall points back this edge
    support: float = 0.0  # fraction of the edge length with wall evidence
    line: tuple[np.ndarray, float] | None = None  # (unit normal, offset) of the fitted wall line

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


@dataclass
class RoomOutline:
    mask: np.ndarray  # boolean room mask on the grid
    polygon: np.ndarray  # (K, 2) metres, counter-clockwise
    edges: list[Edge] = field(default_factory=list)
    trajectory_cells: int = 0


def free_space(
    points: np.ndarray,
    floor_y: float,
    grid: Grid,
    traj_xz: np.ndarray,
    cov: np.ndarray,
    cov_thresh: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (free, barrier) boolean masks on the grid."""
    barrier = cov >= cov_thresh
    barrier_d = cv2.dilate(barrier.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    h = points[:, 1] - floor_y
    ix, iz = grid.index(points[:, [0, 2]])
    on_floor = np.abs(h) <= 0.04
    anyp = (h > 0.04) & (h < BAND_HI)
    floor_ct = _count_grid(grid, ix[on_floor], iz[on_floor])
    any_ct = _count_grid(grid, ix[anyp], iz[anyp])
    seen = (floor_ct >= 2) | (any_ct >= 3)

    free = seen & ~barrier_d
    tx, tz = grid.index(traj_xz)
    ok = grid.inside(tx, tz)
    traj = np.zeros_like(free, dtype=np.uint8)
    traj[tz[ok], tx[ok]] = 1
    traj = cv2.dilate(traj, _ellipse(int(round(0.25 / grid.cell)))).astype(bool)
    free |= traj & ~barrier_d
    # fill small unobserved pockets without bridging across walls
    closed = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(int(round(0.2 / grid.cell)))).astype(bool)
    free = closed & ~barrier_d
    return free, barrier


def close_gaps(barrier: np.ndarray, grid: Grid, max_gap: float = 1.1, min_seg: float = 0.5) -> tuple[np.ndarray, list]:
    """Close doorways by extending wall segments along their own direction.

    A wall that stops at a doorway keeps pointing at the door frame on the other side.
    Every wall segment end that is followed by free cells is marched forward up to
    ``max_gap``; if it reaches another wall the gap is drawn in as wall.
    Returns the closed barrier mask and the list of drawn closures (start, end in metres).
    """
    cell = grid.cell
    b8 = barrier.astype(np.uint8)
    lines = cv2.HoughLinesP(
        b8, 1, np.pi / 180, threshold=int(0.5 / cell), minLineLength=int(min_seg / cell), maxLineGap=int(0.3 / cell)
    )
    closed = b8.copy()
    drawn = []
    if lines is None:
        return closed.astype(bool), drawn
    bar_d = cv2.dilate(b8, np.ones((3, 3), np.uint8))
    h, w = b8.shape
    steps = int(max_gap / cell)
    skip = int(0.25 / cell)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        p = [np.array([x1, y1], float), np.array([x2, y2], float)]
        d = p[1] - p[0]
        n = np.linalg.norm(d)
        if n < 1:
            continue
        d /= n
        for end, direction in ((p[1], d), (p[0], -d)):
            hit = None
            free_run = 0
            leaving = True  # the first cells past the end are still the segment's own wall
            for s in range(1, steps + 1):
                q = end + direction * s
                cx, cy = int(round(q[0])), int(round(q[1]))
                if not (0 <= cx < w and 0 <= cy < h):
                    break
                if bar_d[cy, cx]:
                    if leaving and s <= 4:
                        continue
                    if free_run >= skip:
                        hit = q
                    break
                leaving = False
                free_run += 1
            if hit is not None and free_run >= skip:
                cv2.line(closed, (int(round(end[0])), int(round(end[1]))), (int(round(hit[0])), int(round(hit[1]))), 1, 1)
                drawn.append((grid.to_xz(end[0], end[1]), grid.to_xz(hit[0], hit[1])))
    return closed.astype(bool), drawn


def split_rooms(
    free: np.ndarray,
    traj_xz: np.ndarray,
    grid: Grid,
    neck: float = 0.15,
    min_traj_cells: int = 10,
    min_area: float = 2.0,
) -> list[tuple[np.ndarray, int]]:
    """Connected free space that the camera visited, one mask per room.

    Necks narrower than 2*neck (0.3 m, gaps between furniture and cables) are cut so they
    do not join separate spaces. Doorways are closed earlier, by ``close_gaps``.

    Returns [(mask, n_trajectory_cells)], most visited first.
    """
    r = int(round(neck / grid.cell))
    eroded = cv2.erode(free.astype(np.uint8), _ellipse(r))
    n, lab = cv2.connectedComponents(eroded, connectivity=4)
    tx, tz = grid.index(traj_xz)
    ok = grid.inside(tx, tz)
    traj = np.zeros(free.shape, dtype=np.uint8)
    traj[tz[ok], tx[ok]] = 1
    out = []
    for k in range(1, n):
        core = lab == k
        room = cv2.dilate(core.astype(np.uint8), _ellipse(r)).astype(bool) & free
        if room.sum() * grid.cell**2 < min_area:
            continue
        # the camera often hugs a wall where the eroded core is absent, so count visits
        # against the restored room, not the core
        visits = int((room & traj.astype(bool)).sum())
        if visits < min_traj_cells:
            continue
        out.append((room, visits))
    # rooms grown from different cores can overlap after dilation; give each cell to one
    out.sort(key=lambda t: -t[1])
    taken = np.zeros(free.shape, dtype=bool)
    unique = []
    for room, visits in out:
        room = room & ~taken
        if room.sum() * grid.cell**2 < min_area:
            continue
        taken |= room
        unique.append((room, visits))
    return unique


def outer_polygon(mask: np.ndarray, grid: Grid, epsilon: float = 0.12) -> np.ndarray | None:
    """Simplified outer contour of the largest blob, in metres, counter-clockwise."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(2))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, epsilon / grid.cell, True)[:, 0, :].astype(np.float64)
    if len(approx) < 3:
        return None
    poly = grid.to_xz(approx[:, 0], approx[:, 1])
    if signed_area(poly) < 0:
        poly = poly[::-1]
    return poly


def signed_area(poly: np.ndarray) -> float:
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def fit_line(pts: np.ndarray, trim: float = 0.04, iters: int = 3) -> tuple[np.ndarray, float] | None:
    """Total least squares line through 2D points with iterative outlier trimming.

    Returns (unit normal n, offset c) for the line n . p = c.
    """
    if len(pts) < 5:
        return None
    for _ in range(iters):
        mu = pts.mean(axis=0)
        u, s, vt = np.linalg.svd(pts - mu, full_matrices=False)
        d = vt[0]
        n = np.array([-d[1], d[0]])
        res = (pts - mu) @ n
        keep = np.abs(res) <= max(trim, 2.0 * float(np.std(res)))
        if keep.all() or keep.sum() < 5:
            break
        pts = pts[keep]
    mu = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return n, float(n @ mu)


def intersect(l1: tuple[np.ndarray, float], l2: tuple[np.ndarray, float]) -> np.ndarray | None:
    a = np.array([l1[0], l2[0]])
    if abs(np.linalg.det(a)) < 0.26:  # sin(15 degrees): near-parallel lines do not define a corner
        return None
    return np.linalg.solve(a, np.array([l1[1], l2[1]]))


def snap_polygon(
    polygon: np.ndarray,
    wall_pts: np.ndarray,
    search: float = 0.35,
    min_support: float = 0.35,
    min_pts: int = 12,
) -> list[Edge]:
    """Turn a rough polygon into edges refitted to the wall points next to them."""
    k = len(polygon)
    edges: list[Edge] = []
    for i in range(k):
        p0, p1 = polygon[i], polygon[(i + 1) % k]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        t = d / L
        n = np.array([-t[1], t[0]])
        rel = wall_pts - p0
        along = rel @ t
        perp = rel @ n
        near = (along > -0.15) & (along < L + 0.15) & (np.abs(perp) <= search)
        pts = wall_pts[near]
        line = fit_line(pts) if len(pts) >= min_pts else None
        support = 0.0
        if line is not None:
            # coverage along the edge in 5 cm bins
            a = along[near]
            res = perp[near]
            good = np.abs(res - np.median(res)) <= 0.06
            bins = np.unique(np.floor(a[good] / 0.05).astype(int))
            support = min(1.0, len(bins) * 0.05 / L)
        edges.append(Edge(p0, p1, support >= min_support and line is not None, support, line if support >= min_support else None))
    return edges


def find_gaps(
    edge: Edge,
    wall_pts: np.ndarray,
    min_gap: float = 0.55,
    max_gap: float = 1.8,
    bin_size: float = 0.01,
    band: float = 0.08,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Empty stretches of wall evidence along a supported edge (candidate doorways).

    Wall points within ``band`` of the edge's line are binned along the edge in 1 cm
    bins and smoothed over 5 cm. A bin is empty when its density is below half the
    wall's median density; every empty run between ``min_gap`` and ``max_gap`` long is
    returned as (start, end) in metres.
    """
    if edge.line is None or len(wall_pts) == 0:
        return []
    d = edge.p1 - edge.p0
    length = float(np.linalg.norm(d))
    if length < min_gap:
        return []
    t = d / length
    n, c = edge.line
    on_line = np.abs(wall_pts @ n - c) <= band
    along = (wall_pts[on_line] - edge.p0) @ t
    nb = int(np.ceil(length / bin_size))
    idx = np.floor(along / bin_size).astype(int)
    idx = idx[(idx >= 0) & (idx < nb)]
    counts = np.bincount(idx, minlength=nb).astype(float)
    if counts.sum() == 0:
        return []
    # Depth noise smears wall points a centimetre or two past a door jamb, so "any point
    # present" would shrink the gap. The unbiased jamb is where the local density falls
    # to half of the wall's typical density.
    smooth = np.convolve(counts, np.ones(5) / 5.0, mode="same")
    typical = float(np.median(smooth[smooth > 0]))
    empty = smooth < 0.5 * typical
    gaps = []
    i = 0
    while i < nb:
        if empty[i]:
            j = i
            while j < nb and empty[j]:
                j += 1
            width = (j - i) * bin_size
            touches_both_ends = i == 0 and j == nb
            if min_gap <= width <= max_gap and not touches_both_ends:
                gaps.append((edge.p0 + t * i * bin_size, edge.p0 + t * j * bin_size))
            i = j
        else:
            i += 1
    return gaps


def rebuild_polygon(edges: list[Edge]) -> np.ndarray:
    """Vertices from intersecting neighbouring snapped lines.

    Two supported neighbours meet at their line intersection when it is close to the
    original vertex. A supported edge next to an unsupported one keeps the original
    vertex, moved onto the supported line. Unsupported edges stay straight chords.
    """
    k = len(edges)
    verts = []
    for i in range(k):
        prev, cur = edges[i - 1], edges[i]
        v = cur.p0.copy()
        if prev.line is not None and cur.line is not None:
            x = intersect(prev.line, cur.line)
            if x is not None and np.linalg.norm(x - v) <= 0.7:
                v = x
            else:
                n, c = cur.line
                v = v - (n @ v - c) * n
        elif cur.line is not None:
            n, c = cur.line
            v = v - (n @ v - c) * n
        elif prev.line is not None:
            n, c = prev.line
            v = v - (n @ v - c) * n
        verts.append(v)
    return np.array(verts)


def merge_collinear(edges: list[Edge], angle_deg: float = 6.0, offset: float = 0.06) -> list[Edge]:
    """Merge consecutive supported edges that lie on the same wall line."""
    changed = True
    edges = list(edges)
    while changed and len(edges) > 3:
        changed = False
        for i in range(len(edges)):
            a, b = edges[i], edges[(i + 1) % len(edges)]
            if a.line is None or b.line is None:
                continue
            cosang = abs(float(a.line[0] @ b.line[0]))
            if cosang < np.cos(np.radians(angle_deg)):
                continue
            n = a.line[0] if a.line[0] @ b.line[0] > 0 else -a.line[0]
            if abs(a.line[1] - (n @ b.p0)) > offset and abs(b.line[1] - (n @ a.p0)) > offset:
                continue
            merged = Edge(a.p0, b.p1, True, max(a.support, b.support), a.line if a.length >= b.length else b.line)
            j = (i + 1) % len(edges)
            edges[i] = merged
            del edges[j]
            changed = True
            break
    return edges


def drop_short(edges: list[Edge], min_len: float = 0.25, corner_cut: float = 0.6) -> list[Edge]:
    """Remove tiny edges by extending their neighbours to meet.

    Edges shorter than ``min_len`` always go (furniture bumps, pilasters). Edges up to
    ``corner_cut`` long go only when both neighbouring walls meet within 0.45 m of both
    ends of the edge: that is a bevelled corner, an artefact of thin wall coverage where
    two walls meet, and dropping it moves the area by well under 0.1 m2.
    """
    edges = list(edges)
    changed = True
    while changed and len(edges) > 3:
        changed = False
        for i in range(len(edges)):
            e = edges[i]
            if e.length >= corner_cut:
                continue
            prev, nxt = edges[i - 1], edges[(i + 1) % len(edges)]
            x = None
            if prev.line is not None and nxt.line is not None:
                x = intersect(prev.line, nxt.line)
            if e.length >= min_len and (
                x is None or np.linalg.norm(x - e.p0) > 0.45 or np.linalg.norm(x - e.p1) > 0.45
            ):
                continue
            if x is not None and np.linalg.norm(x - e.p0) <= 0.8:
                prev.p1 = x
                nxt.p0 = x
            else:
                mid = 0.5 * (e.p0 + e.p1)
                prev.p1 = mid
                nxt.p0 = mid
            del edges[i]
            changed = True
            break
    return edges


def build_outline(
    room_mask: np.ndarray,
    grid: Grid,
    wall_pts: np.ndarray,
    trajectory_cells: int = 0,
) -> RoomOutline | None:
    rough = outer_polygon(room_mask, grid)
    if rough is None:
        return None
    edges = snap_polygon(rough, wall_pts)
    edges = merge_collinear(edges)
    verts = rebuild_polygon(edges)
    for i, e in enumerate(edges):
        e.p0 = verts[i]
        e.p1 = verts[(i + 1) % len(edges)]
    edges = drop_short(edges)
    edges = merge_collinear(edges)
    verts = rebuild_polygon(edges)
    for i, e in enumerate(edges):
        e.p0 = verts[i]
        e.p1 = verts[(i + 1) % len(edges)]
    return RoomOutline(room_mask, verts, edges, trajectory_cells)
