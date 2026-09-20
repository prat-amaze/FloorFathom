"""Unroll the surfaces of a room into flat colour patches: the tier-agnostic step in front of ``damage.detect``.

A surface is a rectangle in world space (a wall from floor to ceiling, the ceiling, the floor). Posed RGB frames
are projected onto it and averaged, weighted toward close and face-on views. Where a frame also has depth, a pixel
counts only if the depth agrees with the surface (so furniture, doorways and people in front of it are left out)
and the depth's offset from the surface becomes the relief. Without depth, pixels whose views disagree in
lightness are dropped instead. A pixel needs two accepted views to be valid.

Frames come from an adapter per tier: ``Frame`` holds the picture, its intrinsics, its camera-to-world pose (OpenCV
axes) and optionally depth. ``undo`` maps points from the plan's frame back into the frame the pose was measured in
(the drift correction of a LiDAR walk moves the plan away from the raw poses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .damage import Patch
from .schema import RoomPlan, SurfaceRef

MPP = 0.005  # metres per patch pixel
COS_MIN = 0.4  # views more oblique than about 66 degrees are not used
Z_MIN, Z_MAX = 0.3, 5.0  # metres from the camera
EDGE_JUMP = (0.08, 0.05)  # depth change between neighbouring readings that marks an object outline: metres plus a share of the range
DEPTH_TOL = (0.10, 0.05)  # accepted depth disagreement: at least 10 cm, or 5% of the range
LATTICE = (9, 6)
LUM_SPREAD = 0.06  # views of one point in a depth-free capture may differ this much in lightness (0..1)
DEFAULT_WALL_HEIGHT = 2.0  # used when no ceiling was observed


@dataclass
class Frame:
    rgb: np.ndarray  # (h, w, 3) uint8, RGB
    K: np.ndarray  # (3, 3) intrinsics of ``rgb``
    pose: np.ndarray  # (4, 4) camera to world, OpenCV axes (x right, y down, z forward)
    depth: np.ndarray | None = None  # metres along the optical axis, any resolution, 0 = no reading
    depth_ok: np.ndarray | None = None  # bool, same shape as depth: readings that can be trusted
    undo: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass
class Plane:
    origin: np.ndarray  # world point at s = 0, h = 0
    u: np.ndarray  # unit vector of increasing column
    v: np.ndarray  # unit vector of increasing row
    normal: np.ndarray  # unit vector pointing into the room
    size: tuple[float, float]  # metres along u and v
    inside: np.ndarray | None = None  # (rows, cols) bool: pixels that belong to the surface; None means all
    mpp: float = MPP  # metres per patch pixel

    @property
    def shape(self) -> tuple[int, int]:
        return max(1, int(np.ceil(self.size[1] / self.mpp))), max(1, int(np.ceil(self.size[0] / self.mpp)))


def surface_planes(room: RoomPlan, floor_y: float, ceiling_y: float | None, mpp: float = MPP) -> list[tuple[SurfaceRef, Plane]]:
    """The walls that were seen, the ceiling (if observed) and the floor of a room, in the plan's frame."""
    out: list[tuple[SurfaceRef, Plane]] = []
    if not room.polygon:
        return out
    poly = np.array(room.polygon, float)
    centre = poly.mean(axis=0)
    height = (ceiling_y - floor_y) if ceiling_y is not None else DEFAULT_WALL_HEIGHT
    for w in room.walls:
        a, b = np.array(w.start, float), np.array(w.end, float)
        ln = float(np.linalg.norm(b - a))
        if w.evidence != "wall_points" or ln < 0.3:
            continue
        d = (b - a) / ln
        n_plan = np.array([-d[1], d[0]])
        if n_plan @ (centre - (a + b) / 2) < 0:
            n_plan = -n_plan
        out.append(
            (
                SurfaceRef(kind="wall", id=w.id),
                Plane(
                    origin=np.array([a[0], floor_y, -a[1]]),
                    u=np.array([d[0], 0.0, -d[1]]),
                    v=np.array([0.0, 1.0, 0.0]),
                    normal=np.array([n_plan[0], 0.0, -n_plan[1]]),
                    size=(ln, height),
                    mpp=mpp,
                ),
            )
        )
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    size = (float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    rows, cols = max(1, int(np.ceil(size[1] / mpp))), max(1, int(np.ceil(size[0] / mpp)))
    inside = np.zeros((rows, cols), np.uint8)
    cv2.fillPoly(inside, [np.round((poly - lo) / mpp).astype(np.int32)], 1)
    for kind, y, n_y in (("ceiling", ceiling_y, -1.0), ("floor", floor_y, 1.0)):
        if y is None:
            continue
        out.append(
            (
                SurfaceRef(kind=kind, id=f"{room.id}_{kind}"),
                Plane(
                    origin=np.array([lo[0], y, -lo[1]]),
                    u=np.array([1.0, 0.0, 0.0]),
                    v=np.array([0.0, 0.0, -1.0]),
                    normal=np.array([0.0, n_y, 0.0]),
                    size=size,
                    inside=inside.astype(bool),
                    mpp=mpp,
                ),
            )
        )
    return out


def _depth_edges(frame: Frame) -> np.ndarray:
    """Depth pixels at or next to a jump in depth: colour there may belong to the object in front."""
    edges = getattr(frame, "_edges", None)
    if edges is None:
        d = frame.depth
        jump = np.zeros(d.shape, bool)
        for axis in (0, 1):
            a, b = np.moveaxis(d, axis, 0)[:-1], np.moveaxis(d, axis, 0)[1:]
            big = np.moveaxis(np.abs(a - b) > EDGE_JUMP[0] + EDGE_JUMP[1] * np.minimum(a, b), 0, axis)  # a tilted plane changes with range
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[axis], hi[axis] = slice(0, -1), slice(1, None)
            jump[tuple(lo)] |= big
            jump[tuple(hi)] |= big
        jump |= d <= 0
        edges = cv2.dilate(jump.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        frame._edges = edges
    return edges


def lattice_of(plane: Plane) -> np.ndarray:
    """A coarse grid of world points over the plane, to ask cheaply whether a frame looks at it."""
    s_g, h_g = np.meshgrid(np.linspace(0.05, 0.95, LATTICE[0]), np.linspace(0.05, 0.95, LATTICE[1]))
    return (
        plane.origin[None, :]
        + (s_g.ravel() * plane.size[0])[:, None] * plane.u[None, :]
        + (h_g.ravel() * plane.size[1])[:, None] * plane.v[None, :]
    ).astype(np.float32)


def project(frame: Frame, pts: np.ndarray):
    """Points of the plan's frame into ``frame``: (points in the pose's frame, camera position, z, u, v)."""
    p = frame.undo(pts) if frame.undo is not None else pts
    t = frame.pose[:3, 3].astype(np.float32)
    x = (p - t) @ frame.pose[:3, :3].astype(np.float32)
    z = x[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = frame.K[0, 0] * x[:, 0] / z + frame.K[0, 2]
        v = frame.K[1, 1] * x[:, 1] / z + frame.K[1, 2]
    return p, t, z, u, v


def looks_at(lattice: np.ndarray, frame: Frame) -> bool:
    """True if any lattice point is in front of the camera and inside the picture. Needs only pose, K and rgb's shape."""
    _p, _t, z, u, v = project(frame, lattice)
    h, w = frame.rgb.shape[:2]
    return bool(((z > Z_MIN) & (z < Z_MAX) & (u >= 0) & (u < w) & (v >= 0) & (v < h)).any())


class Accumulator:
    """Frames averaged onto one plane."""

    def __init__(self, plane: Plane, min_views: int = 2, use_relief: bool = True):
        self.plane = plane
        self.min_views = min_views  # views a pixel needs before it counts
        self.use_relief = use_relief  # False: depth only accepts or rejects pixels, no relief is reported
        rows, cols = plane.shape
        self.shape = (rows, cols)
        r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
        s, h = (c.ravel() + 0.5) * plane.mpp, (r.ravel() + 0.5) * plane.mpp
        self.points = (
            plane.origin[None, :] + s[:, None] * plane.u[None, :] + h[:, None] * plane.v[None, :]
        ).astype(np.float32)
        n = rows * cols
        self.w = np.zeros(n, np.float32)
        self.rgb = np.zeros((n, 3), np.float32)
        self.count = np.zeros(n, np.int16)
        self.lum = np.zeros(n, np.float32)
        self.lum2 = np.zeros(n, np.float32)
        self.rel = np.zeros(n, np.float32)
        self.rel_w = np.zeros(n, np.float32)
        self.any_depth = False
        self.no_depth_views = 0
        self.lattice = lattice_of(plane)

    def sees(self, frame: Frame) -> bool:
        return looks_at(self.lattice, frame)

    def add(self, frame: Frame) -> int:
        """Project one frame onto the plane; returns how many pixels it contributed to."""
        if not self.sees(frame):
            return 0
        p, t, z, u, v = project(frame, self.points)
        h, w = frame.rgb.shape[:2]
        ray = p - t
        dist = np.linalg.norm(ray, axis=1)
        cos = np.abs(ray @ self.plane.normal.astype(np.float32)) / np.maximum(dist, 1e-6)
        ok = (z > Z_MIN) & (z < Z_MAX) & (cos > COS_MIN) & (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
        if self.plane.inside is not None:
            ok &= self.plane.inside.ravel()
        rel = None
        if frame.depth is not None:
            dh, dw = frame.depth.shape
            idx = np.nonzero(ok)[0]
            du = np.clip((u[idx] * dw / w).astype(np.int32), 0, dw - 1)
            dv = np.clip((v[idx] * dh / h).astype(np.int32), 0, dh - 1)
            zd = frame.depth[dv, du]
            good = zd > 0
            if frame.depth_ok is not None:
                good &= frame.depth_ok[dv, du]
            tol = np.maximum(DEPTH_TOL[0], DEPTH_TOL[1] * z[idx])
            good &= np.abs(zd - z[idx]) <= tol
            good &= ~_depth_edges(frame)[dv, du]
            keep = np.zeros_like(ok)
            keep[idx[good]] = True
            rel_all = np.zeros(len(ok), np.float32)
            rel_all[idx] = (z[idx] - zd) * dist[idx] / z[idx] * cos[idx]
            ok, rel = keep, rel_all
        idx = np.nonzero(ok)[0]
        if len(idx) < 50:
            return 0
        n = len(idx)
        width = 1024  # cv2.remap wants maps narrower than 32767 columns
        pad = -n % width
        mx = np.pad(u[idx].astype(np.float32), (0, pad)).reshape(-1, width)
        my = np.pad(v[idx].astype(np.float32), (0, pad)).reshape(-1, width)
        col = cv2.remap(frame.rgb, mx, my, cv2.INTER_LINEAR).reshape(-1, 3)[:n].astype(np.float32) / 255.0
        wt = (cos[idx] ** 3 / np.maximum(z[idx], 0.5) ** 2).astype(np.float32)
        lum = col @ np.array([0.299, 0.587, 0.114], np.float32)
        seen = self.w[idx] > 0
        if seen.sum() >= 500:  # match this frame's exposure to what is already there
            ref = (self.lum[idx][seen] / self.w[idx][seen]).mean()
            gain = float(np.clip(ref / max(lum[seen].mean(), 1e-3), 0.7, 1.4))
            col = np.clip(col * gain, 0.0, 1.0)
            lum = lum * gain
        self.w[idx] += wt
        self.rgb[idx] += wt[:, None] * col
        self.lum[idx] += wt * lum
        self.lum2[idx] += wt * lum * lum
        self.count[idx] += 1
        if rel is not None:
            self.any_depth = True
            self.rel[idx] += wt * rel[idx]
            self.rel_w[idx] += wt
        else:
            self.no_depth_views += 1
        return len(idx)

    def patch(self) -> Patch:
        rows, cols = self.shape
        valid = self.count >= self.min_views
        if self.plane.inside is not None:
            valid &= self.plane.inside.ravel()
        w = np.maximum(self.w, 1e-9)
        if self.no_depth_views:  # no depth to say what is in front of the surface: drop pixels whose views disagree
            mean = self.lum / w
            var = np.maximum(self.lum2 / w - mean * mean, 0.0)
            valid &= ~((self.count >= 3) & (np.sqrt(var) > LUM_SPREAD))
        rgb = (self.rgb / w[:, None]).reshape(rows, cols, 3)
        relief = None
        if self.any_depth and self.use_relief:
            rw = np.maximum(self.rel_w, 1e-9)
            relief = np.where(self.rel_w > 0, self.rel / rw, np.nan).astype(np.float32).reshape(rows, cols)
        return Patch(rgb=rgb, valid=valid.reshape(rows, cols), m_per_px=self.plane.mpp, relief=relief)
