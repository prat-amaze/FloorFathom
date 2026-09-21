"""Doorways of a video room from what the camera saw through its walls.

A wall stops every ray: the surface a pixel shows lies on it. Where a wall has a door in it, rays from inside the
room cross the wall line and end far behind it, in the next room or on the balcony. So a doorway is a stretch of
wall across which many rays from several keyframes pass through, and its jambs are the ends of that stretch. This
does not depend on wall points being missing (a sparse cloud is missing points everywhere), and a sliding glass door,
which has no wall points either way, shows up the same way.

Only rays that cross the wall low, between ``LOW_BAND`` above the floor, count: a door reaches the floor and a window
does not, so a window's through-rays are all higher up. A bin of the wall is open when more keyframes saw through it than saw a
surface on the wall there, and at least ``MIN_FRAMES`` keyframes saw through it.
"""

from __future__ import annotations

import numpy as np

from .estimate import OpeningEst
from .layout import Edge
from .video_rays import RayBundle

LOW_BAND = (0.1, 0.8)  # metres above the floor: where a door's rays cross the wall and a window's do not
BEYOND = 0.35  # a ray is "through" when it ends at least this far (m) behind the wall line
ON_WALL = 0.25  # a ray "hits" the wall when it ends within this distance (m) of the line
WALL_MIN_HEIGHT = 0.25  # ... and above the floor by more than this (m)
BIN = 0.03
MIN_FRAMES = 2  # keyframes that must see through a bin
MIN_WIDTH, MAX_WIDTH = 0.55, 2.2  # a doorway, up to a sliding glass door
BRIDGE = 0.10  # gaps this short (m) inside an opening do not end it (a pixel column, a bad depth value)


def find_openings(
    edges: list[Edge],
    bundle: RayBundle,
    rotation: np.ndarray,
    scale: float,
    floor_y: float,
    min_width: float = MIN_WIDTH,
    max_width: float = MAX_WIDTH,
) -> list[OpeningEst]:
    """Openings along the edges of a room outline, in the order of the edges.

    ``bundle`` holds the keyframes' rays in the SfM frame; ``rotation`` and ``scale`` (metres per SfM unit) put them in
    the plan frame, +y up, ``floor_y`` the floor height there. The interior side of an edge is the side the camera
    path is on.
    """
    rot = np.asarray(rotation).T
    origins = (bundle.origins @ rot) * scale
    ends = [(e @ rot) * scale for e in bundle.ends]
    centre = origins[:, [0, 2]].mean(axis=0)
    out: list[OpeningEst] = []
    for k, e in enumerate(edges):
        d = e.p1 - e.p0
        length = float(np.linalg.norm(d))
        if length < min_width:
            continue
        t = d / length
        n = np.array([-t[1], t[0]])
        if float((centre - e.p0) @ n) < 0:
            n = -n  # n points into the room
        nb = int(np.ceil(length / BIN))
        through = np.zeros(nb)  # keyframes with a ray through the bin
        hit = np.zeros(nb)  # keyframes with a ray that ended on the wall in the bin
        for o, en in zip(origins, ends):
            if len(en) == 0:
                continue
            so = float((o[[0, 2]] - e.p0) @ n)
            if so <= 0.05:
                continue  # the camera is not in front of this wall
            se = (en[:, [0, 2]] - e.p0) @ n  # signed distance of each ray's end from the wall line (+ = room side)
            u = so / (so - se)  # where along the ray it meets the line
            crossing = o + u[:, None] * (en - o)
            along = (crossing[:, [0, 2]] - e.p0) @ t
            height = crossing[:, 1] - floor_y
            inside = (along >= 0) & (along < length)
            thru = inside & (se < -BEYOND) & (height >= LOW_BAND[0]) & (height <= LOW_BAND[1])
            # a ray that ends on the floor next to the line shows the floor, not a wall
            on = inside & (np.abs(se) <= ON_WALL) & (height >= LOW_BAND[0]) & (height <= LOW_BAND[1]) & (en[:, 1] - floor_y > WALL_MIN_HEIGHT)
            th = np.bincount(np.minimum((along[thru] / BIN).astype(int), nb - 1), minlength=nb)
            oh = np.bincount(np.minimum((along[on] / BIN).astype(int), nb - 1), minlength=nb)
            through += th > 0
            hit += oh > 0
        open_ = (through >= MIN_FRAMES) & (through > hit)
        bridge = int(round(BRIDGE / BIN))
        i = 0
        while i < nb:
            if not open_[i]:
                i += 1
                continue
            j = i
            gap = 0
            end = i
            while j < nb and gap <= bridge:
                gap = 0 if open_[j] else gap + 1
                if open_[j]:
                    end = j
                j += 1
            width = (end + 1 - i) * BIN
            if min_width <= width <= max_width:
                out.append(OpeningEst(e.p0 + t * i * BIN, e.p0 + t * (end + 1) * BIN, k))
            i = end + 1
    return out
