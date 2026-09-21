"""Wall lines of a video cloud, for the room estimator to cut the free space off behind.

Walls are fitted as vertical planes (``photo_layout.wall_segments``: RANSAC in the band of points between the
camera and the ceiling, cut into dense runs, parallel duplicates merged), which uses every point of every
keyframe on a wall at once instead of the 5 cm cells the grid estimator sees, so a wall broken up by furniture,
a glossy patch or a door stays one line. The estimator then clips the room to the camera side of each line
(``layout.clip_to_walls``), which is where a wall stops depth or camera rays leaking into the next room.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .layout import WallLine
from .photo_layout import wall_segments

MIN_WALL_EXTENT = 1.5  # metres: a shorter line is a pilaster or a cupboard edge, not a wall to cut the room at


def wall_finder(camera_y: float, seed: int = 0, min_extent: float = MIN_WALL_EXTENT) -> Callable[[np.ndarray], list[WallLine]]:
    """Function from a levelled cloud (+y up, metres) to wall lines; ``camera_y`` is the height of the camera path,
    the origin of the height slab the wall fit looks at."""

    def find(points: np.ndarray) -> list[WallLine]:
        shifted = np.asarray(points, float) - np.array([0.0, camera_y, 0.0])
        return [WallLine(s.normal, s.offset, s.t0, s.t1) for s in wall_segments(shifted, seed=seed) if s.t1 - s.t0 >= min_extent]

    return find
