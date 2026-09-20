"""Keep ceiling/floor damage regions valid after a room is rigidly moved (see ``stitch.transform_room``).

Wall damage is stored as ``(s, h)`` along the wall and needs no change under a rigid move. Ceiling and
floor damage is stored in plan coordinates, so it goes stale once the room is rotated and shifted.
"""

import numpy as np

from .schema import RoomPlan

MIN_WALL_LEN = 0.3  # metres; shorter walls give a noisy angle
MAX_LEN_MISMATCH = 0.05  # metres; more than this and the walls are not really the same wall


def _rigid_transform(old: RoomPlan, new: RoomPlan) -> tuple[float, np.ndarray] | None:
    """(angle, shift) mapping ``old``'s onto ``new``'s reference wall, or ``None`` if it can't be trusted."""
    for i, wall in enumerate(old.walls):
        if i >= len(new.walls):
            break
        a0, a1 = np.asarray(wall.start, float), np.asarray(wall.end, float)
        old_len = float(np.linalg.norm(a1 - a0))
        if old_len <= MIN_WALL_LEN:
            continue
        b0, b1 = np.asarray(new.walls[i].start, float), np.asarray(new.walls[i].end, float)
        new_len = float(np.linalg.norm(b1 - b0))
        if abs(new_len - old_len) > MAX_LEN_MISMATCH:
            return None
        angle = float(np.arctan2(*(b1 - b0)[::-1]) - np.arctan2(*(a1 - a0)[::-1]))
        r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return angle, b0 - r @ a0
    return None


def reframe_damage(old: RoomPlan, new: RoomPlan) -> None:
    """Move ``new``'s ceiling/floor damage polygons and centres by the rigid transform from ``old`` to ``new``.

    Does nothing if that transform cannot be recovered from a wall shared by both rooms.
    """
    transform = _rigid_transform(old, new)
    if transform is None:
        return
    angle, shift = transform
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    def pt(p) -> tuple[float, float]:
        q = r @ np.asarray(p, float) + shift
        return round(float(q[0]), 4), round(float(q[1]), 4)

    moved = []
    for region in new.damage:
        if region.surface.kind in ("ceiling", "floor"):
            region = region.model_copy(update={"polygon": [pt(p) for p in region.polygon], "centre": pt(region.centre)})
        moved.append(region)
    new.damage = moved
