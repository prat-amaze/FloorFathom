"""Damage, concealed-damage flags and scope for a photo room: the photo tier's use of the shared ``assess`` driver.

The stills are taken from one spot, so a wall pixel is usually seen by one photo, not two, and the depth model's
relief is centimetres of noise: the driver is run with one view enough and depth used only to reject pixels that are
not on the surface (furniture, a mirror, a doorway). Only walls and the ceiling are judged: the floor of a home is
often glossy tile whose reflections would be read as marks. Everything is measured in the room's own frame; after
stitching, the room's copy in the property plan gets the same regions, moved with the room, and its flags are
recomputed with the neighbouring rooms in view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .assess import assess_room
from .concealed import concealed_flags
from .damage_frame import reframe_damage
from .schema import CapturePlan
from .scope import scope_items

if TYPE_CHECKING:
    from .photo_pipeline import PhotoRoom

KINDS = ("wall", "ceiling")
MPP = 0.005  # metres per patch pixel: the resolution the class rules were tuned at
MIN_VIEWS = 1
POSE_REL = 0.03  # assumed relative error of the wall plane's distance from arm swing and rotation-only registration
NO_SCALE_REL = 0.30  # used when the room carries no scale uncertainty (as for a depth-model-only scale)


def assess(room: PhotoRoom, stitched: CapturePlan, mpp: float = MPP) -> None:
    """Fill damage, concealed flags and scope on the room's own plan and on its copy in ``stitched``."""
    own = room.plan.rooms[0]
    if room.floor_y is None or not room.frames or not own.polygon:
        return
    rel = float(np.hypot(room.scale_rel_sigma if room.scale_rel_sigma is not None else NO_SCALE_REL, POSE_REL))
    notes = assess_room(
        own, room.floor_y, room.ceiling_y, lambda _plane: room.frames, kinds=KINDS, mpp=mpp,
        min_views=MIN_VIEWS, use_relief=False, scale_rel_sigma=rel,
    )
    if notes:
        own.flags = list(dict.fromkeys([*own.flags, "damage_coverage_partial"]))
        room.plan.diagnostics.notes = [*room.plan.diagnostics.notes, *notes]
    moved = next((r for r in stitched.rooms if r.id == own.id), None)
    if moved is None or moved is own:
        return
    moved.damage = [d.model_copy(deep=True) for d in own.damage]
    reframe_damage(own, moved)
    others = [r for r in stitched.rooms if r.id != own.id and r.frame == "capture"] if moved.frame == "capture" else []
    moved.concealed_flags = concealed_flags(moved, moved.damage, others)
    moved.scope = scope_items(moved, moved.damage, moved.concealed_flags)
    if notes:
        moved.flags = list(dict.fromkeys([*moved.flags, "damage_coverage_partial"]))
        stitched.diagnostics.notes = [*stitched.diagnostics.notes, *notes]
