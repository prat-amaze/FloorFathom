"""Output contract: one JSON document per capture.

Every measurement carries ``value``, ``lo`` and ``hi`` (a 95% interval) and ``unit``.
A quantity that could not be measured has ``value = null`` and says why in ``note``;
it never carries an invented number.

Plan coordinates are metres, ``(x, y) = (world x, -world z)``. Viewed from above with
world +y (up) pointing at the viewer, this puts the plan the right way round (not
mirrored) with plan +y drawn upward.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.3.0"

Point = tuple[float, float]


class Measurement(BaseModel):
    value: float | None
    lo: float | None
    hi: float | None
    unit: Literal["m", "m2"]
    method: str = Field(description="how the number was obtained")
    note: str | None = None


class Wall(BaseModel):
    id: str
    start: Point
    end: Point
    length: Measurement
    evidence: Literal["wall_points", "closure"] = Field(
        description="wall_points: fitted to LiDAR points on the wall; closure: a straight chord across a gap with no wall points"
    )


class Opening(BaseModel):
    id: str
    kind: Literal["doorway"]
    wall_id: str
    start: Point
    end: Point
    centre: Point
    width: Measurement
    note: str | None = None


class Station(BaseModel):
    """Where the camera stood in the room's frame (a photo, or a place along a video walk)."""

    position: Point
    height_above_floor: float | None = Field(default=None, description="metres")


class RoomPlan(BaseModel):
    id: str
    name: str | None = Field(default=None, description="the room's name in the capture (folder or clip name), if known")
    frame: Literal["capture", "room"] = Field(
        default="capture",
        description="capture: coordinates are in the frame shared by the whole capture; "
        "room: the room's own local frame, not yet placed relative to the other rooms",
    )
    polygon: list[Point] = Field(description="counter-clockwise corner points, plan coordinates")
    walls: list[Wall]
    ceiling_height: Measurement
    floor_area: Measurement
    openings: list[Opening]
    stations: list[Station] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Diagnostics(BaseModel):
    frames_used: int | None = Field(default=None, description="depth frames (lidar) or keyframes (video); not used for photos")
    points: int | None = None
    floor_height_world: float | None = None
    floor_sharpness: float | None = Field(default=None, description="fraction of all points inside the floor spike")
    bootstrap_replicates: int
    seed: int
    seconds: float
    conventions: str
    models: list[str] = Field(default_factory=list, description="pretrained models used, as id@revision")
    scale_method: str | None = Field(
        default=None, description="how metric scale was obtained, e.g. lidar_metric, reference_object, monocular_depth"
    )
    scale_factor: float | None = Field(default=None, description="metres per unit of the reconstruction, when there is one")
    scale_rel_sigma: float | None = Field(default=None, description="standard error of the scale as a fraction of it")
    notes: list[str] = Field(default_factory=list)


class Link(BaseModel):
    """Which room a doorway opens onto."""

    room: str
    opening: str
    other: str | None = Field(description="the room behind the doorway; null if none is within 1.5 m (outside, or not captured)")
    gap: float | None = Field(description="metres from the doorway to the other room's outline; a wall thickness is a real link")
    mutual: bool = Field(description="the other room has a doorway of its own next to this one")


class Placement(BaseModel):
    """How a room in its own frame was moved into the shared frame, by gluing doorways."""

    room: str
    angle: float = Field(description="radians counter-clockwise, about the room's own origin, applied before the shift")
    shift: Point
    host: str | None = Field(description="the placed room this one hangs off; null for the root")
    host_opening: str | None
    opening: str | None = Field(description="this room's doorway that meets the host's")
    overlap: float = Field(description="m2 shared with the rooms already placed")
    width_diff: float | None = Field(description="metres between the two sides of the doorway")
    margin: float | None = Field(description="runner-up cost minus this cost; small means the placement is ambiguous")


class Overlap(BaseModel):
    a: str
    b: str
    area: float = Field(description="m2 of floor both rooms claim")


class Stitching(BaseModel):
    """The rooms as one property: adjacency, placement, overlaps and footprint."""

    footprint: Measurement = Field(description="m2 of floor covered by all placed rooms, overlaps counted once")
    links: list[Link]
    placements: list[Placement] = Field(default_factory=list, description="empty when the rooms were already in one frame")
    overlaps: list[Overlap] = Field(default_factory=list)
    unplaced: list[str] = Field(default_factory=list, description="rooms with no doorway that fits anywhere")
    drift: str = Field(description="what was done about pose drift, and what it was not")


class CapturePlan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    capture: str
    tier: Literal["lidar", "video", "photo"]
    units: Literal["m"] = "m"
    rooms: list[RoomPlan]
    stitching: Stitching | None = None
    diagnostics: Diagnostics


def json_schema() -> dict:
    return CapturePlan.model_json_schema()
