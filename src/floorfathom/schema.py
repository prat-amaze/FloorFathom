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

SCHEMA_VERSION = "0.1.0"

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


class RoomPlan(BaseModel):
    id: str
    polygon: list[Point] = Field(description="counter-clockwise corner points, plan coordinates")
    walls: list[Wall]
    ceiling_height: Measurement
    floor_area: Measurement
    openings: list[Opening]
    flags: list[str] = Field(default_factory=list)


class Diagnostics(BaseModel):
    frames_used: int
    points: int
    floor_height_world: float | None
    floor_sharpness: float | None = Field(description="fraction of all points inside the floor spike")
    bootstrap_replicates: int
    seed: int
    seconds: float
    conventions: str


class CapturePlan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    capture: str
    tier: Literal["lidar", "video", "photo"]
    units: Literal["m"] = "m"
    rooms: list[RoomPlan]
    diagnostics: Diagnostics


def json_schema() -> dict:
    return CapturePlan.model_json_schema()
