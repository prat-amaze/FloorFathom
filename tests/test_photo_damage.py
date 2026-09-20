"""Damage on a photo room: stills from one spot, one view per pixel, the room then moved by stitching."""

import numpy as np
import pytest
from test_surfaces import BASE, FLOOR_Y, HEIGHT, K, RGB_H, RGB_W, Frame, Quad, _plain, _plan, _room_quads, _stained

from floorfathom.photo_damage import assess
from floorfathom.photo_pipeline import PhotoRoom
from floorfathom.schema import CapturePlan, Diagnostics, RoomPlan
from floorfathom.stitch import _rot, transform_room

STATION = (2.0, 1.3, -1.5)  # world x, y, z: the photographer's spot
ANGLE, SHIFT = 0.7, np.array([3.0, -1.0])
CEILING_STAIN = (3.4, 2.4)  # plan x, y
MPP = 0.01  # coarser than the shipped 0.005 to keep the test quick


def _one_spot_frames(quads, seed=0):
    """Eight yaws x two pitches from one point, as in a photo room: no parallax, one view for most wall pixels."""
    rng = np.random.default_rng(seed)
    uu, vv = np.meshgrid(np.arange(RGB_W), np.arange(RGB_H))
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    o, frames = np.array(STATION), []
    for k in range(8):
        yaw = 2 * np.pi * k / 8
        for pitch in (0.0, 30.0):
            f = np.array([np.sin(yaw) * np.cos(np.radians(pitch)), np.sin(np.radians(pitch)), np.cos(yaw) * np.cos(np.radians(pitch))])
            down = -(np.array([0, 1.0, 0]) - f[1] * f)
            down /= np.linalg.norm(down)
            R = np.stack([np.cross(down, f), down, f], axis=1)
            D = d_cam @ R.T
            best, colour = np.full(len(D), np.inf), np.zeros((len(D), 3))
            for q in quads.values():
                t, s, h = q.hit(o, D)
                closer = t < best
                colour[closer] = q.colour(s[closer], h[closer])
                best = np.where(closer, t, best)
            rgb = np.clip(colour + rng.normal(0, 0.004, colour.shape), 0, 1).reshape(RGB_H, RGB_W, 3)
            depth = best.reshape(RGB_H, RGB_W)[::2, ::2].astype(np.float32)
            depth[~np.isfinite(depth)] = 0
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = R, o
            frames.append(Frame(rgb=(rgb * 255).astype(np.uint8), K=K, pose=pose, depth=depth))
    return frames


def _quads(wall_stain=True, ceiling_stain=True):
    q = _room_quads(stain=wall_stain)
    if ceiling_stain:
        q["ceiling"] = Quad([0, HEIGHT, 0], [1, 0, 0], [0, 0, -1], 4, 3, _stained(*CEILING_STAIN, 0.2, 0.2))
    return q


def _diagnostics():
    return Diagnostics(bootstrap_replicates=0, seed=0, seconds=0.0, conventions="test", models=[], notes=[])


def _photo_room(quads):
    plan = CapturePlan(capture="room", tier="photo", rooms=[_plan()], diagnostics=_diagnostics())
    return PhotoRoom(plan, _one_spot_frames(quads), FLOOR_Y, FLOOR_Y + HEIGHT, 0.03)


def _moved_with_neighbour(own: RoomPlan):
    """The room as stitching leaves it (rotated and shifted) and a room on the other side of its wall w0."""
    moved = transform_room(own, ANGLE, SHIFT)
    a, b = np.array(moved.walls[0].start), np.array(moved.walls[0].end)
    d = (b - a) / np.linalg.norm(b - a)
    out = np.array([d[1], -d[0]])
    if out @ (a - np.mean(moved.polygon, axis=0)) < 0:
        out = -out
    poly = [tuple(a + 0.1 * out), tuple(b + 0.1 * out), tuple(b + 2.1 * out), tuple(a + 2.1 * out)]
    neighbour = _plan().model_copy(update={"id": "room_1", "polygon": poly, "walls": [], "frame": "capture"})
    return moved, neighbour


@pytest.fixture(scope="module")
def assessed():
    room = _photo_room(_quads())
    moved, neighbour = _moved_with_neighbour(room.plan.rooms[0])
    stitched = CapturePlan(capture="hall", tier="photo", rooms=[moved, neighbour], diagnostics=_diagnostics())
    assess(room, stitched, mpp=MPP)
    return room.plan.rooms[0], stitched.rooms[0], stitched


def test_the_wall_stain_and_the_ceiling_stain_are_found_from_one_spot_and_nothing_else(assessed):
    own, _, _ = assessed
    assert sorted((d.surface.kind, d.surface.id, d.damage_class) for d in own.damage) == [
        ("ceiling", "room_0_ceiling", "water_stain"), ("wall", "r0_w0", "water_stain")]
    wall = next(d for d in own.damage if d.surface.kind == "wall")
    assert abs(wall.centre[0] - 1.8) < 0.08 and abs(wall.centre[1] - 1.2) < 0.08  # (s, h) on the wall
    assert abs(wall.width.value - 0.30) < 0.10 and abs(wall.height.value - 0.40) < 0.10
    assert wall.width.lo <= wall.width.value <= wall.width.hi
    ceiling = next(d for d in own.damage if d.surface.kind == "ceiling")
    assert abs(ceiling.centre[0] - CEILING_STAIN[0]) < 0.10 and abs(ceiling.centre[1] - CEILING_STAIN[1]) < 0.10


def test_the_intervals_carry_the_scale_uncertainty_of_the_room(assessed):
    wall = next(d for d in assessed[0].damage if d.surface.kind == "wall")
    rel = np.hypot(0.03, 0.03) * 1.96  # scale sigma 0.03 and the assumed 3% pose error, at 95%
    assert wall.width.hi >= wall.width.value * (1 + rel) - 1e-9 and wall.width.lo <= wall.width.value * (1 - rel) + 1e-9


def test_the_room_in_the_property_plan_gets_its_regions_moved_with_it(assessed):
    own, moved, _ = assessed
    own_wall, moved_wall = (next(d for d in r.damage if d.surface.kind == "wall") for r in (own, moved))
    assert moved_wall.centre == own_wall.centre and moved_wall.id == own_wall.id  # wall coordinates do not move
    own_c, moved_c = (next(d for d in r.damage if d.surface.kind == "ceiling") for r in (own, moved))
    assert np.allclose(_rot(ANGLE) @ np.array(own_c.centre) + SHIFT, moved_c.centre, atol=1e-3)
    assert moved_c.width == own_c.width and own_c.centre[0] < 4.0  # the room's own copy stays in its own frame


def test_flags_and_scope_follow_the_regions_with_the_neighbour_in_view(assessed):
    own, moved, _ = assessed
    assert "ceiling_stain_source_above" in [f.rule for f in own.concealed_flags]
    assert "stain_on_shared_wall" not in [f.rule for f in own.concealed_flags]  # alone, it has no neighbour
    assert "stain_on_shared_wall" in [f.rule for f in moved.concealed_flags]
    assert {i for s in moved.scope for i in s.damage_ids} == {d.id for d in moved.damage}
    assert any(s.action.startswith("Open up") for s in moved.scope)
    assert RoomPlan.model_validate(moved.model_dump()) == moved


def test_a_clean_room_has_no_damage_flags_or_scope():
    room = _photo_room(_quads(wall_stain=False, ceiling_stain=False))
    assess(room, room.plan, mpp=MPP)
    own = room.plan.rooms[0]
    assert own.damage == [] and own.concealed_flags == [] and own.scope == []


def test_a_room_with_no_frames_is_left_alone():
    room = _photo_room(_quads())
    room.frames = []
    assess(room, room.plan, mpp=MPP)
    assert room.plan.rooms[0].damage == [] and room.plan.rooms[0].flags == []
