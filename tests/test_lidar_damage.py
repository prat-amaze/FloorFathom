"""LiDAR tier end to end with damage: a synthetic scan whose RGB video shows a painted room (a water stain on one
wall at a known place), run through ``run`` with drift correction on, then read back from the plan."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from synth import CX, CY, FLOOR_Y, FX, FY, SCALE, Room, loop, rect, write_capture
from test_surfaces import Quad, _plain, _stained

from floorfathom.io_lidar import load_scan
from floorfathom.pipeline import run
from floorfathom.schema import CapturePlan

X1, Z1, HEIGHT = 5.0, 4.0, 2.6
RW, RH = 320, 240  # rendered picture, upscaled into the 1920 x 1440 video


def _quads(stain: bool):
    up = [0, 1, 0]
    return [
        Quad([0, FLOOR_Y, 0], [1, 0, 0], up, X1, HEIGHT, _stained(1.8, 1.2, 0.15, 0.2) if stain else _plain),  # the wall z = 0
        Quad([X1, FLOOR_Y, 0], [0, 0, 1], up, Z1, HEIGHT, _plain),
        Quad([X1, FLOOR_Y, Z1], [-1, 0, 0], up, X1, HEIGHT, _plain),
        Quad([0, FLOOR_Y, Z1], [0, 0, -1], up, Z1, HEIGHT, _plain),
        Quad([0, FLOOR_Y, 0], [1, 0, 0], [0, 0, 1], X1, Z1, lambda s, h: np.tile([0.60, 0.58, 0.55], (len(s), 1))),
        Quad([0, FLOOR_Y + HEIGHT, 0], [1, 0, 0], [0, 0, 1], X1, Z1, lambda s, h: np.tile([0.90, 0.90, 0.88], (len(s), 1))),
    ]


def _paint(root: Path, stain: bool) -> None:
    """Replace the capture's blank rgb.mp4 by pictures rendered from the odometry poses."""
    scan = load_scan(root)
    quads = _quads(stain)
    uu, vv = np.meshgrid(np.arange(RW), np.arange(RH))
    k = SCALE and 1920 / RW
    d_cam = np.stack([(uu - CX * SCALE / k) / (FX * SCALE / k), (vv - CY * SCALE / k) / (FY * SCALE / k), np.ones_like(uu, float)], -1).reshape(-1, 3)
    rng = np.random.default_rng(1)
    vw = cv2.VideoWriter(str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920, 1440))
    from scipy.spatial.transform import Rotation

    for i in range(len(scan)):
        R = Rotation.from_quat(scan.quats[i]).as_matrix()
        o, D = scan.positions[i], d_cam @ R.T
        best, colour = np.full(len(D), np.inf), np.zeros((len(D), 3))
        for q in quads:
            t, s, h = q.hit(o, D)
            closer = t < best
            if closer.any():
                colour[closer] = q.colour(s[closer], h[closer])
                best = np.where(closer, t, best)
        img = np.clip(colour + rng.normal(0, 0.004, colour.shape), 0, 1).reshape(RH, RW, 3)
        vw.write(cv2.resize((img[..., ::-1] * 255).astype(np.uint8), (1920, 1440), interpolation=cv2.INTER_LINEAR))
    vw.release()


def _capture(tmp_path_factory, name: str, stain: bool) -> Path:
    root = tmp_path_factory.mktemp(name)
    write_capture(root, Room(rect(0, 0, X1, Z1), path=loop(0, 0, X1, Z1, n=10)), yaws=8, pitches=(-25.0, 10.0, 35.0))
    _paint(root, stain)
    return root


@pytest.fixture(scope="module")
def stained(tmp_path_factory):
    root = _capture(tmp_path_factory, "stained", True)
    out = tmp_path_factory.mktemp("out_stained")
    return run(root, out, tier="lidar", replicates=4, debug=True), out


def test_the_stain_is_found_on_the_right_wall_at_the_right_place(stained):
    plan, _out = stained
    room = plan.rooms[0]
    stains = [d for d in room.damage if d.damage_class == "water_stain"]
    assert len(room.damage) == 1 and len(stains) == 1 and stains[0].surface.kind == "wall"  # nothing else on the room's six surfaces
    d = stains[0]
    wall = next(w for w in room.walls if w.id == d.surface.id)
    a, b = np.array(wall.start), np.array(wall.end)
    p = a + (b - a) / np.linalg.norm(b - a) * d.centre[0]  # plan point on the wall
    assert np.hypot(p[0] - 1.8, p[1] - 0.0) < 0.15 or np.hypot(p[0] - 1.8, p[1] + 0.0) < 0.15
    assert abs(d.centre[1] - 1.2) < 0.12
    assert abs(d.width.value - 0.30) < 0.1 and abs(d.height.value - 0.40) < 0.1
    assert any(s.damage_ids == [d.id] for s in room.scope)


def test_the_plan_with_damage_is_valid_states_its_assumptions_and_writes_the_shared_files(stained):
    plan, out = stained
    assert any("our own definitions" in n for n in plan.diagnostics.notes)
    assert any("Seconds per stage" in n for n in plan.diagnostics.notes)
    assert CapturePlan.model_validate(plan.model_dump()) == plan
    assert (out / "plan.png").stat().st_size > 0 and (out / "plan.json").is_file()
    per_room = sorted((out / "rooms").glob("*.json"))
    assert [p.stem for p in per_room] == [r.id for r in plan.rooms]
    one = CapturePlan.model_validate_json(per_room[0].read_text())
    assert one.rooms == [plan.rooms[0]] and one.stitching is None
    assert any((out / "debug" / "damage").glob("*.png"))
