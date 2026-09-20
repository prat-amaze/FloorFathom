"""End-to-end checks of the LiDAR tier on synthetic captures with known geometry.

The synthetic generator (synth.py) uses the same camera conventions the pipeline
assumes, so these tests check the geometry and the estimators, not the conventions;
the conventions were settled on the real sample scans (see points.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from synth import Room, Wall, loop, rect, write_capture

from floorfathom.estimate import estimate
from floorfathom.io_lidar import load_scan
from floorfathom.pipeline import run
from floorfathom.points import build_cloud, voxel_downsample
from floorfathom.schema import CapturePlan
from floorfathom.uncertainty import interval, jackknife_scale

YAWS, PITCHES = 8, (-25.0, 10.0, 35.0)


def _estimate(root: Path):
    scan = load_scan(root)
    cloud = build_cloud(scan, target_frames=400, n_chunks=10)
    return estimate(cloud.points, scan.positions[:, [0, 2]])


def _lengths(room) -> list[float]:
    return sorted(e.length for e in room.outline.edges)


def _rotate(pts, deg, about=(0.0, 0.0)):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    out = []
    for x, z in pts:
        dx, dz = x - about[0], z - about[1]
        out.append((about[0] + c * dx - s * dz, about[1] + s * dx + c * dz))
    return out


@pytest.fixture(scope="module")
def rect_capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("rect")
    room = Room(rect(0, 0, 5, 4), path=loop(0, 0, 5, 4, n=16))
    return write_capture(root, room, yaws=YAWS, pitches=PITCHES)


def test_rectangle_walls_area_and_ceiling(rect_capture):
    est = _estimate(rect_capture)
    assert len(est.rooms) == 1
    room = est.rooms[0]
    assert room.area == pytest.approx(20.0, rel=0.01)
    assert room.ceiling_height == pytest.approx(2.6, abs=0.015)
    assert all(e.supported for e in room.outline.edges)
    assert np.allclose(_lengths(room), [4.0, 4.0, 5.0, 5.0], atol=0.03)


def test_rotated_room_needs_no_right_angle_assumption(tmp_path):
    corners = _rotate([(0, 0), (5, 0), (5, 4), (0, 4)], 27, about=(2.5, 2.0))
    walls = [Wall(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    path = _rotate(loop(0, 0, 5, 4, n=16), 27, about=(2.5, 2.0))
    root = write_capture(tmp_path, Room(walls, path=path), yaws=YAWS, pitches=PITCHES)
    room = _estimate(root).rooms[0]
    assert room.area == pytest.approx(20.0, rel=0.015)
    assert np.allclose(_lengths(room), [4.0, 4.0, 5.0, 5.0], atol=0.04)


def test_l_shaped_room(tmp_path):
    poly = [(0, 0), (6, 0), (6, 2), (3, 2), (3, 4), (0, 4)]
    walls = [Wall(poly[i], poly[(i + 1) % 6]) for i in range(6)]
    path = [(x, 1.0) for x in np.linspace(0.8, 5.2, 8)] + [(1.5, z) for z in np.linspace(1.5, 3.4, 6)]
    root = write_capture(tmp_path, Room(walls, path=path), yaws=YAWS, pitches=PITCHES)
    room = _estimate(root).rooms[0]
    assert room.area == pytest.approx(18.0, rel=0.03)
    assert len(room.outline.edges) == 6
    assert np.allclose(_lengths(room), sorted([6, 2, 3, 2, 3, 4]), atol=0.06)


def test_two_rooms_joined_by_a_doorway(tmp_path):
    # 4 x 4 room and 3 x 4 room; a 0.9 m doorway in the shared wall x = 4, centred at z = 2
    walls = rect(0, 0, 4, 4) + rect(4, 0, 7, 4, gaps={"w": (1.55, 2.45)})
    walls = [w for w in walls if not (w.a[0] == 4 and w.b[0] == 4 and abs(w.a[1] - w.b[1]) == 4)]
    walls += [Wall((4, 0), (4, 1.55)), Wall((4, 2.45), (4, 4))]
    path = loop(0, 0, 4, 4, n=10) + [(4.0, 2.0)] + loop(4, 0, 7, 4, n=10)
    root = write_capture(tmp_path, Room(walls, path=path), yaws=YAWS, pitches=PITCHES)
    est = _estimate(root)
    assert len(est.rooms) == 2
    areas = sorted(r.area for r in est.rooms)
    assert areas[0] == pytest.approx(12.0, rel=0.05)
    assert areas[1] == pytest.approx(16.0, rel=0.05)
    widths = [o.width for r in est.rooms for o in r.openings]
    assert widths, "the doorway should be found from at least one room"
    assert all(w == pytest.approx(0.9, abs=0.08) for w in widths)


def test_furniture_does_not_shrink_the_room(tmp_path):
    walls = rect(0, 0, 5, 4)
    walls += [Wall(w.a, w.b, top=0.8) for w in rect(0.2, 0.2, 2.2, 1.1)]
    walls += [Wall(w.a, w.b, top=0.75) for w in rect(3.2, 1.6, 4.0, 2.4)]
    root = write_capture(tmp_path, Room(walls, path=loop(0, 0, 5, 4, n=16)), yaws=YAWS, pitches=PITCHES)
    room = _estimate(root).rooms[0]
    assert room.area == pytest.approx(20.0, rel=0.03)
    assert np.allclose(_lengths(room)[-2:], [5.0, 5.0], atol=0.05)


def test_missing_ceiling_is_reported_as_missing_not_invented(tmp_path):
    room_def = Room(rect(0, 0, 5, 4), has_ceiling=False, path=loop(0, 0, 5, 4, n=16))
    root = write_capture(tmp_path, room_def, yaws=YAWS, pitches=(-25.0, 10.0))
    room = _estimate(root).rooms[0]
    assert room.ceiling_height is None
    assert "ceiling_not_observed" in room.flags


def test_repeatability_gate_same_room_two_captures(tmp_path):
    """Two captures of one room (different noise) agree within 1 cm or 0.5% per wall."""
    room_def = Room(rect(0, 0, 5, 4), path=loop(0, 0, 5, 4, n=16))
    results = []
    for seed in (1, 2):
        root = write_capture(tmp_path / f"s{seed}", room_def, yaws=YAWS, pitches=PITCHES, seed=seed)
        results.append(_estimate(root).rooms[0])
    a, b = results
    for la, lb in zip(_lengths(a), _lengths(b)):
        assert abs(la - lb) <= max(0.01, 0.005 * la)
    assert abs(a.ceiling_height - b.ceiling_height) <= 0.01


def test_pipeline_output_is_valid_deterministic_and_covers_truth(rect_capture, tmp_path):
    plans = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        plans.append(run(rect_capture, out, replicates=8, seed=3, debug=False))
        assert (out / "plan.png").stat().st_size > 5000
    texts = []
    for k in range(2):
        d = json.loads((tmp_path / f"run{k}" / "plan.json").read_text())
        d["diagnostics"].pop("seconds")
        texts.append(json.dumps(d, sort_keys=True))
    assert texts[0] == texts[1], "same input and seed must give the same JSON"

    plan = CapturePlan.model_validate_json((tmp_path / "run0" / "plan.json").read_text())
    (room,) = plan.rooms
    assert room.floor_area.lo <= 20.0 <= room.floor_area.hi
    assert room.ceiling_height.lo <= 2.6 <= room.ceiling_height.hi
    assert sorted(round(w.length.value) for w in room.walls) == [4, 4, 5, 5]
    for w in room.walls:
        truth = 4.0 if round(w.length.value) == 4 else 5.0
        assert w.length.lo <= truth <= w.length.hi
        assert w.length.hi - w.length.lo > 0.02, "an interval this narrow would be overconfident"


def test_plan_polygon_is_counter_clockwise_and_not_mirrored(rect_capture, tmp_path):
    plan = run(rect_capture, tmp_path, replicates=3, debug=False)
    poly = np.array(plan.rooms[0].polygon)
    x, y = poly[:, 0], poly[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0
    # plan y is minus world z; the synthetic room spans world z in [0, 4]
    assert y.max() <= 0.1 and y.min() >= -4.1


def test_interval_ignores_a_lone_outlier_but_flags_a_flipped_shape():
    scale = jackknife_scale(20, 2)
    tight = [15.2, 15.7, 15.0, 15.7, 15.6, 15.5, 28.1, 15.6, 15.2, 15.9, 15.3, 15.6, 15.9, 15.5, 15.5, 14.8, 15.2, 16.0, 15.1, 15.4]
    lo, hi, stable = interval(15.5, tight, 0.05, 0.01, scale)
    assert stable and hi - lo < 6
    flipped = [12.2, 12.0, 12.0, 12.1, 12.1, 12.2, 8.3, 12.3, 13.2, 12.3, 12.3, 8.3, 9.7, 13.1, 8.3, 8.3, 9.3, 12.3, 12.0, 12.2]
    lo, hi, stable = interval(12.1, flipped, 0.05, 0.01, scale)
    assert not stable and lo <= 8.5
    lo, hi, stable = interval(3.0, [], 0.05, 0.01, scale)
    assert not stable and lo >= 0


def test_voxel_downsample_keeps_one_centroid_per_voxel():
    pts = np.array([[0.001, 0, 0], [0.003, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    out = voxel_downsample(pts, 0.02)
    assert len(out) == 2
    assert sorted(out[:, 0].tolist())[0] == pytest.approx(0.002, abs=1e-6)
