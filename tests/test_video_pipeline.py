"""Video pipeline decisions that need no heavy models: intervals, failure paths, input handling."""

from __future__ import annotations

import numpy as np
import pytest

from floorfathom import video_pipeline as vp
from floorfathom.io_video import Keyframes
from floorfathom.pipeline import detect_tier
from floorfathom.schema import CapturePlan, Measurement
from floorfathom.sfm import SfmResult


def _m(value, lo, hi):
    return Measurement(value=value, lo=lo, hi=hi, unit="m", method="test")


def test_widen_adds_relative_error_in_quadrature():
    m = _m(4.0, 3.9, 4.1)  # half-width 0.1
    vp._widen(m, 0.15)  # 0.15 * 4 = 0.6
    half = float(np.hypot(0.1, 0.6))
    assert m.lo == pytest.approx(4.0 - half, abs=1e-3) and m.hi == pytest.approx(4.0 + half, abs=1e-3)


def test_widen_leaves_missing_values_alone_and_never_goes_negative():
    none = Measurement(value=None, lo=None, hi=None, unit="m", method="test", note="not measured")
    vp._widen(none, 0.5)
    assert none.value is None and none.lo is None
    small = _m(0.5, 0.49, 0.51)
    vp._widen(small, 2.0)
    assert small.lo == 0.0


def test_scale_uncertainty_never_drops_below_the_floor():
    steady = {i: 0.4 for i in range(100)}  # perfectly consistent frames
    assert vp._scale_rel_sigma(steady) == pytest.approx(vp.SCALE_FLOOR_REL, rel=1e-3)
    noisy = {i: 0.4 * (1 + 0.3 * (-1) ** i) for i in range(4)}
    assert vp._scale_rel_sigma(noisy) > vp.SCALE_FLOOR_REL


def _keyframes(tmp_path):
    z = np.zeros(3)
    return Keyframes(tmp_path, ["0.jpg"] * 3, z, z, z, (720, 1280), 30.0, tmp_path / "clip.MOV")


def _fake_sfm(fraction: float) -> SfmResult:
    n = 10
    reg = np.zeros(n, dtype=bool)
    reg[: int(fraction * n)] = True
    return SfmResult(
        registered=reg, centers=np.zeros((n, 3)), cam_to_world=np.repeat(np.eye(3)[None], n, axis=0),
        intrinsics=(700.0, 700.0, 360.0, 640.0), image_size=(720, 1280), points=np.zeros((1, 3)),
        track_length=np.ones(1, dtype=int), point_error=np.zeros(1), n_models=2, mean_reprojection_px=0.7,
        flags=["sfm_split_into_several_models"],
    )


def _clip(tmp_path):
    clip = tmp_path / "room.MOV"
    clip.write_bytes(b"not a real movie")
    return clip


def test_failed_reconstruction_gives_an_empty_plan_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(vp, "extract_keyframes", lambda *a, **k: _keyframes(tmp_path))
    monkeypatch.setattr(vp, "run_sfm", lambda *a, **k: None)
    plan = vp.run_video(_clip(tmp_path), tmp_path / "out")
    assert plan.rooms == [] and plan.tier == "video" and "sfm_failed" in plan.diagnostics.notes
    again = CapturePlan.model_validate_json((tmp_path / "out" / "plan.json").read_text())  # valid to the schema
    assert again.capture == "room"


def test_too_few_registered_frames_gives_no_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(vp, "extract_keyframes", lambda *a, **k: _keyframes(tmp_path))
    monkeypatch.setattr(vp, "run_sfm", lambda *a, **k: _fake_sfm(0.3))
    plan = vp.run_video(_clip(tmp_path), tmp_path / "out")
    assert plan.rooms == []
    assert "sfm_registered_too_few_frames" in plan.diagnostics.notes
    assert "sfm_split_into_several_models" in plan.diagnostics.notes  # the reconstruction's own flags are kept


def _fake_clip_plans(monkeypatch, plans):
    """run_clip replaced by canned per-clip plans, keyed by the clip's name."""
    monkeypatch.setattr(vp, "run_clip", lambda clip, out, **k: plans[clip.stem])


def _folder(tmp_path, names):
    for n in names:
        (tmp_path / f"{n}.MOV").write_bytes(b"x")
    return tmp_path


def test_a_folder_of_clips_gives_one_stitched_plan_with_unique_ids(tmp_path, monkeypatch):
    from test_stitch import WALL, _box, _in_own_frame, _one_room_plan

    hub = _box("hub", 0, 0, 6, 4, [("e", (6.0, 2.0), 0.9)]).model_copy(update={"frame": "room"})
    east = _in_own_frame(_box("east", 6 + WALL, 0.5, 4, 3, [("d", (6 + WALL, 2.0), 0.9)]), 0.7, (5.0, -2.0))
    _fake_clip_plans(monkeypatch, {"Hall": _one_room_plan(hub, "Hall"), "B1": _one_room_plan(east, "B1")})
    plan = vp.run_video(_folder(tmp_path, ["Hall", "B1"]), tmp_path / "out")
    assert plan.tier == "video" and plan.capture == tmp_path.name
    assert sorted(r.id for r in plan.rooms) == ["room_0", "room_1"]
    st = plan.stitching
    assert st is not None and st.unplaced == [] and st.overlaps == []
    assert {(ln.room, ln.other) for ln in st.links} == {("room_0", "room_1"), ("room_1", "room_0")}
    assert CapturePlan.model_validate_json((tmp_path / "out" / "plan.json").read_text()) == plan


def test_a_clip_without_a_room_is_listed_and_the_others_are_still_stitched(tmp_path, monkeypatch):
    from test_stitch import _box, _one_room_plan

    ok = _one_room_plan(_box("hub", 0, 0, 6, 4, []).model_copy(update={"frame": "room"}), "A")
    empty = vp._no_plan("B", 1.0, 0, ["sfm_registered_too_few_frames"])
    _fake_clip_plans(monkeypatch, {"A": ok, "B": empty})
    plan = vp.run_video(_folder(tmp_path, ["A", "B"]), tmp_path / "out")
    assert [r.id for r in plan.rooms] == ["room_0"]
    assert any(n.startswith("B: no room") and "sfm_registered_too_few_frames" in n for n in plan.diagnostics.notes)


def test_a_folder_where_no_clip_gave_a_room_says_so(tmp_path, monkeypatch):
    _fake_clip_plans(monkeypatch, {n: vp._no_plan(n, 1.0, 0, ["sfm_failed"]) for n in ("A", "B")})
    plan = vp.run_video(_folder(tmp_path, ["A", "B"]), tmp_path / "out")
    assert plan.rooms == [] and "no_clip_gave_a_room" in plan.diagnostics.notes


def test_a_folder_without_clips_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        vp.run_video(tmp_path, tmp_path / "out")


def test_a_single_movie_file_is_detected_as_video(tmp_path):
    assert detect_tier(_clip(tmp_path)) == "video"


def test_cache_replays_only_for_the_same_settings(tmp_path):
    cache, calls = tmp_path / "work" / "x.pkl", []

    def compute():
        calls.append(1)
        return {"value": len(calls)}

    assert vp._load_or_run(cache, "a", compute) == {"value": 1}
    assert vp._load_or_run(cache, "a", compute) == {"value": 1} and len(calls) == 1  # replayed
    assert vp._load_or_run(cache, "b", compute) == {"value": 2}  # other settings: recomputed
    cache.write_bytes(b"corrupt")
    assert vp._load_or_run(cache, "b", compute) == {"value": 3}  # unreadable cache: recomputed


def test_a_failed_step_is_never_cached(tmp_path):
    cache = tmp_path / "x.pkl"
    assert vp._load_or_run(cache, "a", lambda: None) is None
    assert not cache.exists()


def _anchor(scale, flags=(), rel=0.02):
    from floorfathom.anchor import Anchor

    return Anchor(length_sfm=0.2 / scale, scale=scale, rel_sigma=rel, n_frames=30, max_ray_angle_deg=20.0, residual_px=1.0, flags=list(flags))


def _use_strip(monkeypatch, anchor, n_obs=30):
    monkeypatch.setattr(vp, "track_strip", lambda *a, **k: [object()] * n_obs)
    monkeypatch.setattr(vp, "estimate_anchor", lambda *a, **k: anchor)


def test_strip_scale_is_used_when_found_and_plausible(monkeypatch):
    _use_strip(monkeypatch, _anchor(0.45))
    scale, rel, method, flags = vp._choose_scale(None, None, depth_scale=0.5, depth_rel=0.15)  # strip 10% under the depth model
    assert (scale, method, flags) == (0.45, "reference_object", [])
    assert rel == pytest.approx(float(np.hypot(0.02, vp.STRIP_LENGTH_REL)))  # far tighter than the depth model's 0.15


def test_without_a_strip_the_depth_model_is_used_and_says_why(monkeypatch):
    _use_strip(monkeypatch, None, n_obs=0)
    assert vp._choose_scale(None, None, 0.5, 0.15) == (0.5, 0.15, "monocular_depth", ["reference_strip_not_found"])


def test_a_flagged_strip_is_not_trusted(monkeypatch):
    _use_strip(monkeypatch, _anchor(0.45, flags=["anchor_weak_baseline"]))
    scale, rel, method, flags = vp._choose_scale(None, None, 0.5, 0.15)
    assert (scale, method) == (0.5, "monocular_depth") and flags == ["reference_strip_rejected", "anchor_weak_baseline"]


def test_a_strip_far_from_the_depth_model_is_taken_for_another_object(monkeypatch):
    _use_strip(monkeypatch, _anchor(0.1))  # 5x smaller than the depth model: a door frame, not the strip
    scale, _, method, flags = vp._choose_scale(None, None, 0.5, 0.15)
    assert method == "monocular_depth" and scale == 0.5 and "reference_strip_implausible" in flags


def test_the_measured_length_of_the_ruler_is_passed_on(monkeypatch):
    seen = []
    monkeypatch.setattr(vp, "track_strip", lambda *a, **k: [object()] * 30)
    monkeypatch.setattr(vp, "estimate_anchor", lambda sfm, obs, length, **k: seen.append(length) or _anchor(0.45))
    vp._choose_scale(None, None, 0.5, 0.15)
    vp._choose_scale(None, None, 0.5, 0.15, reference_length_m=0.5)
    assert seen == [vp.REFERENCE_LENGTH_M, 0.5]


def test_the_alignment_file_maps_sfm_points_into_the_plan_frame(tmp_path):
    import json

    rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    vp.write_alignment(tmp_path / "work" / "alignment.json", rot, 0.5, -1.2, (720, 1280))
    a = json.loads((tmp_path / "work" / "alignment.json").read_text())
    p = np.array([2.0, 3.0, 4.0])
    q = (p @ np.array(a["rotation"]).T) * a["scale"]
    assert q.tolist() == pytest.approx(((p @ rot.T) * 0.5).tolist()) and a["floor_y"] == -1.2 and a["image_size"] == [720, 1280]
