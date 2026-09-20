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


def test_a_folder_of_several_clips_is_refused_for_now(tmp_path):
    (tmp_path / "a.MOV").write_bytes(b"x")
    (tmp_path / "b.MOV").write_bytes(b"x")
    with pytest.raises(NotImplementedError):
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
