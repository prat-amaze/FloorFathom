"""scripts/eval_video.py on plans built from the tape values (truth is known)."""

import importlib.util
from pathlib import Path

from floorfathom.schema import CapturePlan, Diagnostics, Measurement, Opening, RoomPlan, Wall

spec = importlib.util.spec_from_file_location("eval_video", Path(__file__).parents[1] / "scripts" / "eval_video.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def _m(v, unit="m", spread=0.02):
    return Measurement(value=v, lo=v - spread, hi=v + spread, unit=unit, method="test")


def _plan(path: Path, scale: float = 1.0, ceiling: float = 2.79, door: float | None = 0.81) -> Path:
    """A B2 plan (3.61 x 3.015 m) whose walls are `scale` times the tape values."""
    w, d = 3.61 * scale, 3.015 * scale
    pts = [(0, 0), (w, 0), (w, d), (0, d)]
    walls = [
        Wall(id=f"w{i}", start=pts[i], end=pts[(i + 1) % 4], length=_m(l), evidence="wall_points")
        for i, l in enumerate([w, d, w, d])
    ]
    ops = [] if door is None else [
        Opening(id="o0", kind="doorway", wall_id="w0", start=(1, 0), end=(1.9, 0), centre=(1.5, 0), width=_m(door))
    ]
    room = RoomPlan(id="room_0", name="B2", frame="room", polygon=pts, walls=walls, ceiling_height=_m(ceiling),
                    floor_area=_m(w * d, "m2"), openings=ops)
    diag = Diagnostics(bootstrap_replicates=1, seed=0, seconds=0.0, conventions="test")
    path.write_text(CapturePlan(capture="B2 video", tier="video", rooms=[room], diagnostics=diag).model_dump_json())
    return path


def test_plan_equal_to_the_tape_passes_every_row(tmp_path, capsys):
    ev.main([str(_plan(tmp_path / "a.json"))])
    out = capsys.readouterr().out
    assert "walls within 3%: 4/4" in out and "openings within 2 cm (missed and phantom count): 1/1" in out
    assert "FAIL" not in out and "NO |" not in out


def test_a_10_percent_error_and_a_wrong_door_fail(tmp_path, capsys):
    ev.main([str(_plan(tmp_path / "a.json", scale=1.10, ceiling=2.85, door=0.70))])
    out = capsys.readouterr().out
    assert "walls within 3%: 0/4" in out and "openings within 2 cm (missed and phantom count): 0/1" in out
    assert out.count("FAIL") >= 6  # four walls, ceiling, door


def test_repeatability_within_and_beyond_one_percent_of_a_wall(tmp_path, capsys):
    a = _plan(tmp_path / "a.json")
    ev.main(["--repeat", str(a), str(_plan(tmp_path / "b.json", scale=1.002))])  # 0.2%
    assert "walls agreeing: 4/4" in capsys.readouterr().out
    ev.main(["--repeat", str(a), str(_plan(tmp_path / "c.json", scale=1.03))])
    assert "walls agreeing: 0/4" in capsys.readouterr().out


def test_skipped_door_is_not_counted_as_missed(tmp_path, capsys):
    plan = str(_plan(tmp_path / "a.json", door=None))
    ev.main([plan])
    assert "MISSED" in capsys.readouterr().out
    ev.main([plan, "--skip", "b2_door"])
    out = capsys.readouterr().out
    assert "MISSED" not in out and "openings within 2 cm (missed and phantom count): 0/0" in out
