"""The ruler option of the photo and video tiers must not break a LiDAR run (a tester may pass it by habit)."""

import json

from synth import Room, loop, rect, write_capture

from floorfathom.cli import main


def test_reference_length_is_accepted_and_ignored_by_the_lidar_tier(tmp_path):
    root = write_capture(tmp_path / "scan", Room(rect(0, 0, 5, 4), path=loop(0, 0, 5, 4, n=16)), yaws=8, pitches=(-25.0, 10.0, 35.0))
    out = tmp_path / "out"
    code = main(["plan", str(root), "--out", str(out), "--reference-length-cm", "31.6", "--bootstrap", "2", "--no-debug"])
    assert code == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["tier"] == "lidar" and len(plan["rooms"]) == 1
    assert any("reference-length-cm was ignored" in n for n in plan["diagnostics"]["notes"])
