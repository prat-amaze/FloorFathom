import importlib.util
from pathlib import Path

from floorfathom.schema import CapturePlan, Diagnostics, Measurement, RoomPlan, Wall

spec = importlib.util.spec_from_file_location(
    "eval_photo_geometry",
    Path(__file__).parents[1] / "scripts" / "eval_photo_geometry.py",
)
eg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg)

score_geometry = eg.score_geometry


def _plan():
    room = RoomPlan(id="room_0", name="synth", frame="room",
                    polygon=[(0, 0), (4, 0), (4, 3), (0, 3)],
                    walls=[Wall(id="w0", start=(0, 0), end=(4, 0),
                                length=Measurement(value=4.0, lo=3.8, hi=4.2, unit="m", method="m", note=None),
                                evidence="wall_points")],
                    openings=[], ceiling_height=Measurement(value=2.6, lo=2.5, hi=2.7, unit="m", method="m", note=None),
                    floor_area=Measurement(value=12.0, lo=11.0, hi=13.0, unit="m2", method="m", note=None))
    return CapturePlan(capture="synth", tier="photo", rooms=[room],
                       diagnostics=Diagnostics(bootstrap_replicates=0, seed=0, seconds=0.0, conventions=""))


def test_score_geometry_against_known_synthetic_truth():
    truth = {"synth": {"walls": {"w0": 4.0}, "ceiling_height": 2.6, "floor_area": 12.0}}
    scores = score_geometry(_plan(), truth)
    assert scores["wall_length_error"][0] == 0.0
    assert scores["ceiling_height_error"] == 0.0
    assert scores["floor_area_error"] == 0.0
