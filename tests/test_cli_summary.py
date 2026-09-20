from floorfathom.cli import _summary
from floorfathom.schema import CapturePlan, Diagnostics, Measurement, RoomPlan


def _m(v, unit="m"):
    return Measurement(value=v, lo=None if v is None else v * 0.9, hi=None if v is None else v * 1.1, unit=unit, method="t")


def test_summary_survives_a_room_without_area_or_ceiling_and_reports_damage_counts():
    unmeasured = RoomPlan(id="a", polygon=[], walls=[], ceiling_height=_m(None), floor_area=_m(None, "m2"), openings=[])
    measured = RoomPlan(id="b", polygon=[], walls=[], ceiling_height=_m(2.6), floor_area=_m(12.0, "m2"), openings=[])
    plan = CapturePlan(
        capture="c", tier="photo", rooms=[unmeasured, measured],
        diagnostics=Diagnostics(bootstrap_replicates=0, seed=0, seconds=1.0, conventions="t"),
    )
    text = _summary(plan)
    assert "a: area n/a, ceiling n/a" in text
    assert "b: area 12.00 [10.80, 13.20] m2, ceiling 2.600 [2.340, 2.860] m" in text
