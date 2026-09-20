from floorfathom.render import render_plan
from floorfathom.schema import SCHEMA_VERSION, CapturePlan, Diagnostics, Measurement, RoomPlan


def _m(value, unit="m"):
    return Measurement(value=value, lo=value, hi=value, unit=unit, method="test")


def _room(rid, polygon, area):
    return RoomPlan(
        id=rid, polygon=polygon, walls=[], ceiling_height=_m(None), floor_area=_m(area, "m2"), openings=[]
    )


def test_render_survives_a_room_with_no_area_and_a_room_with_no_polygon(tmp_path):
    square = [(0, 0), (3, 0), (3, 3), (0, 3)]
    rooms = [_room("full", square, 9.0), _room("open_outline", square, None), _room("failed", [], None)]
    diag = Diagnostics(bootstrap_replicates=0, seed=0, seconds=0.0, conventions="test")
    plan = CapturePlan(schema_version=SCHEMA_VERSION, capture="c", tier="photo", rooms=rooms, diagnostics=diag)
    render_plan(plan, tmp_path / "plan.png")
    assert (tmp_path / "plan.png").stat().st_size > 0
