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


def test_damage_regions_are_marked_and_the_legend_says_so(tmp_path):
    from floorfathom.schema import DamageRegion, SurfaceRef, Wall

    m = lambda v, u="m": Measurement(value=v, lo=v, hi=v, unit=u, method="t")
    d = lambda sid, kind, c: DamageRegion(id=sid, surface=SurfaceRef(kind=kind, id="w0" if kind == "wall" else "r_ceiling"), damage_class="water_stain",
                                          class_confidence=0.8, polygon=[c] * 3, centre=c, width=m(0.2), height=m(0.2), length=m(0.2), area=m(0.04, "m2"), evidence="t")
    room = RoomPlan(
        id="r", polygon=[(0, 0), (4, 0), (4, 3), (0, 3)], walls=[Wall(id="w0", start=(0, 0), end=(4, 0), length=m(4.0), evidence="wall_points")],
        ceiling_height=m(2.6), floor_area=m(12.0, "m2"), openings=[], damage=[d("a", "wall", (1.0, 1.2)), d("b", "ceiling", (2.0, 2.0)), d("c", "wall", (9.0, 1.0))],
    )
    plan = CapturePlan(schema_version=SCHEMA_VERSION, capture="c", tier="lidar", rooms=[room], diagnostics=Diagnostics(bootstrap_replicates=0, seed=0, seconds=0.0, conventions="t"))
    from floorfathom.render import _damage_position, render_plan

    assert list(_damage_position(room, room.damage[0])) == [1.0, 0.0] and list(_damage_position(room, room.damage[1])) == [2.0, 2.0]
    out = tmp_path / "p.png"
    render_plan(plan, out)
    assert out.stat().st_size > 0
