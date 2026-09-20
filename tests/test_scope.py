from floorfathom.scope import ACTIONS, INSPECT, scope_items
from floorfathom.schema import ConcealedFlag, DamageRegion, Measurement, RoomPlan, SurfaceRef

CLASSES = list(ACTIONS)


def _m(v, unit="m"):
    return Measurement(value=v, lo=v * 0.9, hi=v * 1.1, unit=unit, method="test")


def _dmg(did, cls, sid="room_0_w0", length=0.5, area=0.04):
    return DamageRegion(
        id=did, surface=SurfaceRef(kind="wall", id=sid), damage_class=cls, class_confidence=0.8, polygon=[(0, 0)] * 3, centre=(1.0, 1.0),
        width=_m(0.1), height=_m(0.1), length=_m(length), area=_m(area, "m2"), evidence="test",
    )


def _room():
    return RoomPlan(id="room_0", polygon=[], walls=[], ceiling_height=_m(2.6), floor_area=_m(12.0, "m2"), openings=[])


def test_every_damage_class_has_a_keyed_scope_line():
    damage = [_dmg(f"d{i}", c) for i, c in enumerate(CLASSES)]
    items = scope_items(_room(), damage, [])
    assert len(items) == len(CLASSES)
    for it, d in zip(items, damage):
        assert it.surface == d.surface and it.damage_ids == [d.id] and it.action


def test_crack_is_measured_in_metres_and_the_rest_in_square_metres():
    items = {i.damage_ids[0]: i for i in scope_items(_room(), [_dmg("c", "structural_crack"), _dmg("s", "water_stain")], [])}
    assert items["c"].quantity.unit == "m" and items["c"].quantity.value == 0.5
    assert items["s"].quantity.unit == "m2" and items["s"].quantity.value == 0.04
    assert items["s"].quantity.lo == 0.04 * 0.9 and items["s"].quantity.hi == 0.04 * 1.1  # the interval is kept


def test_flagged_surface_gets_one_inspection_item_with_the_summed_area():
    damage = [_dmg("a", "mould", area=0.04), _dmg("b", "water_stain", area=0.06), _dmg("c", "hole_or_impact", "room_0_w1")]
    flags = [
        ConcealedFlag(id="f0", rule="r1", surface=damage[0].surface, damage_ids=["a"], reason="x"),
        ConcealedFlag(id="f1", rule="r2", surface=damage[1].surface, damage_ids=["b"], reason="y"),
    ]
    items = scope_items(_room(), damage, flags)
    inspect = [i for i in items if i.action == INSPECT]
    assert len(inspect) == 1 and inspect[0].surface.id == "room_0_w0" and set(inspect[0].damage_ids) == {"a", "b"}
    assert abs(inspect[0].quantity.value - 0.10) < 1e-9
    assert len({i.id for i in items}) == len(items)


def test_no_damage_no_scope():
    assert scope_items(_room(), [], []) == []
